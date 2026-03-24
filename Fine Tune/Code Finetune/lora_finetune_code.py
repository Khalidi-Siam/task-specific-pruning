# lora_finetune_code.py
# Normal SFT (no chat template): "### Instruction" + "### Response"
# - Reads JSONL: {"instruction": "...", "response": "..."}
# - Builds [PROMPT][RESPONSE+EOS], masks prompt tokens in labels
# - LoRA BF16 finetune for Qwen2.5-Coder-* pruned folders
# - OUTPUT_DIR is set per model by the batch runner
# - Exposes run() for batch runner
#
# Speed focus:
# - QuickEM is disabled by default (generation is expensive)
# - Eval/save cadence set in batch runner
# - Gradient checkpointing is optional (off = faster, on = lower VRAM)

import os, json, math, random, glob, inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# --- Fast kernels ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

# ===================== CONFIG (batch runner overrides MODEL_DIR/OUTPUT_DIR) =====================
MODEL_DIR   = os.path.join(".", "MODEL")     # overwritten
OUTPUT_DIR  = os.path.join("adapters", "_tmp")  # overwritten

TRAIN_FILE  = os.path.join("Code FineTune", "dataset_splits_code", "train.jsonl")
VAL_FILE    = os.path.join("Code FineTune", "dataset_splits_code", "val.jsonl")

# Requested by you (batch runner will override too)
MAX_LENGTH       = 1024
BATCH_SIZE       = 2
GRAD_ACCUM       = 8
EPOCHS           = 2

LR               = 2e-4
WARMUP_RATIO     = 0.03
WEIGHT_DECAY     = 0.01
LOGGING_STEPS    = 20
EVAL_STEPS       = 400
SAVE_STEPS       = 400
SAVE_TOTAL_LIMIT = 2
SEED             = 42

USE_BF16          = True
USE_FP16          = False
GRADIENT_CKPT     = False   # OFF by default for speed on A6000; batch runner can enable
USE_TORCH_COMPILE = False   # keep False on Windows

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05

TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"

# QuickEM (generate-based metric) is expensive; default OFF for speed.
ENABLE_QUICK_EM    = False
INTERIM_EM_SAMPLE  = 16
INTERIM_MAX_NEW    = 128
# ================================================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    cands = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not cands:
        return None
    def step_or_mtime(p):
        base = os.path.basename(p)
        try:
            return (1, int(base.split("-")[-1]))
        except Exception:
            return (0, os.path.getmtime(p))
    cands.sort(key=step_or_mtime, reverse=True)
    return cands[0]

def default_prompt(instruction: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Response:\n"
    )

def normalize_for_em(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = "\n".join([ln.rstrip() for ln in s.splitlines()]).strip()
    return s

def clear_gpu():
    import gc, time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    time.sleep(0.2)

class SupervisedJsonl(Dataset):
    """
    JSONL rows: {"instruction": "...", "response": "..."}
    Creates [PROMPT][RESPONSE + EOS], masks prompt tokens in loss.
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int):
        super().__init__()
        raw = read_jsonl(path)
        good, bad = [], 0
        for r in raw:
            ins = r.get("instruction", None)
            rsp = r.get("response", None)
            if isinstance(ins, str) and isinstance(rsp, str) and ins.strip() and rsp.strip():
                good.append({"instruction": ins, "response": rsp})
            else:
                bad += 1
        if bad:
            print(f"[data] Skipped {bad} malformed row(s) in {os.path.basename(path)}")
        self.rows = good
        self.tok = tokenizer
        self.maxlen = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        ex = self.rows[idx]
        prompt = default_prompt(ex["instruction"])
        answer = ex["response"]

        q_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]

        eos_id = self.tok.eos_token_id
        if eos_id is not None and (len(a_ids) == 0 or a_ids[-1] != eos_id):
            a_ids = a_ids + [eos_id]

        maxlen = self.maxlen
        min_prompt_tokens = 1

        # Keep as much answer as possible (tail), reserve at least 1 token for prompt
        keep_answer = min(len(a_ids), maxlen - min_prompt_tokens)
        a_ids = a_ids[-keep_answer:]

        room_for_prompt = maxlen - keep_answer
        room_for_prompt = max(room_for_prompt, min_prompt_tokens)
        q_ids = q_ids[-room_for_prompt:]

        input_ids = q_ids + a_ids
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()

        # mask prompt tokens
        for i in range(len(q_ids)):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id

        def _tensor_or_dummy(f, key, default_val):
            t = f.get(key, None)
            if t is None:
                return torch.tensor([default_val], dtype=torch.long)
            return t

        input_list = [ _tensor_or_dummy(f, "input_ids", pad_id) for f in features ]
        mask_list  = [ _tensor_or_dummy(f, "attention_mask", 0) for f in features ]
        label_list = [ _tensor_or_dummy(f, "labels", -100) for f in features ]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        if self.pad_to_multiple_of:
            for k, pad_val in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
                L = batch[k].size(1)
                pad_len = (-L) % self.pad_to_multiple_of
                if pad_len:
                    pad = torch.full((batch[k].size(0), pad_len), pad_val, dtype=batch[k].dtype)
                    batch[k] = torch.cat([batch[k], pad], dim=1)

        return batch

class BestEvalLossCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = None
        self.last_eval_loss = None
        self.last_val_em_exact = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if "eval_loss" in metrics:
            self.last_eval_loss = float(metrics["eval_loss"])
            if self.best_eval_loss is None or self.last_eval_loss < self.best_eval_loss:
                self.best_eval_loss = self.last_eval_loss
        if "val_em_exact" in metrics and metrics["val_em_exact"] is not None:
            self.last_val_em_exact = float(metrics["val_em_exact"])

@torch.no_grad()
def quick_exact_match(model, tok, dataset: SupervisedJsonl, num_eval=16, max_new_tokens=128, device="cuda"):
    model.eval()
    old_pad_side = getattr(tok, "padding_side", "right")
    old_trunc_side = getattr(tok, "truncation_side", "right")
    tok.padding_side = "left"
    tok.truncation_side = "left"
    try:
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        n = min(num_eval, len(dataset))
        idxs = list(range(len(dataset)))
        random.Random(1234).shuffle(idxs)
        idxs = idxs[:n]

        correct = 0
        model.to(device)

        for i in idxs:
            ins = dataset.rows[i]["instruction"]
            gt  = normalize_for_em(dataset.rows[i]["response"])
            prompt = default_prompt(ins)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            pred = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
            pred = normalize_for_em(pred)
            if pred == gt:
                correct += 1

        em = (correct / n) * 100 if n else 0.0
        return {"val_em_exact": em, "evaluated": n}
    finally:
        tok.padding_side = old_pad_side
        tok.truncation_side = old_trunc_side

class QuickEMCallback(TrainerCallback):
    def __init__(self, tok, val_ds: SupervisedJsonl, sample_n: int, max_new: int):
        self.tok = tok
        self.val_ds = val_ds
        self.sample_n = sample_n
        self.max_new = max_new

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        em = quick_exact_match(
            model, self.tok, self.val_ds,
            num_eval=self.sample_n,
            max_new_tokens=self.max_new,
            device=device
        )
        if metrics is not None and isinstance(metrics, dict):
            metrics.update(em)
        print(f"[eval] quick_exact_match={em['val_em_exact']:.2f}% (n={em['evaluated']})")

def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    if "evaluation_strategy" not in supported:
        for k in ["evaluation_strategy", "eval_steps",
                  "load_best_model_at_end", "metric_for_best_model", "greater_is_better"]:
            kwargs.pop(k, None)
    if "save_strategy" not in supported:
        kwargs.pop("save_strategy", None)

    allowed = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(k for k in kwargs.keys() if k not in supported))
    if dropped:
        print(f"[compat] Ignoring unsupported TrainingArguments keys: {dropped}")
    return TrainingArguments(**allowed)

def run():
    main()

def main():
    for p in (MODEL_DIR, TRAIN_FILE, VAL_FILE):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    print("=== LoRA Fine-tune (Code, Normal SFT) ===")
    print("MODEL_DIR :", MODEL_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("VAL_FILE  :", VAL_FILE)
    print("BF16:", USE_BF16, "FP16:", USE_FP16, "Grad-CKPT:", GRADIENT_CKPT)
    print(f"LoRA r/alpha/dropout: {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print("MAX_LENGTH:", MAX_LENGTH, "EPOCHS:", EPOCHS)
    print("BATCH_SIZE:", BATCH_SIZE, "GRAD_ACCUM:", GRAD_ACCUM)
    print("QuickEM:", ENABLE_QUICK_EM)
    print("========================================")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = MAX_LENGTH

    load_kwargs = dict(device_map="auto")
    if USE_BF16:
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif USE_FP16:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kwargs)
    model.config.use_cache = False

    targets = [m.strip() for m in TARGET_MODULES.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Only do the expensive input-grad plumbing if using gradient checkpointing.
    if GRADIENT_CKPT:
        # ✅ Critical fix for grad checkpointing + LoRA
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            emb = model.get_input_embeddings()
            emb.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))

        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[speed] torch.compile enabled.")
        except Exception as e:
            print(f"[speed] torch.compile unavailable: {e}")

    train_ds = SupervisedJsonl(TRAIN_FILE, tok, MAX_LENGTH)
    val_ds   = SupervisedJsonl(VAL_FILE, tok, MAX_LENGTH)
    collator = DataCollator(tokenizer=tok)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM)))
    total_steps = steps_per_epoch * EPOCHS
    print(f"[info] steps/epoch≈{steps_per_epoch} total_steps≈{total_steps}")

    best_cb = BestEvalLossCallback()

    # Note: setting TrainingArguments.gradient_checkpointing=True is fine,
    # but we also directly enabled it above to ensure the LoRA+input-grad fix is applied first.
    targs = build_training_args(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,

        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        lr_scheduler_type="cosine",
        bf16=USE_BF16,
        fp16=(USE_FP16 and not USE_BF16),
        gradient_checkpointing=GRADIENT_CKPT,

        report_to="none",
        logging_dir=os.path.join(OUTPUT_DIR, "tb"),
        optim="adamw_torch",
        save_safetensors=True,

        max_grad_norm=1.0,
        group_by_length=True,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,

        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    callbacks = [best_cb]
    if ENABLE_QUICK_EM:
        callbacks.append(QuickEMCallback(tok, val_ds, sample_n=INTERIM_EM_SAMPLE, max_new=INTERIM_MAX_NEW))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        callbacks=callbacks,
    )

    resume_ckpt = find_latest_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"[resume] Found checkpoint: {resume_ckpt} — resuming.")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("[resume] No checkpoint found — starting fresh.")
        trainer.train()

    print("[save] Saving adapter + tokenizer…")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    meta = {
        "train_file": TRAIN_FILE,
        "val_file": VAL_FILE,
        "model_dir": MODEL_DIR,
        "output_dir": OUTPUT_DIR,
        "bf16": USE_BF16,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": [m.strip() for m in TARGET_MODULES.split(",") if m.strip()],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "total_steps_estimate": total_steps,
        "prompt_format": "### Instruction / ### Response (no chat template)",
        "best_eval_loss": best_cb.best_eval_loss,
        "last_eval_loss": best_cb.last_eval_loss,
        "last_val_em_exact": best_cb.last_val_em_exact,
        "quick_em_enabled": ENABLE_QUICK_EM,
    }
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    clear_gpu()
    print("Done.")
