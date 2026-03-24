# lora_finetune.py — Option B (\boxed{} training) with EOS taught  [MODIFIED + CSV + FAST EVAL + ROBUST IO + SDPA]
import os, json, math, random, re, glob, inspect, csv
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
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model

# --- Fast kernels ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True  # speed on fixed shapes

# ======== CONFIG ========
MODEL_DIR           = os.path.join("Pruned Model", "Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-7B-Instruct-pruned-35.135p")
TRAIN_FILE          = os.path.join("dataset_splits_boxed", "train.jsonl")
VAL_FILE            = os.path.join("dataset_splits_boxed", "val.jsonl")
OUTPUT_DIR          = os.path.join(".", "lora_out")

MAX_LENGTH          = 1024
BATCH_SIZE          = 2
GRAD_ACCUM          = 4
EPOCHS              = 3
LR                  = 2e-5
WARMUP_RATIO        = 0.03
WEIGHT_DECAY        = 0.01
LOGGING_STEPS       = 10
EVAL_STEPS          = 200     # fewer pauses; no change to training math
SAVE_STEPS          = 200     # fewer saves; best-at-end still loaded
SAVE_TOTAL_LIMIT    = 3
SEED                = 42

USE_BF16            = True
USE_FP16            = False
GRADIENT_CKPT       = False
USE_TORCH_COMPILE   = False  # keep False on Windows (Triton/Inductor)

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.10
TARGET_MODULES      = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"

WANDB_PROJECT       = ""  # kept but unused

# ---- Fast evaluation knobs ----
INTERIM_EM_SAMPLE   = 60   # random sample size for interim EM (fast)
INTERIM_EM_EVERY    = 2    # run interim EM every N evals (pairs well with EVAL_STEPS)
FINAL_EM_BATCH      = 8    # batch size for full-val EM generation (keeps VRAM in check)
INTERIM_MAX_NEW     = 384  # cap tokens for interim EM generation
FINAL_MAX_NEW       = 384  # cap tokens for final full-val EM generation

# ======== UTILS ========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def default_prompt(question: str) -> str:
    return (
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Answer:\n"
        "Please reason step by step, and put your final answer within \\boxed{} on the last line.\n\n"
    )

def extract_final_numeric(s: str) -> Optional[str]:
    """Prefer \\boxed{num}, fallback to {num} at the end of the string."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    num_pattern = r"[-+]?(?:\d+/\d+|\d+(?:\.\d+)?)"
    m = re.search(rf"\\boxed\{{\s*({num_pattern})\s*\}}\s*$", s)
    if m:
        return m.group(1)
    m = re.search(rf"\{{\s*({num_pattern})\s*\}}\s*$", s)
    return m.group(1) if m else None

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

# ======== Regex stop + trim ========
NUM = r"[-+]?(?:\d+/\d+|\d+(?:\.\d+)?)"
ANSWER_REGEXES = [
    re.compile(rf"(.*?\\boxed\{{\s*{NUM}\s*\}})", re.S),     # \boxed{...}
    re.compile(rf"(.*?\{{\s*{NUM}\s*\}})", re.S),            # {...}
    re.compile(rf"(.*?####\s*{NUM})", re.S),                 # #### ...
]

class RegexStopper(StoppingCriteria):
    def __init__(self, tokenizer, patterns, window_tokens=128):
        self.tok = tokenizer
        self.patterns = patterns
        self.window = window_tokens
    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        tail = ids[-self.window:] if len(ids) > self.window else ids
        txt = self.tok.decode(tail, skip_special_tokens=True)
        for pat in self.patterns:
            if pat.search(txt):
                return True
        return False

def build_stops(tok):
    return StoppingCriteriaList([RegexStopper(tok, ANSWER_REGEXES, window_tokens=128)])

def trim_to_first_answer(text: str) -> str:
    for pat in ANSWER_REGEXES:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return text.strip()

# ======== Numeric equivalence helpers (Math EM) ========
from fractions import Fraction
from decimal import Decimal, InvalidOperation, getcontext
getcontext().prec = 50  # robust equality via rational form

def _canon_number(s: Optional[str]) -> Optional[Fraction]:
    """Canonicalize numeric strings to rational numbers for exact math equality."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if not s:
        return None
    if s.startswith("$"):
        s = s[1:].lstrip()
    if s.endswith("%"):
        base = s[:-1].strip()
        try:
            return Fraction(Decimal(base)) / 100
        except (InvalidOperation, ValueError):
            return None
    if "/" in s and all(ch.isdigit() or ch in "+-/. " for ch in s):
        try:
            return Fraction(s)
        except Exception:
            pass
    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError):
        return None

# ======== Data ========
class SupervisedJsonl(Dataset):
    """
    JSONL rows: {"question": "...", "answer": "..."}
    Creates [PROMPT][ANSWER + EOS], masks prompt tokens, preserves full answer tail (+EOS).
    Drops malformed rows and prints a summary at init.
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int,
                 mask_prompt_loss: bool = True):
        super().__init__()
        raw = read_jsonl(path)
        good, bad = [], 0
        for r in raw:
            q = r.get("question", None)
            a = r.get("answer", None)
            if isinstance(q, str) and isinstance(a, str) and q.strip() != "" and a.strip() != "":
                good.append({"question": q, "answer": a})
            else:
                bad += 1
        if bad:
            print(f"[data] Skipped {bad} malformed row(s) in {os.path.basename(path)}")
        self.rows = good
        self.tok = tokenizer
        self.maxlen = max_length
        self.mask_prompt_loss = mask_prompt_loss

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        ex = self.rows[idx]
        prompt = default_prompt(ex["question"])
        answer = ex["answer"]

        q_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]

        # Teach EOS: append once if not present
        eos_id = self.tok.eos_token_id
        if eos_id is not None and (len(a_ids) == 0 or a_ids[-1] != eos_id):
            a_ids = a_ids + [eos_id]

        # Keep full tail of answer (+EOS); trim prompt from left
        maxlen = self.maxlen
        min_prompt_tokens = 1
        keep_answer = min(len(a_ids), maxlen - min_prompt_tokens)
        a_ids = a_ids[-keep_answer:]

        room_for_prompt = maxlen - keep_answer
        room_for_prompt = max(room_for_prompt, min_prompt_tokens)
        q_ids = q_ids[-room_for_prompt:]

        input_ids = q_ids + a_ids
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()

        # Mask ONLY the prompt; do not mask EOS
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
        """
        Robust collator:
          - If any example is missing a key, backfill a 1-token dummy to keep the batch valid.
          - Labels for dummy tokens are -100 so they don't affect loss.
        """
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

        # Optionally pad to multiple of N (kernel-friendly)
        if self.pad_to_multiple_of:
            for k, pad_val in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
                L = batch[k].size(1)
                pad_len = (-L) % self.pad_to_multiple_of
                if pad_len:
                    pad = torch.full((batch[k].size(0), pad_len), pad_val, dtype=batch[k].dtype)
                    batch[k] = torch.cat([batch[k], pad], dim=1)

        return batch

# ======== Callbacks ========
class FitSignalCallback(TrainerCallback):
    def __init__(self):
        self.last_train = None
        self.last_eval = None
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        msg = []
        if "loss" in logs:
            self.last_train = logs["loss"]
            msg.append(f"train_loss={self.last_train:.4f}")
        if "eval_loss" in logs:
            self.last_eval = logs["eval_loss"]
            msg.append(f"eval_loss={self.last_eval:.4f}")
        if msg:
            print(f"[progress] step={state.global_step} | " + " | ".join(msg))
        if self.last_train is not None and self.last_eval is not None:
            if self.last_eval > self.last_train * 1.25:
                print("[signal] Possible overfitting: eval ≫ train.")
            if self.last_eval > 2.5 and self.last_train > 2.5:
                print("[signal] Possible underfitting: both losses high.")

class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, every_steps: int):
        self.every = max(1, int(every_steps))
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and (state.global_step % self.every == 0):
            control.should_evaluate = True
            return control

class BestSaverCallback(TrainerCallback):
    """Best-by-loss saver (adapters)"""
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.best = None
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        cur = metrics["eval_loss"]
        model = kwargs.get("model", None)
        if model is None:
            return
        if self.best is None or cur < self.best:
            self.best = cur
            save_dir = os.path.join(self.out_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"[best] New best eval_loss={cur:.4f} -> saved adapter to {save_dir}")

class CSVLoggerCallback(TrainerCallback):
    """Write per-step metrics to OUTPUT_DIR/metrics_log.csv"""
    def __init__(self, out_dir: str, filename: str = "metrics_log.csv"):
        self.path = os.path.join(out_dir, filename)
        self.header_written = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._keys = ["step","epoch","split","loss","eval_loss","val_em_numeric","val_em_strict",
                      "val_found_rate","val_parse_fail_rate","lr"]
    def _write(self, row: dict):
        row = {k: row.get(k) for k in self._keys}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._keys)
            if not self.header_written:
                w.writeheader()
                self.header_written = True
            w.writerow(row)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "split": "train",
            "loss": logs.get("loss"),
            "lr": logs.get("learning_rate"),
        }
        self._write(row)
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics: return
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "split": "eval",
            "eval_loss": metrics.get("eval_loss"),
            "val_em_numeric": metrics.get("val_em_numeric"),
            "val_em_strict": metrics.get("val_em_strict"),
            "val_found_rate": metrics.get("val_found_rate"),
            "val_parse_fail_rate": metrics.get("val_parse_fail_rate"),
        }
        self._write(row)

class EvalMetricsEcho(TrainerCallback):
    """Pretty-print eval metrics (including EMs merged by BestEMSaverCallback)."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        msg = []
        for k in ["eval_loss", "val_em_numeric", "val_em_strict",
                  "val_found_rate", "val_parse_fail_rate"]:
            if k in metrics and metrics[k] is not None:
                try:
                    msg.append(f"{k}={metrics[k]:.4f}" if isinstance(metrics[k], (int,float)) else f"{k}={metrics[k]}")
                except Exception:
                    msg.append(f"{k}={metrics[k]}")
        if msg:
            print("[eval] " + " | ".join(msg))

class BestEMSaverCallback(TrainerCallback):
    """
    Runs Math EM on a random sample every N-th eval, merges EM metrics into Trainer logs,
    and saves best-by-numeric-EM adapter to {out}/best_em.
    """
    def __init__(self, out_dir: str, tok, val_ds, every_n_evals: int = INTERIM_EM_EVERY, em_sample: int = INTERIM_EM_SAMPLE):
        self.out_dir = out_dir
        self.tok = tok
        self.val_ds = val_ds
        self.best_em = None
        self.every_n_evals = max(1, int(every_n_evals))
        self.em_sample = max(1, int(em_sample))
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        eval_steps = getattr(args, "eval_steps", None)
        if eval_steps:
            eval_index = max(1, state.global_step // max(1, eval_steps))
            if eval_index % self.every_n_evals != 0:
                return
        if model is None:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        em_metrics = evaluate_em(
            model, self.tok, self.val_ds,
            num_eval=min(self.em_sample, len(self.val_ds)),
            device=device,
            batch_size=1,                 # keep interim EM cheap on VRAM
            use_regex_stop=True,          # regex stop helps shorten interim generations
            max_new_tokens=INTERIM_MAX_NEW
        )
        if metrics is not None and isinstance(metrics, dict):
            metrics.update(em_metrics)
        em = em_metrics.get("val_em_numeric", 0.0)
        print(f"[eval] sample Math EM={em:.2f}% (n={em_metrics.get('evaluated')})")
        if self.best_em is None or em > self.best_em:
            self.best_em = em
            save_dir = os.path.join(self.out_dir, "best_em")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"[best-em] New best val_em_numeric={em:.2f}% -> saved to {save_dir}")

# ======== TrainingArguments compat ========
def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    eval_supported = ("evaluation_strategy" in supported)
    if not eval_supported:
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

# ======== Eval (fast path, supports batching) ========
@torch.no_grad()
def evaluate_em(model,
                tok,
                dataset: SupervisedJsonl,
                num_eval=200,
                max_ceiling=512,
                device="cuda",
                batch_size=1,
                use_regex_stop=False,
                max_new_tokens=None):
    """
    Fast EM evaluation:
      - Random sample (stable seed) of num_eval items
      - Batched generation when batch_size > 1 (regex stopper disabled in batch mode)
      - Greedy decode; trim to first answer; Math EM + strict EM + validity rates
      - NEW: temporarily switch tokenizer to LEFT padding during generation (correct for decoder-only)
    """
    model.eval()
    orig_cache = getattr(model.config, "use_cache", None)
    old_pad_side  = getattr(tok, "padding_side", "right")
    old_trunc_side = getattr(tok, "truncation_side", "right") 


    try:
        # Speed up generation path
        model.config.use_cache = True

        # Ensure tokenizer has pad_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        # Make sure model has matching ids
        try:
            model.config.eos_token_id = tok.eos_token_id
            model.config.pad_token_id = tok.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.eos_token_id = tok.eos_token_id
                model.generation_config.pad_token_id = tok.pad_token_id
        except Exception:
            pass

        # --- NEW: use LEFT padding for decoder-only generation ---
        tok.padding_side = "left"
        tok.truncation_side = "left"

        # Context length + stopper
        model_ctx = getattr(model.config, "max_position_embeddings", None)
        if not model_ctx:
            model_ctx = getattr(tok, "model_max_length", MAX_LENGTH)
        safety_buffer = 8
        stops = build_stops(tok) if use_regex_stop and batch_size == 1 else None

        # Sample subset deterministically
        n = min(num_eval, len(dataset))
        idxs = list(range(len(dataset)))
        random.Random(1234).shuffle(idxs)
        idxs = idxs[:n]

        correct_num = 0
        correct_strict = 0
        found_final = 0
        parse_fail = 0

        model.to(device)

        def dyn_new_tokens(in_len):
            # allow caller override
            if max_new_tokens is not None:
                return max_new_tokens
            return max(128, min(max_ceiling, max(32, model_ctx - in_len - safety_buffer)))

        # Batched generation
        for start in range(0, n, batch_size):
            chunk = idxs[start:start+batch_size]
            prompts = [ default_prompt(dataset.rows[i]["question"]) for i in chunk ]
            gts = [ extract_final_numeric(dataset.rows[i]["answer"]) for i in chunk ]

            inputs = tok(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True
            ).to(device)

            # choose a single max_new_tokens across the batch for simplicity
            in_lens = inputs["input_ids"].shape[1]
            this_max_new = dyn_new_tokens(in_lens)

            gen_kwargs = dict(
                max_new_tokens=this_max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            if stops is not None:
                gen_kwargs["stopping_criteria"] = stops

            outputs = model.generate(**inputs, **gen_kwargs)

            # decode per-sample
            for bi in range(outputs.size(0)):
                # prompt length = non-pad tokens (since left-padding)
                prompt_len = (inputs["input_ids"][bi] != tok.pad_token_id).sum().item()
                text = tok.decode(outputs[bi][prompt_len:], skip_special_tokens=True)
                text = trim_to_first_answer(text)
                pred = extract_final_numeric(text)

                if pred is not None:
                    found_final += 1
                else:
                    parse_fail += 1

                gt = gts[bi]
                if gt is not None and pred is not None and gt.strip() == pred.strip():
                    correct_strict += 1

                gt_num = _canon_number(gt) if gt is not None else None
                pr_num = _canon_number(pred) if pred is not None else None
                if gt_num is not None and pr_num is not None and gt_num == pr_num:
                    correct_num += 1

        em_num = (correct_num / n) * 100 if n else 0.0
        em_strict = (correct_strict / n) * 100 if n else 0.0
        found_rate = (found_final / n) * 100 if n else 0.0
        fail_rate = (parse_fail / n) * 100 if n else 0.0

        return {
            "val_em_numeric": em_num,
            "val_em_strict": em_strict,
            "val_found_rate": found_rate,
            "val_parse_fail_rate": fail_rate,
            "evaluated": n
        }

    finally:
        try:
            tok.padding_side   = old_pad_side
            tok.truncation_side = old_trunc_side
        except Exception:
            pass
        if orig_cache is not None:
            model.config.use_cache = orig_cache



# ======== Main ========
def main():
    for p in (MODEL_DIR, TRAIN_FILE, VAL_FILE):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_seed(SEED)

    print("=== LoRA Fine-tune Config ===")
    print("MODEL_DIR:", MODEL_DIR)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("VAL_FILE:", VAL_FILE)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("BF16:", USE_BF16, "FP16:", USE_FP16, "Grad-CKPT:", GRADIENT_CKPT)
    print("LoRA r/alpha/dropout:", LORA_R, LORA_ALPHA, LORA_DROPOUT)
    print("==============================")

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
    try:
        model.config.eos_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    # Prefer SDPA attention if available (fast forward/eval, no accuracy change)
    if hasattr(model.config, "attn_implementation"):
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass

    # Keep cache OFF during training (turned ON inside eval)
    model.config.use_cache = False

    # ---- LoRA before grad-ckpt ----
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

    # Enable grad-ckpt after PEFT
    if GRADIENT_CKPT:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    # No torch.compile on Windows (Triton/Inductor issue)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[speed] torch.compile enabled.")
        except Exception as e:
            print(f"[speed] torch.compile unavailable: {e}")

    train_ds = SupervisedJsonl(TRAIN_FILE, tok, MAX_LENGTH, mask_prompt_loss=True)
    val_ds   = SupervisedJsonl(VAL_FILE, tok, MAX_LENGTH, mask_prompt_loss=True)
    collator = DataCollator(tokenizer=tok)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM)))
    total_steps = steps_per_epoch * EPOCHS

    targs = build_training_args(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,     # bigger eval batch = faster eval
        eval_accumulation_steps=1,         # no micro-batching for eval
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,

        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        lr_scheduler_type="cosine",
        bf16=USE_BF16,
        fp16=(USE_FP16 and not USE_BF16),
        gradient_checkpointing=GRADIENT_CKPT,

        # Logging: disable dashboards for quickest run; CSV + console still work
        report_to="none",
        logging_dir=os.path.join(OUTPUT_DIR, "tb"),

        run_name=None,
        optim="adamw_torch",
        save_safetensors=True,

        # Faster/safer input pipeline
        max_grad_norm=1.0,
        group_by_length=True,
        dataloader_num_workers=6,      # more CPU workers for feeding
        dataloader_pin_memory=True,

        # Keep all columns and return only loss for eval
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[
            FitSignalCallback(),
            PeriodicEvalCallback(EVAL_STEPS),
            BestSaverCallback(OUTPUT_DIR),  # loss-based best
            BestEMSaverCallback(OUTPUT_DIR, tok, val_ds, every_n_evals=INTERIM_EM_EVERY, em_sample=INTERIM_EM_SAMPLE),  # EM-based best (fast)
            EvalMetricsEcho(),
            CSVLoggerCallback(OUTPUT_DIR),  # per-step CSV logs
        ],
    )

    resume_ckpt = find_latest_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"[resume] Found checkpoint: {resume_ckpt} — resuming training.")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("[resume] No checkpoint found — starting fresh.")
        trainer.train()

    # -------- Final, FULL validation EM (fast, batched, no regex stopper) --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    em_metrics = evaluate_em(
        trainer.model, tok, val_ds,
        num_eval=len(val_ds),
        device=device,
        batch_size=FINAL_EM_BATCH,     # batched for speed, safe on VRAM
        use_regex_stop=False,          # skip stopper in batch mode for speed
        max_new_tokens=FINAL_MAX_NEW
    )
    print(f"[eval] Full EM on val: {em_metrics}")

    print("Saving adapter and tokenizer…")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # Save a human-readable scorecard
    with open(os.path.join(OUTPUT_DIR, "final_scores.txt"), "w", encoding="utf-8") as f:
        f.write(f"val_em_numeric={em_metrics.get('val_em_numeric'):.2f}\n")
        f.write(f"val_em_strict={em_metrics.get('val_em_strict'):.2f}\n")
        f.write(f"val_found_rate={em_metrics.get('val_found_rate'):.2f}\n")
        f.write(f"val_parse_fail_rate={em_metrics.get('val_parse_fail_rate'):.2f}\n")
        f.write(f"evaluated={em_metrics.get('evaluated')}\n")

    # Rich, structured run meta
    meta = {
        "train_file": TRAIN_FILE,
        "val_file": VAL_FILE,
        "model_dir": MODEL_DIR,
        "output_dir": OUTPUT_DIR,
        "bf16": USE_BF16,
        "fp16": USE_FP16,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": [m.strip() for m in TARGET_MODULES.split(",") if m.strip() ],
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
        "note": "Trained to emit \\boxed{...} + EOS; preserves answer tail; fast interim EM; batched full EM; best adapter saved by loss and by Math EM.",
        "val_em_numeric": em_metrics.get("val_em_numeric"),
        "val_em_strict": em_metrics.get("val_em_strict"),
        "val_found_rate": em_metrics.get("val_found_rate"),
        "val_parse_fail_rate": em_metrics.get("val_parse_fail_rate"),
        "val_evaluated": em_metrics.get("evaluated"),
        "tensorboard_dir": os.path.join(OUTPUT_DIR, "tb"),
        "best_loss_adapter_dir": os.path.join(OUTPUT_DIR, "best"),
        "best_em_adapter_dir": os.path.join(OUTPUT_DIR, "best_em"),
        "csv_log": os.path.join(OUTPUT_DIR, "metrics_log.csv"),
        "final_scores_txt": os.path.join(OUTPUT_DIR, "final_scores.txt"),
        "interim_em_sample": INTERIM_EM_SAMPLE,
        "final_em_batch_size": FINAL_EM_BATCH,
        "interim_max_new": INTERIM_MAX_NEW,
        "final_max_new": FINAL_MAX_NEW,
    }
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Done.")

def run():
    main()

if __name__ == "__main__":
    main()
