import modal
import sys
import os

app = modal.App("prune-run3")

# Reuse your existing volume
volume = modal.Volume.from_name("datasets-volume")

# Build image with requirements + all files in root
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app")  # ✅ all root files
)

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/root/datasets": volume},
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run():
    # Make all code files importable
    sys.path.append("/app")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")  # modal injects it at runtime

    # print("Started pruning...")
    # import physical_pruning

    # print("🚀 Starting pruning output script...")
    # import prune_model_output_script
    # print("started random pruning...")
    # import random_pruning
    print("🚀 Starting random pruning output script...")
    import random_prune_model_output_script

    print("✅ Script finished")

@app.local_entrypoint()
def main():
    run.remote()