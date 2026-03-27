import modal
import tarfile
import os

app = modal.App("download-random33")

# Reuse your volume
volume = modal.Volume.from_name("datasets-volume")

@app.function(
    volumes={"/root/datasets": volume},
    timeout=60 * 10,  # enough for small files
)
def download_random33_outputs():
    source_dir = "/root/datasets/random33 model outputs"
    archive_path = "/tmp/random33_model_outputs.tar.gz"

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"{source_dir} not found!")

    print("📦 Compressing random33 model outputs...")

    # This includes ALL subfolders + files
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname="random33 model outputs")

    print("✅ Compression done")

    # Return file as bytes
    with open(archive_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    data = download_random33_outputs.remote()

    with open("random33_model_outputs.tar.gz", "wb") as f:
        f.write(data)

    print("✅ Download complete: random33_model_outputs.tar.gz")