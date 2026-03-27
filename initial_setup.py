import modal

app = modal.App("initial-setup")

volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("datasets", remote_path="/tmp/datasets")  # ✅ IMPORTANT
)

@app.function(
    image=image,
    volumes={"/root/datasets": volume},
)
def upload():
    import os
    import shutil

    local_path = "/tmp/datasets"   # 👈 now inside container
    remote_path = "/root/datasets"

    for item in os.listdir(local_path):
        src = os.path.join(local_path, item)
        dst = os.path.join(remote_path, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print("✅ Dataset uploaded!")

    for root, dirs, files in os.walk(remote_path):
        print(root, "->", len(files), "files")

    volume.commit()


@app.local_entrypoint()
def main():
    upload.remote()