import os
import tarfile
from huggingface_hub import hf_hub_download

repo_id = "Aber-r/SA-1B_backup"
download_dir = "/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/data/SA-1B/SA-1B-tars"
extract_dir = "/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/data/SA-1B/images"

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)

for i in [158,185,492,656,697,715,742,859,915]:
    filename = f"sa_{i:06d}.tar"
    print(f"\n📥 Downloading {filename}...")

    try:
        tar_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"📦 Extracting {filename} to {extract_dir}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)
    except:
        print(f"⚠️ Warning: Failed to download or extract {filename}. Skipping...")

print("\n✅ All files downloaded and extracted!")
