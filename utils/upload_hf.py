from huggingface_hub import HfApi
from huggingface_hub import login
import argparse

api = HfApi()


parser = argparse.ArgumentParser(
    description="Upload a local model folder to Hugging Face Hub"
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Local path to the model/checkpoint folder"
)
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="Target Hugging Face repo id, e.g. username/repo_name"
)
args = parser.parse_args()

api.upload_folder(
    folder_path=f"work_dirs/{args.model_path}",
    repo_id=args.repo_id,
    repo_type="model",
)
