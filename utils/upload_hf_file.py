from huggingface_hub import HfApi
from huggingface_hub import login
import argparse

api = HfApi()


parser = argparse.ArgumentParser(
    description="Upload a local model folder to Hugging Face Hub"
)
parser.add_argument(
    "--file_path",
    type=str,
    required=True,
    help="Local path to the model/checkpoint folder"
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="Lillyr/seg",
    help="Target Hugging Face repo id, e.g. username/repo_name"
)
args = parser.parse_args()

api.upload_file(
    path_or_fileobj=f"{args.file_path}",
    path_in_repo=args.file_path.split("/")[-1],
    repo_id=args.repo_id,
    repo_type="dataset",
)
