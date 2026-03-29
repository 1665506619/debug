from huggingface_hub import HfApi
api = HfApi()
from huggingface_hub import login

import glob
# data_list = glob.glob('/Users/yyq/Downloads/obj365_json.zip*')
# for data_path in data_list:
#     print(data_path)
#     api.upload_file(
#         path_or_fileobj=data_path,
#         path_in_repo=data_path.split('/')[-1],
#         repo_id="Lillyr/seg",
#         repo_type="dataset"

#     )
api.upload_large_folder(
    folder_path='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/work_dirs/1202_pretrain_v0_lora/checkpoint-20752',
    repo_id="Lillyr/pretrain_v0",
    repo_type="model",
)