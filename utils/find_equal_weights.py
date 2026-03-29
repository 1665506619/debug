import torch
from safetensors.torch import load_file

m1 = load_file("/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/code/seg-sam3/work_dirs/1223_pretrain_v1/checkpoint-5000/model.safetensors")
m2 = load_file("/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/code/seg-sam3/work_dirs/1223_pretrain_v1/checkpoint-5247/model.safetensors")

for name in sorted(set(m1.keys()) & set(m2.keys())):
    if m1[name].shape == m2[name].shape \
       and m1[name].dtype == m2[name].dtype \
       and torch.equal(m1[name], m2[name]):
        print(name)
