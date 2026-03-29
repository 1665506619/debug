from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lillyr/0306_ft_v11_lora_8b_base_0222_pretrain_v2_bs64_1e_5",
    repo_type="model",
    local_dir="work_dirs/0306_ft_v11_lora_8b_base_0222_pretrain_v2_bs64_1e_5",
    local_dir_use_symlinks=False,
)