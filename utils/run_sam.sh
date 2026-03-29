torchrun --nproc_per_node=8 bbox2mask.py \
  --json_path /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k.json \
  --image_root /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data \
  --out_dir /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k_mask \
  --save_every 1000 \
  --thr 0.5
