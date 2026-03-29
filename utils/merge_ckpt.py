import argparse
import os
import sys
sys.path.append('./')
from easy_vlm.models import load_pretrained_model
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/code/video-seg/work_dirs')
parser.add_argument("--model_path", type=str, default='0110_pretrain_pretrain_v0_20_lora/checkpoint-17000')
parser.add_argument("--save_path", type=str, default='0110_pretrain_pretrain_v0_20_lora_merge')
args = parser.parse_args()

save_path = os.path.join(args.base_dir, args.save_path)
model_path = os.path.join(args.base_dir, args.model_path)
tokenizer, model, processor = load_pretrained_model(model_path, None, attn_implementation='sdpa', save_path=save_path)

print("Model saved to ", save_path)