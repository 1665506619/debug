import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from videollama3 import disable_torch_init, model_init, mm_infer, mm_infer_segmentation
from videollama3.mm_utils import annToMask, load_video_from_ids, load_images
import json
import numpy as np
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from evaluation.metrics import db_eval_iou, db_eval_boundary, get_r2vos_accuracy, get_r2vos_robustness

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
    
def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class ReasonVOS_Dataset(Dataset):
    def __init__(self, video_folder, data_list):
        self.video_folder = video_folder
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        instruction = f"Can you segment '{data['expression']}' in the video?"
        
        # Get dimensions from data (like official code)
        height = data['height']
        width = data['width']
        
        gt_masks = []
        for msk in data['masks']:
            if msk is not None:
                gt_masks.append(annToMask(msk))
            else:
                gt_masks.append(np.zeros((height, width)))

        video_path = os.path.join(self.video_folder, data['video'])
        images, timestamps = load_video_from_ids(video_path, max_frames=16, must_sample_frames=None)

        return {
            'idx': idx,
            'video': (images, timestamps),
            'masks': gt_masks,
            'instruction': instruction,
            'video_path': video_path,
            'type_id': data.get('type_id', 0),
            'video_id': data.get('video_id', ''),
            'exp_id': data.get('exp_id', ''),
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    img = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    ins = [x['instruction'] for x in batch]
    vp = [x['video_path'] for x in batch]
    tid = [x['type_id'] for x in batch]
    vid = [x['video_id'] for x in batch]
    eid = [x['exp_id'] for x in batch]
    return idx, img, msk, ins, vp, tid, vid, eid

def build_eval_dataloader(args, processor, distributed):
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = ReasonVOS_Dataset(args.video_folder, questions)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)

    return dataloader

def save_results(result, save_path, mask_dict_foreground):
    # Separate by type_id
    referring_results = [d for d in result if d.get('type_id') == 0]
    reasoning_results = [d for d in result if d.get('type_id') == 1]
    
    # Calculate metrics for referring (type_id=0)
    j_referring = np.mean([d['j'] for d in referring_results]) if referring_results else 0
    f_referring = np.mean([d['f'] for d in referring_results]) if referring_results else 0
    a_referring = np.mean([d['a'] for d in referring_results]) if referring_results else 0
    r_referring = np.mean([d['r'] for d in referring_results]) if referring_results else 0
    jf_referring = (j_referring + f_referring) / 2
    
    # Calculate metrics for reasoning (type_id=1)
    j_reasoning = np.mean([d['j'] for d in reasoning_results]) if reasoning_results else 0
    f_reasoning = np.mean([d['f'] for d in reasoning_results]) if reasoning_results else 0
    a_reasoning = np.mean([d['a'] for d in reasoning_results]) if reasoning_results else 0
    r_reasoning = np.mean([d['r'] for d in reasoning_results]) if reasoning_results else 0
    jf_reasoning = (j_reasoning + f_reasoning) / 2
    
    # Overall (average of referring and reasoning)
    j_overall = (j_referring + j_reasoning) / 2
    f_overall = (f_referring + f_reasoning) / 2
    a_overall = (a_referring + a_reasoning) / 2
    r_overall = (r_referring + r_reasoning) / 2
    jf_overall = (jf_referring + jf_reasoning) / 2
    
    metrics = {
        'referring': {
            'J': j_referring,
            'F': f_referring,
            'A': a_referring,
            'R': r_referring,
            'JF': jf_referring,
            'num_samples': len(referring_results)
        },
        'reasoning': {
            'J': j_reasoning,
            'F': f_reasoning,
            'A': a_reasoning,
            'R': r_reasoning,
            'JF': jf_reasoning,
            'num_samples': len(reasoning_results)
        },
        'overall': {
            'J': j_overall,
            'F': f_overall,
            'A': a_overall,
            'R': r_overall,
            'JF': jf_overall,
            'num_samples': len(result)
        }
    }
    
    output_data = {
        'metrics': metrics,
        'details': result
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Answer saved at:{save_path}")
    print("\n" + "="*60)
    print("ReasonVOS Evaluation Results")
    print("="*60)
    print(f"\nReferring (type_id=0): {metrics['referring']['num_samples']} samples")
    print(f"  J: {metrics['referring']['J']:.4f}")
    print(f"  F: {metrics['referring']['F']:.4f}")
    print(f"  J&F: {metrics['referring']['JF']:.4f}")
    print(f"  A: {metrics['referring']['A']:.4f}")
    print(f"  R: {metrics['referring']['R']:.4f}")
    
    print(f"\nReasoning (type_id=1): {metrics['reasoning']['num_samples']} samples")
    print(f"  J: {metrics['reasoning']['J']:.4f}")
    print(f"  F: {metrics['reasoning']['F']:.4f}")
    print(f"  J&F: {metrics['reasoning']['JF']:.4f}")
    print(f"  A: {metrics['reasoning']['A']:.4f}")
    print(f"  R: {metrics['reasoning']['R']:.4f}")
    
    print(f"\nOverall: {metrics['overall']['num_samples']} samples")
    print(f"  J: {metrics['overall']['J']:.4f}")
    print(f"  F: {metrics['overall']['F']:.4f}")
    print(f"  J&F: {metrics['overall']['JF']:.4f}")
    print(f"  A: {metrics['overall']['A']:.4f}")
    print(f"  R: {metrics['overall']['R']:.4f}")
    print("="*60)

def run_inference(args):
    distributed = os.getenv('WORLD_SIZE', '1') > '1'
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        disable_torch_init()
        model, processor = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})
    else:
        local_rank = 0
        global_rank = 0
        disable_torch_init()
        model, processor = model_init(args.model_path)

    model.to(torch.bfloat16)
    
    # Load foreground mask dict for robustness calculation
    mask_dict_foreground = json.load(open(args.foreground_mask_path))
    
    val_loader = build_eval_dataloader(args, processor, distributed)
    
    results = []
    for i, (idx, img, masks_, instruction, video_paths, type_ids, video_ids, exp_ids) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        video = img[0]
        gt_masks = masks_[0]
        instruction = instruction[0]
        video_path = video_paths[0]
        type_id = type_ids[0]
        video_id = video_ids[0]
        exp_id = exp_ids[0]
  
        try:
            output, masks, cls_scores = mm_infer_segmentation(
                video,
                processor,
                instruction,
                model=model,
                tokenizer=processor.tokenizer,
                do_sample=False,
                modal='video',
                video_path=video_path
            )
            
            gt_masks = np.array(gt_masks)
            h, w = gt_masks.shape[1], gt_masks.shape[2]

            if masks is None:
                masks = torch.zeros((len(gt_masks),1, h, w)).to(next(model.parameters()).device)
            
            else:
                mask_list = list(masks.values())  # 变成列表
                stacked = torch.stack(mask_list, dim=0)  # (num_obj, n, 1, h, w)
                stacked = stacked>0

                # 每一帧对所有目标取并集
                masks = torch.any(stacked.bool(), dim=0).float()
                # print(masks)
            
            pred_masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)

            pred_masks = (pred_masks>0).squeeze(1).detach().cpu().numpy()
            mask_rles = []
            for pred_mask in pred_masks:
                mask_rle = singleMask2rle(pred_mask)
                mask_rles.append(mask_rle)

            # Load foreground masks for robustness calculation
            foreground_masks = []
            if video_id in mask_dict_foreground:
                for frame_idx in range(len(gt_masks)):
                    fg_rle = mask_dict_foreground[video_id]["masks_rle"][frame_idx]
                    fg_mask = maskUtils.decode(fg_rle)
                    fg_mask = fg_mask.sum(axis=2).astype(np.uint8) if fg_mask.ndim == 3 else fg_mask.astype(np.uint8)
                    foreground_masks.append(fg_mask)
            else:
                # No foreground mask available, use all ones
                foreground_masks = [np.ones((h, w), dtype=np.uint8) for _ in range(len(gt_masks))]

            # Calculate metrics
            j = db_eval_iou(gt_masks, pred_masks).mean()
            f = db_eval_boundary(gt_masks, pred_masks).mean()
            a = get_r2vos_accuracy(gt_masks, pred_masks).mean()
            r = get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()
            
            record = {
                'idx': idx,
                'instruction': instruction,
                'prediction': output,
                'j': float(j),
                'f': float(f),
                'a': float(a),
                'r': float(r),
                'jf': float((j + f) / 2),
                'type_id': type_id,
                'video_id': video_id,
                'exp_id': exp_id,
                'mask_rle': mask_rles,
                'video_path': video_path,
            }
            results.append(record)
        except Exception as e:
            print(f"Data {idx} Error: {e}")
            continue
   

    if distributed:
        torch.cuda.empty_cache()
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            obj=results,
            object_gather_list=gathered_results if global_rank == 0 else None,
            dst=0,
        )
        if global_rank == 0:
            print("\n" * dist.get_world_size())
            results = sum(gathered_results, [])
            save_results(results, args.output_file, mask_dict_foreground)
        dist.destroy_process_group()
    else:
        save_results(results, args.output_file, mask_dict_foreground)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--video_folder', help='Directory containing video files.', default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--foreground_mask_path', help='Path to foreground mask dict.', default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/revos/mask_dict_foreground.json')
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    run_inference(args)

