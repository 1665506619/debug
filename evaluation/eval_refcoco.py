import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from transformers import AutoProcessor
from easy_vlm.models import load_pretrained_model
from easy_vlm import disable_torch_init, mm_infer_segmentation

def compute_mask_IoU(masks, target):
    if target.sum()==0 and masks.sum()==0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)

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

class Refcoco(Dataset):
    def __init__(self, image_folder, data_list, data_type=None, only_mask_img = True):
        data_list_new = []
        for d in data_list:
            for i, cat in enumerate(d['cat']):
                if os.path.exists(os.path.join(image_folder, 'coco/train2017', d['image'].split('_')[-1])):
                    image_path = os.path.join(image_folder, 'coco/train2017', d['image'].split('_')[-1])
                else:
                    image_path = os.path.join(image_folder, 'coco/val2017', d['image'].split('_')[-1])

                data_list_new.append(
                    {
                        "image": image_path,
                        "height": d["height"],
                        "width": d["width"],
                        "category": cat,
                        "gt_mask": d["masks"][i]
                    }
                )
                # if len(data_list_new)>=50:
                #     break
        self.data_list = data_list_new
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        instruction = f"Can you segment '{data['category']}' in the image?" 
        masks = []
        mask_nums = []

        if 'height' in data:
            h = data['height']
            w = data['width']
        else:
            h = None
            w = None

        if data["gt_mask"] is None:
            gt_mask = np.zeros((data['height'], data['width']), dtype=np.uint8)
        else:
            try:
                gt_mask = annToMask(data["gt_mask"], h, w)
            except: # grefcoco multi-mask
                gt_mask = np.zeros((data['height'], data['width']), dtype=np.uint8)
                for msk in data["gt_mask"]:
                    mask = annToMask(msk, h, w)
                    gt_mask = np.maximum(gt_mask, mask)
        
        contents = []
        contents.append({"type": "image", "image": data['image']})
        contents.append({"type": "text", "text": instruction})

        conversation = [{"role": "user", "content": contents}]

   
        return {
            'idx': idx,
            'conversation': conversation,
            'masks': gt_mask,
            'image_path': data["image"],
            'instruction': instruction,
            'gt_mask_rle': data["gt_mask"]
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    conv = [x['conversation'] for x in batch]
    msk = [x['masks'] for x in batch]
    ip = [x['image_path'] for x in batch]
    mskr = [x['gt_mask_rle'] for x in batch]
    inst = [x['instruction'] for x in batch]
    return idx, conv, msk, ip, mskr, inst

def build_eval_dataloader(args, processor, distributed):
    # convert parquet to json
    questions = json.load(open(args.question_file))#[:10]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Refcoco(args.image_folder, questions)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)

    return dataloader

def save_results(result, save_path):
    giou = sum(d['iou'] for d in result) / len(result)
    inter_ = sum(d['inter'] for d in result)
    union_ = sum(d['union'] for d in result)
    ciou = inter_ / (union_ + 1e-10)

    metrics = {
        'giou': giou,
        'ciou': ciou
    }
    result.insert(0, metrics)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in result:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format.")
    print(f"Answer saved at:{save_path}")
    

def run_inference(args):
    distributed = os.getenv('WORLD_SIZE', '1') > '1'
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_map = {"": local_rank}

        disable_torch_init()
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map=device_map,
        )

        processor = AutoProcessor.from_pretrained(
            args.model_path,
        )

    else:
        local_rank = 0
        global_rank = 0
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}
        disable_torch_init()
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path,
        )

    model.to(torch.bfloat16)
    
    val_loader = build_eval_dataloader(args, processor, distributed)
    
    results = []
    for i, (idx, conversation, masks_, image_paths, gt_mask_rles, instruction) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        conversation = conversation[0]
        gt_masks = masks_[0]
        image_path = image_paths[0]
        gt_mask_rle = gt_mask_rles[0]
        instruction = instruction[0]
  
        # try:
        output, masks, cls_scores = mm_infer_segmentation(
            image_path,
            processor,
            conversation,
            model,
            tokenizer,
        )
        h, w = gt_masks.shape[0], gt_masks.shape[1]
        
        if masks is not None:
            keep = cls_scores > args.threshold
            selected_masks = masks[keep]

        if masks is None or selected_masks.numel() == 0:
            print(output)
            selected_masks = None
            pred_masks = torch.zeros((1, 1, h, w), dtype=bool, device=model.device)
        else:
            selected_masks = F.interpolate(selected_masks.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)>0
            selected_masks_ = selected_masks.any(dim=0)          
            pred_masks = selected_masks_[None].float().unsqueeze(0)  
            
        mask_rle = []
        if selected_masks is None:
            mask_rle.append(None)
        else:
            for pred_msk in selected_masks:
                mask_rle.append(singleMask2rle(pred_msk.detach().cpu().numpy()))
        gt_masks = torch.from_numpy(gt_masks)

        # print(pred_masks.shape, gt_masks.shape)
        inter, union, iou = compute_mask_IoU(pred_masks.contiguous().view(1,-1), gt_masks.contiguous().view(1,-1).to(pred_masks.device))
        
        if args.vis=='mask':
            output_folder = f'visualization/'
            os.makedirs(output_folder, exist_ok=True)
            for num, pm in enumerate(masks):
                plt.imshow(pm.detach().cpu().numpy())
                plt.savefig(os.path.join(output_folder, f'{num}_pred.png'))
                plt.imshow(gt_masks[num].detach().cpu().numpy())
                plt.savefig(os.path.join(output_folder, f'{num}_gt.png'))
                plt.imshow(video_tensor[0][num])
                plt.savefig(os.path.join(output_folder, f'{num}_rgb.png'))

        record = {
            'idx': idx,
            'instruction': instruction,
            'prediction': output,
            'inter': float(inter),
            'iou': float(iou),
            'union': float(union),
            'mask_rle': mask_rle,
            'image_path': image_path,
            'gt_mask_rle': gt_mask_rle
        }
        results.append(record)
   

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
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        save_results(results, args.output_file)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='', default='checkpoints/stage3_segmentation_0629_only_exist_with_313k_normal/5.2k_merge')
    parser.add_argument('--image_folder', help='Directory containing video files.', default='/mnt')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', default='/public/hz_oss/yunxuan/data/scene_understanding/eval_data/old_qa/seg_eval_4_room.json')
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='visualization/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--vis", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    run_inference(args)

