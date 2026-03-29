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
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from evaluation.metrics import db_eval_iou, db_eval_boundary
import cv2

def compute_mask_IoU(masks, target):
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

class VIDEO_DATA(Dataset):
    def __init__(self, video_folder, data_list, max_frames=16):
        data_list_new = []
        for d in data_list:
            if 'Expression' in d or 'expression' in d:
                if not os.path.exists(os.path.join(video_folder, d['video'])) or len(os.listdir(os.path.join(video_folder, d['video'])))==0:
                    print(f"Video path does not exist: {os.path.join(video_folder, d['video'])}")
                    continue
                if 'Expression' in d:
                    expression = d['Expression']
                else:
                    expression = d['expression']
                data_list_new.append(
                    {
                        "video": os.path.join(video_folder, d['video']),
                        "category": expression,
                        "gt_mask": d["masks"],
                        "frame_names": d.get('frame_names', None)
                    }
                )
            else:
                data_list_new.append(
                    {
                        "video": os.path.join(video_folder, d['video']),
                        "question": d['question'],
                        "gt_mask": d["masks"],
                        "frame_names": d.get('frame_names', None)
                    }
                )
        self.max_frames = max_frames
                
        self.data_list = data_list_new
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if 'category' in data:
            # 匹配训练时的格式："Can you segment {text} in the video?" -> "It is [SEG]."
            category = data['category'][0].lower() + data['category'][1:]
            if category.endswith('.'):
                category = category[:-1]
            instruction = f"Can you segment '{category}' in the video?"
        else:
            instruction = data['question']
        masks = []
        mask_nums = []

        # 获取图像尺寸（从 GT mask 或视频第一帧）
        height, width = None, None
        for msk in data['gt_mask']:
            if msk is not None:
                height, width = msk['size']
                break
        
        # 如果没有 GT mask，从视频第一帧获取尺寸
        if height is None:
            if data.get('frame_names') is not None:
                frame_names = data['frame_names']
                video_dir = data['video']
                first_frame_path = os.path.join(video_dir, f"{frame_names[0]}.jpg")
                img = cv2.imread(first_frame_path)
                height, width = img.shape[:2]
            else:
                # 默认尺寸
                height, width = 480, 640
        
        gt_masks = []
        for msk in data['gt_mask']:
            if msk is not None:
                gt_masks.append(annToMask(msk))
            else:
                gt_masks.append(np.zeros((height,width)))

        # 如果有 frame_names，直接加载这些帧（确保帧对齐）
        frame_paths = None  # 用于传给mm_infer_segmentation
        if data.get('frame_names') is not None:
            frame_paths = []
            frame_names = data['frame_names']
            video_dir = data['video']
            for frame_name in frame_names:
                frame_paths.append(os.path.join(video_dir, f"{frame_name}.jpg"))

            if len(frame_names) > self.max_frames:
                video_ids = np.linspace(0, len(frame_names)-1, self.max_frames, dtype=int)
                frame_names = [frame_names[i] for i in video_ids]

            
            images = []
            timestamps = []
            for frame_name in frame_names:
                # 添加 .jpg 扩展名
                img_path = os.path.join(video_dir, f"{frame_name}.jpg")
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                images.append(img)
                timestamps.append(int(frame_name.replace('img', ''))//30) 
            images = np.array(images)
        else:
            # 没有 frame_names，使用原来的自动采样方式
            images, timestamps = load_video_from_ids(data['video'], max_frames=self.max_frames, must_sample_frames=None)

   
        return {
            'idx': idx,
            'video': (images, timestamps),
            'masks': gt_masks,
            'instruction': instruction,
            'video_path': frame_paths if frame_paths is not None else data["video"],
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    img = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    ins = [x['instruction'] for x in batch]
    ip = [x['video_path'] for x in batch]
    return idx, img, msk, ins, ip

def build_eval_dataloader(args, processor, distributed):
    # convert parquet to json
    questions = json.load(open(args.question_file))#[605:606]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VIDEO_DATA(args.video_folder, questions)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)

    return dataloader

def save_results(result, save_path):
    if len(result) == 0:
        print("Warning: No results to save!")
        return
        
    j = sum(d['j'] for d in result) / len(result)
    f = sum(d['f'] for d in result) / len(result)

    metrics = {
        'j': j,
        'f': f,
        'j&f': (j+f)/2,
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

        disable_torch_init()
        model, processor = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})
    else:
        local_rank = 0
        global_rank = 0
        disable_torch_init()
        model, processor = model_init(args.model_path)

    model.to(torch.bfloat16)
    
    val_loader = build_eval_dataloader(args, processor, distributed)
    
    results = []
    for i, (idx, img, masks_, instruction, image_paths) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        video = img[0]
        gt_masks = masks_[0]
        instruction = instruction[0]
        video_path = image_paths[0]
  
        # try:
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
        pred_masks = pred_masks>0
        pred_masks = pred_masks.squeeze(1).detach().cpu().numpy()
    

        # 根据cls_scores过滤mask，然后再编码RLE
        mask_rles = []
        gt_cls = []
        for ii in range(len(pred_masks)):
            pred_mask = pred_masks[ii]
            mask_rle = singleMask2rle(pred_mask)
            mask_rles.append(mask_rle)
        
        # 检查维度是否匹配
        if gt_masks.shape[0] == pred_masks.shape[0]:                
            j = db_eval_iou(gt_masks, pred_masks).mean()
            f = db_eval_boundary(gt_masks, pred_masks).mean()
        else:
            # 维度不匹配，跳过中间评估（最终会用官方脚本评估）
            print(f"Data {idx}: GT frames ({gt_masks.shape[0]}) != Pred frames ({pred_masks.shape[0]}), skip intermediate eval")
            j = 0.0
            f = 0.0
        
        record = {
            'idx': idx,
            'instruction': instruction,
            'prediction': output,
            'j': float(j),
            'f': float(f),
            'mask_rle': mask_rles,
            'video_path': video_path,
        }
        results.append(record)
        # except Exception as e:
        #     print(f"Data {idx} Error: {e}")
        #     continue
   

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
    parser.add_argument('--video_folder', help='Directory containing video files.', default='/mnt')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', default='/public/hz_oss/yunxuan/data/scene_understanding/eval_data/old_qa/seg_eval_4_room.json')
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='visualization/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--vis", type=str, default=None)
    args = parser.parse_args()

    run_inference(args)


