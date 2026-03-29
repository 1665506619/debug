import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from videollama3 import disable_torch_init, model_init, mm_infer, mm_infer_segmentation
from videollama3.mm_utils import annToMask, load_video, load_images
import json
import numpy as np
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2
from evaluation.utils import postprocess_seg_result, save_results, postprocess_prop_result
from evaluation.metrics import calculate_iou, calculate_iou_flatten, db_eval_boundary
from pycocotools import mask as maskUtils

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def get_type(caption, question):
    simple_expression_match = re.search(r'\[simple expression\]\s+(.*?)(?=\[complex expression\]|\Z)', caption, re.DOTALL)
    complex_expression_match = re.search(r'\[complex expression\]\s+(.*)', caption, re.DOTALL)
    simple_expression_text = simple_expression_match.group(1).strip() if simple_expression_match else None
    complex_expression_text = complex_expression_match.group(1).strip() if complex_expression_match else None
    assert not (simple_expression_text is None and complex_expression_text is None), caption
    if simple_expression_text is not None:
        simple_pattern = re.compile(re.escape(simple_expression_text.rstrip('.')), re.IGNORECASE)
    if complex_expression_text is not None:
        complex_pattern = re.compile(re.escape(complex_expression_text.rstrip('.')), re.IGNORECASE)

    # 使用正则表达式进行匹配
    if simple_expression_text is not None and simple_pattern.search(question):
        return 'direct referring'
    elif complex_expression_text is not None and complex_pattern.search(question):
        return 'situational referring'
    else:
        raise ValueError("No matching")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Mask_Dataset(Dataset):
    def __init__(self, video_folder, data_list, data_type=None, only_mask_img = True):
        self.video_folder = video_folder
        self.data_list = data_list
        self.data_type = data_type
        self.only_mask_img = only_mask_img
    
    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        data_folder = self.video_folder
        data = self.data_list[idx]
        video_root = data["video_root"].replace('embodied_cognition', 'damorobot/scene_understanding/videos_all')
        instruction = data['conversations'][0]['value']
        data["mask_ids"] = [mid for mid in data["mask_ids"]]
        video_file = data["video"]
        caption = data['caption']
        task_type = get_type(caption, instruction)
        

        masks = []
        mask_nums = []
        image2maskids = [None]*len(video_file)
        maskid = 0
        video_paths = []
        if 'masks' in data and data['masks'] is not None:
            mask_ids = data["mask_ids"]
            if 'height' in data:
                h = data['height']
                w = data['width']
            else:
                h = None
                w = None

            if isinstance(data['masks'], str):
                masks_ = json.load(open(data['masks']))
            else:
                masks_= data['masks']
            for ann in masks_:
                for k in ann.keys():
                    mask = annToMask(ann[k], h, w)
                    masks.append(mask)
                    image2maskids[mask_ids[maskid]] = maskid
                    maskid+=1

                mask_nums.append(len(ann.keys()))
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            masks = None
            image2maskids = None   
            mask_ids = None

        if self.only_mask_img:
            video_file = [video_file[i] for i in mask_ids]
            if isinstance(video_file, list) and len(video_file) == 1 and 'timestamps' not in data:
                video_file = os.path.join(data_folder, video_root, video_file[0])
                images, timestamps = load_video(video_file)
            elif isinstance(video_file, list): #images
                images = []
                for vf in video_file:
                    img_path = os.path.join(data_folder, video_root, vf)
                    images+=load_images(img_path)
                    video_paths.append(img_path)
                timestamps = data['timestamps']
            
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

            gt_masks = masks

        else:
            if isinstance(video_file, list) and len(video_file) == 1 and 'timestamps' not in data:
                video_file = os.path.join(data_folder, video_root, video_file[0])
                images, timestamps = load_video(video_file)
            elif isinstance(video_file, list): #images
                images = []
                for vf in video_file:
                    images+=load_images(os.path.join(data_folder, video_root, vf))
                    video_paths.append(os.path.join(data_folder, video_root, vf))
                timestamps = data['timestamps']
            
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

            gt_masks = torch.zeros((len(images), images[0].height, images[0].width))
            for i, mid in enumerate(mask_ids):
                gt_masks[mid] = masks[i]
        
        return {
            'idx': idx,
            'video': (images, timestamps),
            'masks': gt_masks,
            'instruction': instruction,
            'image2maskids': image2maskids,
            'type': task_type,
            'mask_ids': torch.tensor(mask_ids),
            'video_path': video_paths,

        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    ins = [x['instruction'] for x in batch]
    mid = [x['image2maskids'] for x in batch]
    typ = [x['type'] for x in batch]
    maskids = [x['mask_ids'] for x in batch]
    ip = [x['video_path'] for x in batch]
    return idx, vid, msk, ins, mid, typ, maskids, ip


def build_eval_dataloader(args, processor, distributed=False):
    # convert parquet to json
    questions = json.load(open(args.question_file))#[:10]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Mask_Dataset(args.video_folder, questions, only_mask_img=args.only_mask_img)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader

    

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
    for i, (idx, video, masks_, instruction, image2maskids, typ, mask_ids, image_paths) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        video_tensor = video[0]
        gt_masks = masks_[0]
        instruction = instruction[0]
        image2maskids = image2maskids[0]
        type_ = typ[0]

        mask_ids = mask_ids[0]
        video_path = image_paths[0]
        
        # try:
        output, masks, cls_scores = mm_infer_segmentation(
            video_tensor,
            processor,
            instruction,
            model=model,
            tokenizer=processor.tokenizer,
            do_sample=False,
            modal='video',
            video_path = video_path,
        )

        # gt_masks = np.array(gt_masks)
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

        masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        masks = masks.squeeze(1)>0
        print(masks.shape, gt_masks.shape)

        pred_masks = masks.detach().cpu().numpy()
        mask_rles = []
        for pred_mask in pred_masks:
            mask_rle = singleMask2rle(pred_mask)
            mask_rles.append(mask_rle)


        # from matplotlib import pyplot as plt
        # output_folder = f'visualization/stage3_maskonly/{idx}'
        # os.makedirs(output_folder, exist_ok=True)
        # for num, pm in enumerate(masks):
        #     plt.imshow(pm.detach().cpu().numpy())
        #     plt.savefig(os.path.join(output_folder, f'{num}_pred.png'))
        #     plt.imshow(gt_masks[num].detach().cpu().numpy())
        #     plt.savefig(os.path.join(output_folder, f'{num}_gt.png'))
        #     plt.imshow(video_tensor[0][num])
        #     plt.savefig(os.path.join(output_folder, f'{num}_rgb.png'))

        # print(output)
        record = {
            'idx': idx,
            'prediction': output,
            'instruction': instruction,
            'type': type_,
            'mask_rle': mask_rles,
        }
        iou_zero, iou_non_zero = calculate_iou(masks, gt_masks.to(masks))
        iou = calculate_iou_flatten(masks, gt_masks.to(masks)).item()
        record['iou'] = iou
        record['iou_non_zero'] = iou_non_zero.mean().item()
        if len(iou_zero) > 0:
            record['iou_zero'] = iou_zero.mean().item()
        
        # Ensure mask_ids are within valid range
        valid_mask_ids = mask_ids[mask_ids < len(masks)]
        if len(valid_mask_ids) > 0:
            f = db_eval_boundary(masks[valid_mask_ids].cpu().detach().numpy(), gt_masks[valid_mask_ids].cpu().detach().numpy()).mean()
            record['f'] = float(f)
        else:
            record['f'] = 0.0
        results.append(record)
        # except Exception as e:
        #     print(f"Data {i} Error: {e}")
            

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
            results = postprocess_seg_result(results)
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        results = postprocess_seg_result(results)
        save_results(results, args.output_file)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='', default='checkpoints/stage3_segmentation_0629_only_exist_with_313k_normal/5.2k_merge')
    parser.add_argument('--video_folder', help='Directory containing video files.', default='/mnt')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', default='/public/hz_oss/yunxuan/data/scene_understanding/eval_data/old_qa/seg_eval_4_room.json')
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='/mnt/damovl/scene_understanding/eval_data/human_qa_1st_eval_1.json')
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='visualization/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--only_mask_img", action='store_true')
    args = parser.parse_args()
    print(args)

    run_inference(args)


