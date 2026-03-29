#!/usr/bin/env python3
"""
收集失败案例信息（不做可视化，只输出JSON）
"""

import json
import os
import argparse
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def decode_rle(rle):
    """解码RLE格式的mask"""
    if rle is None:
        return None
    try:
        return maskUtils.decode(rle)
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--question_file', type=str, required=True)
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_fn_samples', type=int, default=15)
    parser.add_argument('--num_low_iou_samples', type=int, default=15)
    parser.add_argument('--low_iou_threshold', type=float, default=0.3)
    args = parser.parse_args()
    
    # 加载数据
    print(f'加载数据...')
    predictions = load_json(args.pred_file)
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        predictions = predictions[1:]
    
    questions = load_json(args.question_file)
    meta_exp = load_json(args.meta_file)
    
    # 收集两种失败案例
    print(f'\n收集失败案例...')
    fn_cases = []  # False Negative cases
    low_iou_cases = []  # Low IoU True Positive cases
    
    empty_mask_patterns = ['PP\\9', 'PPQ7', 'PP\\\\9']
    
    for sample in tqdm(predictions):
        idx = sample['idx']
        if idx >= len(questions):
            continue
        
        question_item = questions[idx]
        video_id = question_item['video_id']
        exp_id = question_item['exp_id']
        obj_id = question_item['obj_id']
        expression = question_item.get('expression', 'Unknown')
        
        if video_id not in meta_exp['videos']:
            continue
        
        mask_rles = sample.get('mask_rle', [])
        
        # 加载GT
        gt_mask_file = os.path.join(args.annotation_root, 'annotations', video_id, f'{obj_id}.json')
        try:
            with open(gt_mask_file, 'r') as f:
                gt_mask_dict = json.load(f)
        except:
            continue
        
        meta_frames = meta_exp['videos'][video_id]['frames']
        all_frame_names = question_item.get('frame_names', meta_frames)
        
        # 收集该样本的FN和Low IoU帧
        fn_frames = []
        low_iou_frames = []
        
        for i, frame_name in enumerate(all_frame_names):
            if i >= len(mask_rles):
                break
            
            # 加载GT
            gt_mask = None
            if frame_name in gt_mask_dict and gt_mask_dict[frame_name] is not None:
                gt_mask = decode_rle(gt_mask_dict[frame_name])
            
            gt_is_empty = (gt_mask is None or np.sum(gt_mask) == 0)
            
            # 加载预测
            pred_rle = mask_rles[i]
            pred_mask = None
            if pred_rle is not None and pred_rle.get('counts', '') not in empty_mask_patterns:
                pred_mask = decode_rle(pred_rle)
            
            pred_is_empty = (pred_mask is None or np.sum(pred_mask) == 0)
            
            # 计算IoU
            iou = 0.0
            if not pred_is_empty and not gt_is_empty:
                intersection = np.sum((pred_mask > 0) & (gt_mask > 0))
                union = np.sum((pred_mask > 0) | (gt_mask > 0))
                if union > 0:
                    iou = intersection / union
            
            # 收集FN帧
            if pred_is_empty and not gt_is_empty:
                fn_frames.append({
                    'frame_name': frame_name,
                    'frame_idx': i,
                    'gt_area': int(np.sum(gt_mask > 0)) if gt_mask is not None else 0,
                    'iou': 0.0
                })
            
            # 收集Low IoU TP帧
            if not pred_is_empty and not gt_is_empty and iou < args.low_iou_threshold:
                low_iou_frames.append({
                    'frame_name': frame_name,
                    'frame_idx': i,
                    'gt_area': int(np.sum(gt_mask > 0)),
                    'pred_area': int(np.sum(pred_mask > 0)),
                    'iou': float(iou)
                })
        
        # 如果有FN帧，添加到案例列表
        if len(fn_frames) > 0:
            fn_cases.append({
                'video_id': video_id,
                'exp_id': exp_id,
                'obj_id': obj_id,
                'expression': expression,
                'frames': fn_frames,
                'num_fn': len(fn_frames)
            })
        
        # 如果有Low IoU帧，添加到案例列表
        if len(low_iou_frames) > 0:
            avg_iou = np.mean([f['iou'] for f in low_iou_frames])
            low_iou_cases.append({
                'video_id': video_id,
                'exp_id': exp_id,
                'obj_id': obj_id,
                'expression': expression,
                'frames': low_iou_frames,
                'avg_iou': float(avg_iou)
            })
    
    print(f'\n找到 {len(fn_cases)} 个False Negative样本')
    print(f'找到 {len(low_iou_cases)} 个Low IoU TP样本')
    
    # 排序
    fn_cases.sort(key=lambda x: x['num_fn'], reverse=True)
    low_iou_cases.sort(key=lambda x: x['avg_iou'])
    
    # 输出到JSON
    output_data = {
        'false_negative_cases': fn_cases[:args.num_fn_samples],
        'low_iou_cases': low_iou_cases[:args.num_low_iou_samples]
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\n结果已保存到 {args.output_file}')
    
    # 打印一些统计信息
    print(f'\n=== False Negative 案例（前{min(5, len(fn_cases))}个）===')
    for i, case in enumerate(fn_cases[:5]):
        print(f'{i+1}. Video: {case["video_id"]}, Exp: {case["exp_id"]}, FN帧数: {case["num_fn"]}')
        print(f'   Expression: {case["expression"][:80]}...')
    
    print(f'\n=== Low IoU TP 案例（前{min(5, len(low_iou_cases))}个）===')
    for i, case in enumerate(low_iou_cases[:5]):
        print(f'{i+1}. Video: {case["video_id"]}, Exp: {case["exp_id"]}, 平均IoU: {case["avg_iou"]:.3f}')
        print(f'   Expression: {case["expression"][:80]}...')


if __name__ == '__main__':
    main()

