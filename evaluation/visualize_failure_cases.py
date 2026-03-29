#!/usr/bin/env python3
"""
可视化失败案例：
1. False Negative: 预测空但GT非空
2. Low IoU True Positive: 预测非空且GT非空但IoU很低
"""

import json
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
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


def overlay_mask(image, mask, color, alpha=0.5):
    """在图像上叠加mask"""
    if mask is None or np.sum(mask) == 0:
        return image
    
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)


def visualize_frames(video_id, exp_id, obj_id, expression, frames_info, 
                     image_root, output_path, case_type):
    """
    可视化一组帧
    
    Args:
        frames_info: list of (frame_name, gt_mask, pred_mask, iou)
    """
    n_frames = len(frames_info)
    if n_frames == 0:
        return False
    
    # 创建图像（3行：原图、GT、预测）- 高清大图
    fig, axes = plt.subplots(3, n_frames, figsize=(6 * n_frames, 18))
    if n_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (frame_name, gt_mask, pred_mask, iou) in enumerate(frames_info):
        # 加载图像 - 保持原始分辨率
        image_path = os.path.join(image_root, video_id, f'{frame_name}.jpg')
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.ones((360, 640, 3), dtype=np.uint8) * 255
        
        # 第一行：原始图像
        axes[0, col_idx].imshow(image)
        axes[0, col_idx].set_title(f'{frame_name}', fontsize=14)
        axes[0, col_idx].axis('off')

        # 第二行：GT mask（绿色）
        if gt_mask is not None and np.sum(gt_mask) > 0:
            gt_overlay = overlay_mask(image, gt_mask, [0, 255, 0], alpha=0.5)
            axes[1, col_idx].imshow(gt_overlay)
            gt_area = np.sum(gt_mask > 0)
            axes[1, col_idx].set_title(f'GT\nArea: {gt_area}px', fontsize=14, color='green')
        else:
            axes[1, col_idx].imshow(image)
            axes[1, col_idx].set_title('GT: None', fontsize=14, color='gray')
        axes[1, col_idx].axis('off')

        # 第三行：预测mask（红色）
        if pred_mask is not None and np.sum(pred_mask) > 0:
            pred_overlay = overlay_mask(image, pred_mask, [255, 0, 0], alpha=0.5)
            axes[2, col_idx].imshow(pred_overlay)
            pred_area = np.sum(pred_mask > 0)
            axes[2, col_idx].set_title(f'Pred\nArea: {pred_area}px\nIoU: {iou:.3f}',
                                      fontsize=14, color='red')
        else:
            axes[2, col_idx].imshow(image)
            axes[2, col_idx].set_title('Pred: None', fontsize=14, color='red')
        axes[2, col_idx].axis('off')
    
    # 设置总标题 - 增大字体
    title = f'{case_type}\nVideo: {video_id} | Exp {exp_id} (obj {obj_id})\nExpression: {expression}'
    fig.suptitle(title, fontsize=16, y=0.99, wrap=True)
    
    plt.tight_layout()
    
    # 保存 - 高DPI
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # 强制垃圾回收
    import gc
    gc.collect()
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--question_file', type=str, required=True)
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_fn_samples', type=int, default=10, help='False Negative样本数')
    parser.add_argument('--num_low_iou_samples', type=int, default=10, help='Low IoU TP样本数')
    parser.add_argument('--max_frames', type=int, default=4, help='每个样本最多显示的帧数')
    parser.add_argument('--low_iou_threshold', type=float, default=0.3, help='低IoU阈值')
    args = parser.parse_args()
    
    # 加载数据
    print(f'加载数据...')
    predictions = load_json(args.pred_file)
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        predictions = predictions[1:]
    
    questions = load_json(args.question_file)
    meta_exp = load_json(args.meta_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
                fn_frames.append((frame_name, gt_mask, pred_mask, iou))
            
            # 收集Low IoU TP帧
            if not pred_is_empty and not gt_is_empty and iou < args.low_iou_threshold:
                low_iou_frames.append((frame_name, gt_mask, pred_mask, iou))
        
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
            avg_iou = np.mean([f[3] for f in low_iou_frames])
            low_iou_cases.append({
                'video_id': video_id,
                'exp_id': exp_id,
                'obj_id': obj_id,
                'expression': expression,
                'frames': low_iou_frames,
                'avg_iou': avg_iou
            })
    
    print(f'\n找到 {len(fn_cases)} 个False Negative样本')
    print(f'找到 {len(low_iou_cases)} 个Low IoU TP样本')
    
    # 可视化False Negative案例
    print(f'\n可视化False Negative案例...')
    fn_cases.sort(key=lambda x: x['num_fn'], reverse=True)  # 按FN帧数排序
    
    for i, case in enumerate(tqdm(fn_cases[:args.num_fn_samples])):
        # 选择要显示的帧
        frames = case['frames']
        if len(frames) > args.max_frames:
            indices = np.linspace(0, len(frames) - 1, args.max_frames, dtype=int)
            selected_frames = [frames[idx] for idx in indices]
        else:
            selected_frames = frames
        
        output_path = os.path.join(args.output_dir, 
                                   f'FN_{i+1}_{case["video_id"]}_exp{case["exp_id"]}.png')
        visualize_frames(case['video_id'], case['exp_id'], case['obj_id'], 
                        case['expression'], selected_frames, 
                        args.image_root, output_path, 
                        f'False Negative ({len(frames)} FN frames)')
    
    # 可视化Low IoU TP案例
    print(f'\n可视化Low IoU TP案例...')
    low_iou_cases.sort(key=lambda x: x['avg_iou'])  # 按平均IoU从低到高排序
    
    for i, case in enumerate(tqdm(low_iou_cases[:args.num_low_iou_samples])):
        # 选择要显示的帧
        frames = case['frames']
        if len(frames) > args.max_frames:
            indices = np.linspace(0, len(frames) - 1, args.max_frames, dtype=int)
            selected_frames = [frames[idx] for idx in indices]
        else:
            selected_frames = frames
        
        output_path = os.path.join(args.output_dir, 
                                   f'LowIoU_{i+1}_{case["video_id"]}_exp{case["exp_id"]}.png')
        visualize_frames(case['video_id'], case['exp_id'], case['obj_id'], 
                        case['expression'], selected_frames, 
                        args.image_root, output_path, 
                        f'Low IoU TP (avg IoU={case["avg_iou"]:.3f})')
    
    print(f'\n完成！可视化结果保存在 {args.output_dir}')


if __name__ == '__main__':
    main()

