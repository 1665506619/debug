"""
分析TF类型失败案例（GT有物体但模型没预测）
检查三个问题：
1. 置信度阈值是否太高
2. 模型是否真的在预测
3. 可视化TF样本看看为什么漏检
"""

import json
import os
import sys
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from collections import defaultdict

def analyze_tf_failures(result_file, question_file, gt_dir, image_dir, output_dir, num_samples=10):
    """
    分析TF类型失败案例
    """
    
    print('='*80)
    print('TF失败案例分析')
    print('='*80)
    
    # 加载结果
    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # 加载question文件
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # 创建video_id到question的映射
    vid_to_question = {}
    for q in questions:
        key = f"{q['video_id']}_{q['exp_id']}_{q['obj_id']}"
        vid_to_question[key] = q
    
    # 统计TF情况
    tf_samples = []
    
    for result in results:
        seq_exp = result['seq_exp']
        gt_temporal = result['T']['gt_temporal']
        res_temporal = result['T']['res_temporal']
        
        # 统计TF帧数
        tf_count = sum(1 for gt, pred in zip(gt_temporal, res_temporal) if gt and not pred)
        total_gt = sum(gt_temporal)
        
        if tf_count > 0:
            tf_samples.append({
                'seq_exp': seq_exp,
                'tf_count': tf_count,
                'total_gt': total_gt,
                'tf_ratio': tf_count / total_gt if total_gt > 0 else 0,
                'gt_temporal': gt_temporal,
                'res_temporal': res_temporal,
                'gold_iou': result['gold_iou']
            })
    
    # 按TF比例排序
    tf_samples.sort(key=lambda x: x['tf_ratio'], reverse=True)
    
    print(f'\n总样本数: {len(results)}')
    print(f'有TF的样本数: {len(tf_samples)} ({len(tf_samples)/len(results)*100:.1f}%)')
    print(f'\nTF比例最高的{num_samples}个样本:')
    print(f'{"序号":<6} {"样本ID":<50} {"TF帧数":<10} {"GT帧数":<10} {"TF比例":<10} {"gold_iou":<10}')
    print('-'*110)
    
    for i, sample in enumerate(tf_samples[:num_samples]):
        print(f'{i+1:<6} {sample["seq_exp"]:<50} {sample["tf_count"]:<10} {sample["total_gt"]:<10} {sample["tf_ratio"]*100:>8.1f}% {sample["gold_iou"]*100:>8.1f}%')
    
    # 可视化前几个样本
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\n开始可视化前{min(num_samples, len(tf_samples))}个TF样本...')
    
    for i, sample in enumerate(tf_samples[:num_samples]):
        seq_exp = sample['seq_exp']
        
        # 解析seq_exp
        parts = seq_exp.split('_')
        if len(parts) >= 3:
            video_id = '_'.join(parts[:-2])
            exp_id = parts[-2]
            obj_id = parts[-1]
        else:
            print(f'无法解析seq_exp: {seq_exp}')
            continue
        
        # 获取question信息
        key = f"{video_id}_{exp_id}_{obj_id}"
        if key not in vid_to_question:
            print(f'找不到question: {key}')
            continue
        
        question = vid_to_question[key]
        expression = question.get('expression', 'Unknown')
        frame_names = question['frame_names']
        
        # 加载GT masks
        gt_file = os.path.join(gt_dir, f'{video_id}/{obj_id}.json')
        if not os.path.exists(gt_file):
            print(f'GT文件不存在: {gt_file}')
            continue
        
        with open(gt_file, 'r') as f:
            gt_mask_rle = json.load(f)
        
        # 找TF帧（GT有但预测无）
        tf_frames = []
        for j, (fname, gt, pred) in enumerate(zip(frame_names, sample['gt_temporal'], sample['res_temporal'])):
            if gt and not pred:
                tf_frames.append((j, fname))
        
        # 选择第1个TF帧可视化
        print(f'\n样本 {i+1}: {seq_exp}')
        print(f'  Expression: {expression}')
        print(f'  TF帧数: {len(tf_frames)}/{len(frame_names)}')

        if len(tf_frames) == 0:
            continue

        # 只可视化第一个TF帧
        frame_idx, fname = tf_frames[0]

        img_path = os.path.join(image_dir, f'{video_id}/{fname}.jpg')
        if not os.path.exists(img_path):
            print(f'  图片不存在: {img_path}')
            continue

        img = cv2.imread(img_path)

        # 叠加GT mask
        if fname in gt_mask_rle:
            gt_mask = maskUtils.decode(gt_mask_rle[fname])
            img[gt_mask > 0] = (img[gt_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)

        # 添加文字
        cv2.putText(img, f'Sample {i+1}: {expression[:60]}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Frame {frame_idx}: {fname}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, 'GT: YES (green), Pred: NO', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f'TF: {sample["tf_ratio"]*100:.0f}%, IoU: {sample["gold_iou"]*100:.1f}%', (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 保存
        output_path = os.path.join(output_dir, f'tf_sample_{i+1}_{video_id[:30]}.jpg')
        cv2.imwrite(output_path, img)
        print(f'  保存到: {output_path}')
    
    print(f'\n可视化完成！共生成{min(num_samples, len(tf_samples))}张图片')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True, help='结果文件路径')
    parser.add_argument('--question_file', type=str, required=True, help='Question文件路径')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT annotations目录')
    parser.add_argument('--image_dir', type=str, required=True, help='图片目录')
    parser.add_argument('--output_dir', type=str, default='tf_analysis', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='可视化样本数')
    
    args = parser.parse_args()
    
    analyze_tf_failures(
        args.result_file,
        args.question_file,
        args.gt_dir,
        args.image_dir,
        args.output_dir,
        args.num_samples
    )

