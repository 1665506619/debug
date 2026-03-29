#!/usr/bin/env python3
"""
分析EgoMask推理结果，找出问题所在
"""
import json
import os
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

def analyze_predictions(pred_file, question_file, meta_file, annot_root):
    """分析预测结果"""
    
    # 加载数据
    print(f"加载预测结果: {pred_file}")
    with open(pred_file) as f:
        predictions = json.load(f)
    
    # 跳过metrics summary
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        predictions = predictions[1:]
    
    print(f"加载question文件: {question_file}")
    with open(question_file) as f:
        questions = json.load(f)
    
    print(f"加载meta文件: {meta_file}")
    with open(meta_file) as f:
        meta = json.load(f)
    
    # 统计变量
    stats = {
        'total_frames': 0,
        'gt_present_pred_present': 0,  # GT有+预测有
        'gt_present_pred_absent': 0,   # GT有+预测无
        'gt_absent_pred_present': 0,   # GT无+预测有（误报）
        'gt_absent_pred_absent': 0,    # GT无+预测无
        'pred_mask_areas': [],          # 预测mask的面积分布
    }
    
    print(f"\n开始分析 {len(predictions)} 个样本...")

    skipped_count = 0
    for pred in tqdm(predictions[:50], desc="分析中"):  # 只分析前50个样本
        idx = pred['idx']
        if idx >= len(questions):
            skipped_count += 1
            continue

        question_item = questions[idx]
        video_id = question_item['video_id']
        exp_id = question_item['exp_id']
        obj_id = question_item['obj_id']

        if video_id not in meta['videos']:
            skipped_count += 1
            print(f"\n警告: video_id {video_id} 不在meta中")
            continue
        
        meta_frames = meta['videos'][video_id]['frames']
        all_frame_names = question_item.get('frame_names', meta_frames)
        mask_rles = pred.get('mask_rle', [])

        if idx == 0:  # 打印第一个样本的信息
            print(f"\n[DEBUG] 第一个样本:")
            print(f"  video_id: {video_id}")
            print(f"  obj_id: {obj_id}")
            print(f"  meta_frames数量: {len(meta_frames)}")
            print(f"  all_frame_names数量: {len(all_frame_names)}")
            print(f"  mask_rles数量: {len(mask_rles)}")

        # 加载GT mask
        # Short subset的annotations目录使用完整的video_id（带后缀）
        gt_mask_file = os.path.join(annot_root, video_id, f'{obj_id}.json')

        try:
            with open(gt_mask_file) as f:
                gt_mask_dict = json.load(f)
        except Exception as e:
            if idx == 0:
                print(f"  [DEBUG] 无法加载GT: {e}")
            continue
        
        # 逐帧分析
        for i, frame_name in enumerate(all_frame_names):
            if i >= len(mask_rles):
                break
            
            stats['total_frames'] += 1
            
            # 检查GT是否存在
            has_gt = frame_name in gt_mask_dict and gt_mask_dict[frame_name] is not None
            
            # 检查预测是否存在
            pred_mask_rle = mask_rles[i]
            has_pred = False
            pred_area = 0
            
            if pred_mask_rle is not None and isinstance(pred_mask_rle, dict) and 'counts' in pred_mask_rle:
                try:
                    pred_mask_array = maskUtils.decode(pred_mask_rle)
                    pred_area = np.sum(pred_mask_array)
                    has_pred = pred_area > 0
                    if has_pred:
                        stats['pred_mask_areas'].append(pred_area)
                except:
                    pass
            
            # 统计四种情况
            if has_gt and has_pred:
                stats['gt_present_pred_present'] += 1
            elif has_gt and not has_pred:
                stats['gt_present_pred_absent'] += 1
            elif not has_gt and has_pred:
                stats['gt_absent_pred_present'] += 1
            elif not has_gt and not has_pred:
                stats['gt_absent_pred_absent'] += 1
    
    # 打印统计结果
    print(f"\n{'='*60}")
    print(f"分析结果（前50个样本）")
    print(f"{'='*60}")
    print(f"跳过的样本数: {skipped_count}")
    print(f"总帧数: {stats['total_frames']}")

    if stats['total_frames'] == 0:
        print("\n⚠️  没有分析到任何帧！请检查数据路径和格式。")
        return

    print(f"\n四种情况分布:")
    print(f"  GT有+预测有: {stats['gt_present_pred_present']:4d} ({stats['gt_present_pred_present']/stats['total_frames']*100:5.1f}%) ✓ 正确预测")
    print(f"  GT有+预测无: {stats['gt_present_pred_absent']:4d} ({stats['gt_present_pred_absent']/stats['total_frames']*100:5.1f}%) ✗ 漏检")
    print(f"  GT无+预测有: {stats['gt_absent_pred_present']:4d} ({stats['gt_absent_pred_present']/stats['total_frames']*100:5.1f}%) ✗ 误报（问题所在！）")
    print(f"  GT无+预测无: {stats['gt_absent_pred_absent']:4d} ({stats['gt_absent_pred_absent']/stats['total_frames']*100:5.1f}%) ✓ 正确")
    
    if stats['pred_mask_areas']:
        print(f"\n预测mask面积统计:")
        print(f"  最小: {min(stats['pred_mask_areas'])}")
        print(f"  最大: {max(stats['pred_mask_areas'])}")
        print(f"  平均: {np.mean(stats['pred_mask_areas']):.1f}")
        print(f"  中位数: {np.median(stats['pred_mask_areas']):.1f}")
    
    # 计算误报率
    false_positive_rate = stats['gt_absent_pred_present'] / (stats['gt_absent_pred_present'] + stats['gt_absent_pred_absent']) if (stats['gt_absent_pred_present'] + stats['gt_absent_pred_absent']) > 0 else 0
    print(f"\n误报率: {false_positive_rate*100:.1f}% (在GT无物体的帧中，预测有物体的比例)")
    
    print(f"\n{'='*60}")
    print(f"结论:")
    print(f"{'='*60}")
    if stats['gt_absent_pred_present'] > stats['total_frames'] * 0.1:
        print(f"⚠️  误报率过高！模型在GT无物体的帧也预测了mask")
        print(f"   可能原因:")
        print(f"   1. cls_scores没有正确过滤（总是>0）")
        print(f"   2. 模型没有学会判断物体是否存在")
        print(f"   3. 阈值设置不当")
    else:
        print(f"✓ 误报率正常，问题可能在其他地方")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="short")
    args = parser.parse_args()
    
    question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}.json'
    annot_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
    meta_file = os.path.join(annot_root, f'subset/{args.dataset_type}/meta_expressions.json')
    annot_root = os.path.join(annot_root, 'annotations')
    
    analyze_predictions(args.pred_file, question_file, meta_file, annot_root)

