#!/usr/bin/env python3
"""
后处理脚本：将模型推理结果转换为 EgoMask 评估格式

输入格式 (eval_video.py 的输出):
[
    {
        "idx": 0,
        "instruction": "...",
        "j": 0.xx,
        "f": 0.xx,
        "mask_rle": [{"size": [h, w], "counts": "..."}, ...],
        "video_path": "..."
    },
    ...
]

输出格式 (EgoMask 评估所需):
output_dir/
    video_id_1/
        exp_id_1/
            exp_id_1-obj_id.json  # {"frame_name": {"size": [h,w], "counts": "..."}, ...}
    video_id_2/
        ...
"""

import json
import os
import argparse
from tqdm import tqdm
from pycocotools import mask as maskUtils
import numpy as np


def main(args):
    print(f"正在加载推理结果: {args.pred_result}")
    with open(args.pred_result, 'r') as f:
        predictions = json.load(f)
    
    # 第一个元素可能是 metrics summary，跳过
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        print("检测到 metrics summary，跳过第一个元素")
        predictions = predictions[1:]
    
    # 加载对应的 question 文件以获取元信息
    # 检测是否为测试模式（文件名包含_test）
    is_test_mode = '_test' in os.path.basename(args.pred_result)
    
    if is_test_mode:
        question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}_test.json'
        print("检测到测试模式")
    else:
        question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}.json'
    
    print(f"正在加载 question 文件: {question_file}")
    
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # 加载 meta_expressions 以获取帧信息
    annotation_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
    if args.dataset_type == 'full':
        meta_file = os.path.join(annotation_root, 'meta_expressions.json')
    else:
        meta_file = os.path.join(annotation_root, f'subset/{args.dataset_type}/meta_expressions.json')
    
    print(f"正在加载 meta_expressions: {meta_file}")
    with open(meta_file, 'r') as f:
        meta_exp = json.load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n开始处理 {len(predictions)} 条预测结果...\n")
    
    total_frames = 0
    skipped_no_gt_frames = 0
    
    for pred in tqdm(predictions, desc="转换格式"):
        idx = pred['idx']
        
        # 从 questions 中获取元信息
        if idx >= len(questions):
            print(f"\n警告: idx {idx} 超出 questions 范围，跳过")
            continue
        
        question_item = questions[idx]
        if 'video_id' in question_item:
            video_id = question_item['video_id']
        else:
            video_id = question_item['video'].split('/')[-1]
        exp_id = question_item['exp_id']
        obj_id = question_item['obj_id']
        
        # 【核心逻辑】使用meta中的frames（官方评估遍历的）
        if video_id not in meta_exp['videos']:
            print(f"\n警告: video_id {video_id} 不在 meta_expressions 中，跳过")
            continue
        
        meta_frames = meta_exp['videos'][video_id]['frames']          # 官方评估遍历的帧
        # 现在 frame_names 已经是正确格式（Short: img前缀，Medium/Long: 绝对编号）
        all_frame_names = question_item.get('frame_names', meta_frames)
        
        # 获取预测的 mask RLE 列表
        mask_rles = pred.get('mask_rle', [])
        
        # 创建frame_name到mask的映射（用于查找）
        frame_to_mask = {}
        for i, fn in enumerate(all_frame_names):
            if i < len(mask_rles) and mask_rles[i] is not None:
                frame_to_mask[fn] = mask_rles[i]
        
        # 【关键】加载GT mask，用于判断帧是否有GT
        # 注意：只有Medium的annotations目录需要去掉后缀，Short和Long都不需要
        if args.dataset_type == 'medium' and '--' in video_id:
            annot_vid_name = video_id.split("--")[0]  # Medium: 去掉后缀
        else:
            annot_vid_name = video_id  # Short/Long: 保留原样
        gt_mask_file = os.path.join(annotation_root, 'annotations', annot_vid_name, f'{obj_id}.json')
        try:
            with open(gt_mask_file, 'r') as f:
                gt_mask_dict = json.load(f)
        except:
            print(f"\n警告: 无法加载GT mask: {gt_mask_file}")
            gt_mask_dict = {}
        
        # 构建 frame_name -> mask_rle 的映射
        # 遍历meta_frames（官方评估遍历的），从frame_to_mask中查找对应的mask
        mask_dict = {}
        
        for frame_name in meta_frames:
            # 从all_frame_names中查找这个frame的mask
            if frame_name not in frame_to_mask:
                continue  # 这个帧没有预测
            
            mask_rle = frame_to_mask[frame_name]
            has_gt = frame_name in gt_mask_dict
            
            try:
                if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                    # 解码RLE以计算面积
                    mask_array = maskUtils.decode(mask_rle)
                    mask_area = np.sum(mask_array)
                    
                    # 【核心逻辑】
                    if has_gt:
                        # 有GT的帧：保存所有预测（包括area=0）
                        mask_dict[frame_name] = mask_rle
                        total_frames += 1
                    else:
                        # 无GT的帧：只保存area>0的预测
                        if mask_area > 0:
                            mask_dict[frame_name] = mask_rle
                            total_frames += 1
                        else:
                            # area=0的预测不保存，让官方评估认为"无预测"
                            skipped_no_gt_frames += 1
                else:
                    # 如果格式不对，保守地保留
                    mask_dict[frame_name] = mask_rle
                    total_frames += 1
            except Exception as e:
                # 如果解码失败，保守地保留
                print(f"\n警告: 无法处理mask (idx {idx}, frame {frame_name}): {e}")
                mask_dict[frame_name] = mask_rle
                total_frames += 1
        
        # 创建输出目录结构: output_dir/video_id/exp_id/
        # 注意：这里必须使用完整的 video_id（带后缀），因为官方脚本用 vid_name 查找预测
        video_dir = os.path.join(args.output_dir, video_id)
        exp_dir = os.path.join(video_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存为 JSON 文件: {exp_id}-{obj_id}.json（官方脚本期望的文件名格式）
        output_file = os.path.join(exp_dir, f'{exp_id}-{obj_id}.json')
        with open(output_file, 'w') as f:
            json.dump(mask_dict, f)
    
    print(f"\n✓ 后处理完成！")
    print(f"结果保存到: {args.output_dir}")
    print(f"\n统计信息:")
    print(f"  - 保存的帧数: {total_frames}")
    print(f"  - 跳过的帧数（无GT且预测为0）: {skipped_no_gt_frames}")
    print(f"\n【逻辑】")
    print(f"  - 有GT的帧: 保存所有预测（包括预测为0）")
    print(f"  - 无GT的帧: 只保存预测>0的（预测=0的不保存，iou_overall算1.0）")
    print(f"\n可以使用以下命令进行评估:")
    print(f"python EgoMask-main/evaluation/eval_egomask.py \\")
    print(f"    --dataset_type {args.dataset_type} \\")
    print(f"    --pred_path {os.path.dirname(args.output_dir)} \\")
    print(f"    --save_name results_{args.dataset_type}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将模型推理结果转换为 EgoMask 评估格式")
    parser.add_argument("--pred_result", type=str, required=True,
                        help="模型推理结果的 JSON 文件路径 (eval_video.py 的输出)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录路径 (EgoMask 评估格式)")
    parser.add_argument("--dataset_type", type=str, default="long",
                        choices=["long", "medium", "short", "full"],
                        help="数据集类型")
    args = parser.parse_args()
    
    main(args)
