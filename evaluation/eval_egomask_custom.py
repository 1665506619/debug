#!/usr/bin/env python3
"""
自定义 EgoMask 评估脚本
适配从 eval_video.py 输出的 JSON 格式

输入格式:
[
    {
        "idx": 0,
        "instruction": "...",
        "j": 0.xx,
        "f": 0.xx,
        "mask_rle": [{"size": [h, w], "counts": "..."}, ...],
        "video_path": "...",
        ...
    },
    ...
]

评估指标:
- J: Region Similarity (IoU)
- F: Contour Accuracy (F-measure)
- J&F: 综合指标
- iou_overall: 所有帧的平均 IoU
- iou_gold: 有 GT 的帧的平均 IoU
- T_*: 时序检测指标
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import numpy as np
import pycocotools.mask as maskUtils
from tqdm import tqdm

# 导入评估指标函数
from evaluation.metrics import db_eval_iou, db_eval_boundary, db_eval_boundary_temporal


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def evaluate_single_sample(pred_item, question_item, meta_exp, annotation_root, dataset_type):
    """
    评估单个样本
    
    Args:
        pred_item: 预测结果字典
        question_item: question 文件中的元信息
        meta_exp: meta_expressions 字典
        annotation_root: GT annotations 根目录
        dataset_type: 数据集类型 (short/medium/long)
    
    Returns:
        评估结果字典
    """
    video_id = question_item['video_id']
    exp_id = question_item['exp_id']
    obj_id = question_item['obj_id']
    
    # 获取帧列表
    if video_id not in meta_exp['videos']:
        print(f"警告: video_id {video_id} 不在 meta_expressions 中")
        return None
    
    vid_info = meta_exp['videos'][video_id]
    meta_frames = vid_info['frames']
    all_frame_names = question_item.get('frame_names', meta_frames)
    
    # 加载 GT masks
    if dataset_type == 'medium' and '--' in video_id:
        annot_vid_name = video_id.split("--")[0]
    else:
        annot_vid_name = video_id
    
    gt_mask_file = os.path.join(annotation_root, 'annotations', annot_vid_name, f'{obj_id}.json')
    try:
        with open(gt_mask_file, 'r') as f:
            gt_mask_dict = json.load(f)
    except:
        print(f"警告: 无法加载 GT mask: {gt_mask_file}")
        return None
    
    # 获取图像尺寸
    h, w = None, None
    for fname, mask_rle in gt_mask_dict.items():
        if mask_rle is not None:
            mask = maskUtils.decode(mask_rle)
            h, w = mask.shape
            break
    
    if h is None or w is None:
        print(f"警告: 无法获取图像尺寸 for {video_id}")
        return None
    
    # 准备 mask 数组
    vid_len = len(meta_frames)
    gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
    pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
    
    # 获取预测的 mask RLE 列表
    mask_rles = pred_item.get('mask_rle', [])
    
    # 创建 frame_name 到 mask 的映射
    frame_to_mask = {}
    for i, fn in enumerate(all_frame_names):
        if i < len(mask_rles) and mask_rles[i] is not None:
            frame_to_mask[fn] = mask_rles[i]
    
    # IoU 列表
    overall_iou_list = []
    gold_iou_list = []
    gold_with_pred_iou_list = []
    
    # 遍历所有帧
    for fidx, fname in enumerate(meta_frames):
        # 加载 GT mask
        if fname in gt_mask_dict:
            gold_segm = gt_mask_dict[fname]
            if gold_segm is not None:
                mask = maskUtils.decode(gold_segm)
                gt_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)
        
        # 加载预测 mask
        if fname in frame_to_mask:
            pred_segm = frame_to_mask[fname]
            if pred_segm is not None and isinstance(pred_segm, dict) and 'counts' in pred_segm:
                try:
                    mask = maskUtils.decode(pred_segm)
                    pred_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)
                except:
                    pass
        
        # 计算 IoU
        overlap = np.logical_and(gt_masks[fidx], pred_masks[fidx])
        union = np.logical_or(gt_masks[fidx], pred_masks[fidx])
        
        if union.sum() > 0:
            iou_ = overlap.sum() / union.sum()
            overall_iou_list.append(iou_)
        else:
            # 没有 GT 也没有预测
            iou_ = 1.0
            overall_iou_list.append(iou_)
        
        # 分类统计
        has_gt = fname in gt_mask_dict and gt_mask_dict[fname] is not None
        has_pred = fname in frame_to_mask and frame_to_mask[fname] is not None

        if has_gt:
            gold_iou_list.append(iou_)
            gold_with_pred_iou_list.append(iou_)
        elif has_pred:
            # 只有预测，没有 GT，iou_ = 0.0
            gold_with_pred_iou_list.append(iou_)
    
    # 计算评估指标
    try:
        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        t = db_eval_boundary_temporal(gt_masks, pred_masks)
    except Exception as e:
        print(f"警告: 计算指标失败 for {video_id}: {e}")
        j = 0.0
        f = 0.0
        t = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    # 计算 IoU 指标
    overall_iou = np.mean(overall_iou_list) if overall_iou_list else 0.0
    gold_iou = np.mean(gold_iou_list) if gold_iou_list else 0.0
    gold_with_pred_iou = np.mean(gold_with_pred_iou_list) if gold_with_pred_iou_list else 0.0
    
    exp_name = f"{video_id}_{exp_id}_{obj_id}"
    
    return {
        "exp_name": exp_name,
        "J": j,
        "F": f,
        "T": t,
        "overall_iou": overall_iou,
        "gold_iou": gold_iou,
        "gold_with_pred_iou": gold_with_pred_iou,
    }


def main(args):
    print(f"正在加载预测结果: {args.pred_result}")
    with open(args.pred_result, 'r') as f:
        predictions = json.load(f)
    
    # 跳过第一个元素（如果是 metrics summary）
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        print("检测到 metrics summary，跳过第一个元素")
        predictions = predictions[1:]
    
    print(f"共有 {len(predictions)} 条预测结果")
    
    # 加载 question 文件
    question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}.json'
    print(f"正在加载 question 文件: {question_file}")
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # 加载 meta_expressions
    annotation_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
    if args.dataset_type == 'full':
        meta_file = os.path.join(annotation_root, 'meta_expressions.json')
    else:
        meta_file = os.path.join(annotation_root, f'subset/{args.dataset_type}/meta_expressions.json')
    
    print(f"正在加载 meta_expressions: {meta_file}")
    with open(meta_file, 'r') as f:
        meta_exp = json.load(f)
    
    # 评估所有样本
    print(f"\n开始评估 {len(predictions)} 个样本...\n")
    
    results_list = []
    for pred in tqdm(predictions, desc="评估进度"):
        idx = pred['idx']
        
        if idx >= len(questions):
            print(f"\n警告: idx {idx} 超出 questions 范围，跳过")
            continue
        
        question_item = questions[idx]
        
        result = evaluate_single_sample(
            pred, question_item, meta_exp, annotation_root, args.dataset_type
        )
        
        if result is not None:
            results_list.append(result)
    
    print(f"\n成功评估 {len(results_list)} 个样本")
    
    if len(results_list) == 0:
        print("❌ 错误: 没有成功评估的样本")
        return
    
    # 汇总统计
    j_list = [r["J"] for r in results_list]
    f_list = [r["F"] for r in results_list]
    t_f1_list = [r["T"]["f1"] for r in results_list]
    t_acc_list = [r["T"]["accuracy"] for r in results_list]
    t_precision_list = [r["T"]["precision"] for r in results_list]
    t_recall_list = [r["T"]["recall"] for r in results_list]

    iou_overall_list = [r["overall_iou"] for r in results_list]
    iou_gold_list = [r["gold_iou"] for r in results_list]
    iou_gold_with_pred_list = [r["gold_with_pred_iou"] for r in results_list]

    # 过滤掉 NaN 值
    def safe_mean(lst):
        valid_values = [x for x in lst if not np.isnan(x)]
        return np.mean(valid_values) if valid_values else 0.0

    summary = {
        "J": round(100 * float(safe_mean(j_list)), 2),
        "F": round(100 * float(safe_mean(f_list)), 2),
        "J&F": round(100 * float((safe_mean(j_list) + safe_mean(f_list)) / 2), 2),
        "T_f1": round(100 * float(safe_mean(t_f1_list)), 2),
        "T_acc": round(100 * float(safe_mean(t_acc_list)), 2),
        "T_precision": round(100 * float(safe_mean(t_precision_list)), 2),
        "T_recall": round(100 * float(safe_mean(t_recall_list)), 2),
        "iou_overall": round(100 * float(safe_mean(iou_overall_list)), 2),
        "iou_gold": round(100 * float(safe_mean(iou_gold_list)), 2),
        "iou_gold_with_pred": round(100 * float(safe_mean(iou_gold_with_pred_list)), 2),
        "num_samples": len(results_list),
    }
    
    # 保存结果
    output_file = args.output_file
    print(f"\n保存汇总结果到: {output_file}")
    write_json(summary, output_file)
    
    # 保存详细结果
    detailed_output = output_file.replace('.json', '_detailed.json')
    print(f"保存详细结果到: {detailed_output}")
    write_json(results_list, detailed_output)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    for key, value in summary.items():
        if key != "num_samples":
            print(f"  {key:20s}: {value:6.2f}")
        else:
            print(f"  {key:20s}: {value}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 EgoMask 预测结果")
    parser.add_argument("--pred_result", type=str, required=True,
                        help="预测结果 JSON 文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出结果文件路径")
    parser.add_argument("--dataset_type", type=str, default="short",
                        choices=["long", "medium", "short", "full"],
                        help="数据集类型")
    args = parser.parse_args()
    
    main(args)

