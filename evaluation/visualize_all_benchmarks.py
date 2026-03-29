"""
可视化所有benchmark的结果
包括：EgoMask, LISA, RefCOCO等
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_egomask_results(result_file):
    """加载EgoMask结果"""
    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # 统计
    J_mean = np.mean([r['J'] for r in results])
    F_mean = np.mean([r['F'] for r in results])

    stats = {
        'name': 'EgoMask SHORT',
        'num_samples': len(results),
        'J': J_mean,
        'F': F_mean,
        'J&F': (J_mean + F_mean) / 2,  # 计算J&F
        'iou_overall': np.mean([r['overall_iou'] for r in results]),
        'iou_gold': np.mean([r['gold_iou'] for r in results]),
        'iou_gold_with_pred': np.mean([r['gold_with_pred_iou'] for r in results]),
        'T_acc': np.mean([r['T']['accuracy'] for r in results]),
        'T_precision': np.mean([r['T']['precision'] for r in results]),
        'T_recall': np.mean([r['T']['recall'] for r in results]),
        'T_f1': np.mean([r['T']['f1'] for r in results]),
    }
    
    # 四种类型统计
    TT, TF, FT, FF = 0, 0, 0, 0
    for r in results:
        gt_temporal = r['T']['gt_temporal']
        res_temporal = r['T']['res_temporal']
        for gt, pred in zip(gt_temporal, res_temporal):
            if gt and pred:
                TT += 1
            elif gt and not pred:
                TF += 1
            elif not gt and pred:
                FT += 1
            else:
                FF += 1
    
    total = TT + TF + FT + FF
    stats['TT'] = TT
    stats['TF'] = TF
    stats['FT'] = FT
    stats['FF'] = FF
    stats['TT_pct'] = TT / total * 100
    stats['TF_pct'] = TF / total * 100
    stats['FT_pct'] = FT / total * 100
    stats['FF_pct'] = FF / total * 100
    
    return stats

def load_lisa_results(result_file):
    """加载LISA结果"""
    with open(result_file, 'r') as f:
        content = f.read()
    
    # 尝试解析为JSON数组
    try:
        results = json.loads(content)
    except:
        # 如果失败，尝试逐行解析
        results = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except:
                    pass
    
    if not results:
        return None
    
    # 统计
    stats = {
        'name': os.path.basename(result_file).replace('.json', ''),
        'num_samples': len(results),
    }
    
    # 计算平均值
    for key in ['giou', 'ciou']:
        if key in results[0]:
            stats[key] = np.mean([r[key] for r in results if key in r])
    
    return stats

def visualize_benchmarks(egomask_stats, lisa_val_stats, lisa_test_stats, output_dir):
    """可视化所有benchmark结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. EgoMask详细统计
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 四种类型分布
    ax = axes[0, 0]
    labels = ['TT\n(GT有+Pred有)', 'TF\n(GT有+Pred无)', 'FT\n(GT无+Pred有)', 'FF\n(GT无+Pred无)']
    sizes = [egomask_stats['TT_pct'], egomask_stats['TF_pct'], 
             egomask_stats['FT_pct'], egomask_stats['FF_pct']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10})
    ax.set_title('EgoMask: 帧类型分布', fontsize=14, fontweight='bold')
    
    # 1.2 IoU指标对比
    ax = axes[0, 1]
    iou_metrics = ['iou_overall', 'iou_gold', 'iou_gold_with_pred']
    iou_values = [egomask_stats[m] * 100 for m in iou_metrics]
    iou_labels = ['Overall IoU\n(所有帧)', 'Gold IoU\n(只GT帧)', 'Gold+Pred IoU\n(GT或Pred帧)']
    
    bars = ax.bar(range(len(iou_metrics)), iou_values, color=['#3498db', '#e74c3c', '#f39c12'])
    ax.set_xticks(range(len(iou_metrics)))
    ax.set_xticklabels(iou_labels, fontsize=10)
    ax.set_ylabel('IoU (%)', fontsize=12)
    ax.set_title('EgoMask: IoU指标对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(iou_values) * 1.2])
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, iou_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 1.3 J&F指标
    ax = axes[1, 0]
    jf_metrics = ['J', 'F', 'J&F']
    jf_values = [egomask_stats[m] * 100 for m in jf_metrics]
    jf_labels = ['J\n(Region Similarity)', 'F\n(Boundary Accuracy)', 'J&F\n(Average)']
    
    bars = ax.bar(range(len(jf_metrics)), jf_values, color=['#9b59b6', '#1abc9c', '#34495e'])
    ax.set_xticks(range(len(jf_metrics)))
    ax.set_xticklabels(jf_labels, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('EgoMask: J&F指标', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(jf_values) * 1.2])
    
    for i, (bar, val) in enumerate(zip(bars, jf_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 1.4 Temporal指标
    ax = axes[1, 1]
    t_metrics = ['T_acc', 'T_precision', 'T_recall', 'T_f1']
    t_values = [egomask_stats[m] * 100 for m in t_metrics]
    t_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    bars = ax.bar(range(len(t_metrics)), t_values, color=['#16a085', '#27ae60', '#2980b9', '#8e44ad'])
    ax.set_xticks(range(len(t_metrics)))
    ax.set_xticklabels(t_labels, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('EgoMask: Temporal Grounding指标', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(t_values) * 1.2])
    
    for i, (bar, val) in enumerate(zip(bars, t_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'egomask_detailed_stats.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'保存EgoMask详细统计: {output_path}')
    plt.close()
    
    # 2. LISA结果对比
    if lisa_val_stats and lisa_test_stats:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['giou', 'ciou']
        val_values = [lisa_val_stats.get(m, 0) * 100 for m in metrics]
        test_values = [lisa_test_stats.get(m, 0) * 100 for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_values, width, label='Val', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('LISA: Val vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['GIoU', 'CIoU'], fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, max(val_values + test_values) * 1.2])
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'lisa_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'保存LISA对比: {output_path}')
        plt.close()
    
    # 3. 综合对比
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    benchmarks = ['EgoMask\n(gold_iou)', 'EgoMask\n(J&F)', 'LISA Val\n(GIoU)', 'LISA Test\n(GIoU)']
    scores = [
        egomask_stats['iou_gold'] * 100,
        egomask_stats['J&F'] * 100,
        lisa_val_stats.get('giou', 0) * 100 if lisa_val_stats else 0,
        lisa_test_stats.get('giou', 0) * 100 if lisa_test_stats else 0,
    ]
    colors = ['#e74c3c', '#34495e', '#3498db', '#e74c3c']
    
    bars = ax.bar(range(len(benchmarks)), scores, color=colors)
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Benchmark综合对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(scores) * 1.2])
    
    for i, (bar, val) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'保存综合对比: {output_path}')
    plt.close()
    
    # 4. 打印文本报告
    print('\n' + '='*80)
    print('BENCHMARK结果总结')
    print('='*80)
    
    print(f'\n【EgoMask SHORT】')
    print(f'  样本数: {egomask_stats["num_samples"]}')
    print(f'  J&F: {egomask_stats["J&F"]*100:.2f}%')
    print(f'  gold_iou: {egomask_stats["iou_gold"]*100:.2f}%')
    print(f'  Temporal F1: {egomask_stats["T_f1"]*100:.2f}%')
    print(f'  帧类型: TT={egomask_stats["TT_pct"]:.1f}%, TF={egomask_stats["TF_pct"]:.1f}%, FT={egomask_stats["FT_pct"]:.1f}%, FF={egomask_stats["FF_pct"]:.1f}%')
    
    if lisa_val_stats:
        print(f'\n【LISA Val】')
        print(f'  样本数: {lisa_val_stats["num_samples"]}')
        print(f'  GIoU: {lisa_val_stats.get("giou", 0)*100:.2f}%')
        print(f'  CIoU: {lisa_val_stats.get("ciou", 0)*100:.2f}%')
    
    if lisa_test_stats:
        print(f'\n【LISA Test】')
        print(f'  样本数: {lisa_test_stats["num_samples"]}')
        print(f'  GIoU: {lisa_test_stats.get("giou", 0)*100:.2f}%')
        print(f'  CIoU: {lisa_test_stats.get("ciou", 0)*100:.2f}%')
    
    print('='*80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--egomask_result', type=str, required=True)
    parser.add_argument('--lisa_val_result', type=str, default=None)
    parser.add_argument('--lisa_test_result', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='benchmark_vis')
    
    args = parser.parse_args()
    
    # 加载结果
    egomask_stats = load_egomask_results(args.egomask_result)
    
    lisa_val_stats = None
    if args.lisa_val_result and os.path.exists(args.lisa_val_result):
        lisa_val_stats = load_lisa_results(args.lisa_val_result)
    
    lisa_test_stats = None
    if args.lisa_test_result and os.path.exists(args.lisa_test_result):
        lisa_test_stats = load_lisa_results(args.lisa_test_result)
    
    # 可视化
    visualize_benchmarks(egomask_stats, lisa_val_stats, lisa_test_stats, args.output_dir)

