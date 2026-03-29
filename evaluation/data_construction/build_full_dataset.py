#!/usr/bin/env python3
"""
生成完整的EgoMask训练集数据（不覆盖原有文件）
"""

import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(__file__))

from construct_egomask_diverse import EgoMaskTrainDiverse

if __name__ == '__main__':
    print("=" * 60)
    print("开始生成完整的EgoMask训练集数据")
    print("=" * 60)
    
    # 创建带时间戳的后缀，避免覆盖原有文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_suffix = f'_{timestamp}'
    
    print(f"输出文件后缀: {output_suffix}")
    print("这将生成新文件，不会覆盖原有文件")
    print()
    
    # 创建实例
    builder = EgoMaskTrainDiverse()
    
    # 生成完整数据（处理所有视频）
    builder.build(num_obj=5, max_videos=None, output_suffix=output_suffix)
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - EGOMASK_train_diverse{output_suffix}.json")
    print(f"  - EGOMASK_8f_start{output_suffix}.json")
    print(f"  - EGOMASK_8f_middle{output_suffix}.json")
    print(f"  - EGOMASK_8f_end{output_suffix}.json")
    print(f"  - ... (其他分组文件)")
    print("\n文件保存在: /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/")
    print("=" * 60)

