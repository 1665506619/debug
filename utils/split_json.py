import ijson
import json
import os

# ===== 配置 =====
INPUT_JSON = "/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/data/seg_train/new/mapillary.json"        # 11GB 原始文件
OUT_DIR = "/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/data/seg_train/new/"          # 输出目录
NUM_SPLITS = 5                  # 拆成 40 个文件

os.makedirs(OUT_DIR, exist_ok=True)

# 先统计总样本数（一次轻量遍历，不加载进内存）
print("Counting samples...")
total = 0
with open(INPUT_JSON, "rb") as f:
    for _ in ijson.items(f, "item"):
        total += 1

print(f"Total samples: {total}")

per_file = (total + NUM_SPLITS - 1) // NUM_SPLITS
print(f"Samples per file: {per_file}")

# ===== 开始拆分 =====
with open(INPUT_JSON, "rb") as f:
    items = ijson.items(f, "item")

    file_idx = 0
    sample_idx = 0
    out_f = None
    buffer = []

    for obj in items:
        if sample_idx % per_file == 0:
            if out_f:
                json.dump(buffer, out_f)
                out_f.close()
                buffer = []

            out_path = os.path.join(OUT_DIR, f"mapillary_{file_idx:02d}.json")
            print(f"Writing {out_path}")
            out_f = open(out_path, "w", encoding="utf-8")
            file_idx += 1

        buffer.append(obj)
        sample_idx += 1

    if buffer:
        json.dump(buffer, out_f)
        out_f.close()

print("Done.")
