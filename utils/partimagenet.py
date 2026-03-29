import json
import os

input_json = "/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/data/seg_train/partimagenet.json"
output_json = "/lustre/fsw/portfolios/dir/projects/dir_cosmos_misc/users/mins/data/region/data/seg_train/partimagenet.json"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    path = item["image"]
    dir_path, filename = os.path.split(path)

    # n04252225_8354.JPEG -> n04252225
    class_name = filename.split("_")[0]

    # 新路径
    new_path = os.path.join(dir_path, class_name, filename)
    item["image"] = new_path

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Done!")
