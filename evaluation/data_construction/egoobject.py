from collections import defaultdict
import json
def convert(data):
    # 建立 image_id → image_info 的映射
    image_info = {img["id"]: img for img in data["images"]}

    # 最终结果
    result = []

    # 临时结构：image_id → { "image":xxx, "height":h, "width":w, "annotation": { text → [bbox1,bbox2] } }
    temp = {}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]
        text = ann["category_freeform"]

        # 初始化图片条目
        if img_id not in temp:
            img = image_info[img_id]
            temp[img_id] = {
                "image": 'EgoObjects/images/'+img["url"],
                "height": img["height"],
                "width": img["width"],
                "annotation": defaultdict(list)  # 类别名 → bbox 列表
            }

        # 添加 bounding box
        temp[img_id]["annotation"][text].append(bbox)

    # 转换 annotation 为目标格式
    for img_id, info in temp.items():
        annotations_list = []
        for text, bboxes in info["annotation"].items():
            for ann in annotations_list:
                if ann["text"] == text:
                    ann["bbox"].extend(bboxes)
                    break
            else:
                annotations_list.append({
                    "type": "phrase",
                    "text": text,
                    "bbox": bboxes
                })
        result.append({
            "image": info["image"],
            "height": info["height"],
            "width": info["width"],
            "annotation": annotations_list
        })

    return result


data = json.load(open('/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoObjects/EgoObjectsV1_unified_train.json'))
converted = convert(data)

with open('/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoObjects/egobjectsv1_train.json', 'w') as f:
    json.dump(converted, f, indent=2)
