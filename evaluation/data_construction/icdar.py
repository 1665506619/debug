import json
import os
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

def to_jsonable(obj):
    import numpy as np

    if isinstance(obj, bytes):
        # rle counts 一般是 ascii
        try:
            return obj.decode("ascii")
        except UnicodeDecodeError:
            return obj.decode("utf-8", errors="ignore")

    # numpy 标量
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 容器类型递归处理
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # 其他类型原样返回（str, int, float, bool, None）
    return obj

def polygon_to_rle(points, height, width):
    """
    points: [[x1,y1], [x2,y2], ...]
    return COCO RLE
    """
    poly = [coord for x, y in points for coord in (x, y)]

    # COCO 要求 polygon 是 list[list]
    rles = maskUtils.frPyObjects([poly], height, width)
    rle = maskUtils.merge(rles)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")

    return rle


def convert_icdar_format(
    anno_path,
    image_root,
    save_path,
    image_ext=".jpg",
    skip_illegible=True,
):
    # 读取原始 ICDAR json
    with open(anno_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    output = []

    # raw 的结构类似：{"gt_1726": [ {...}, {...} ], "gt_1727": [...]}
    for img_id, annos in tqdm(raw.items()):
        # 根据你的文件名规则自己改
        image_name = img_id + image_ext
        image_path = os.path.join(image_root, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: image not found: {image_path}")
            continue

        # 读图片拿到宽高
        img = Image.open(image_path)
        width, height = img.size

        new_item = {
            "image": image_root.split('/')[-1] + '/' + image_name,
            "height": height,
            "width": width,
            "annotation": [],
        }

        for a in annos:

            text = a.get("transcription", "")
            points = a["points"]

            # poly -> RLE
            try:
                rle = polygon_to_rle(points, height, width)
            except:
                print(points)
                continue

            new_item["annotation"].append(
                {
                    "type": "OCR",
                    "text": text,     # 这里已经是正常中文字符串
                    "mask": [rle],    # 按你给的格式，放在 list 里
                }
            )
        if len(new_item["annotation"]) > 0:
            output.append(new_item)

    print(f"Converted {len(output)} images from ICDAR format.")
    
    output_clean = to_jsonable(output)

    # 写出 json，关闭 ascii 转义，确保是中文 UTF-8
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_clean, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    convert_icdar_format(
        anno_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/train_labels.json",
        image_root="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/train_images",
        save_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/icdar_2019.json",
        image_ext=".jpg",  # 如果是 .png 就改成 ".png"
    )

    convert_icdar_format(
        anno_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/train_task2_labels.json",
        image_root="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/train_task2_images",
        save_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ICDAR2019/icdar_2019_task2.json",
        image_ext=".jpg",  # 如果是 .png 就改成 ".png"
    )
