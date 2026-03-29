import os
import json
from collections import defaultdict
from tqdm import tqdm
import random
from pycocotools import mask as maskUtils


def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def convert_dataset(
    anno_json_path,
    mask_root_dir,
    output_json_path,
    bbox_ratio_threshold=0.25,
):
    """
    anno_json_path: 原始 sam_obj365_train_1742k.json 的路径
    mask_root_dir:  mask json 的根目录，例如：
                    /Users/yyq/Downloads/sam_mask_json/sam_obj365_train_1742k/
    output_json_path:  输出文件路径
    bbox_ratio_threshold: bbox 占整张图面积的阈值，大于此值会被过滤掉
    """

    # 1. 读取原始 json
    print(f"Loading annotations from {anno_json_path} ...")
    with open(anno_json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # 2. 建立一些映射，方便查找
    # image_id -> image_info
    imgid_to_img = {img["id"]: img for img in images}

    # category_id -> category_name（作为 phrase 文本）
    catid_to_name = {cat["id"]: cat["name"] for cat in categories}

    # image_id -> list[annotations]
    imgid_to_anns = defaultdict(list)
    for ann in annotations:
        imgid_to_anns[ann["image_id"]].append(ann)

    results = []

    # 3. 遍历每一张图，按 image 聚合
    print("Processing images and masks ...")
    for image_id, anns in tqdm(imgid_to_anns.items()):
        flag = True
        try:
            img_info = imgid_to_img.get(image_id)
            width = img_info["width"]
            height = img_info["height"]
            file_name = 'Object365/train/'+img_info["file_name"]          # e.g. 'patch8/objects365_v1_00420917.jpg'
            seg_file_rel = 'Object365/sam_mask_json/sam_obj365_train_1742k/'+ img_info.get("segmentation_file")  # e.g. 'patch8/objects365_v1_00420917.json'
        except:
            continue

        seg_file_path = os.path.join(mask_root_dir, img_info.get("segmentation_file"))


        # 4. 读取该图片对应的 mask json
        with open(seg_file_path, "r") as f:
            seg_data = json.load(f)

        seg_dict = seg_data.get("segmentations", {})

        # 5. 对该图的所有 annotation 按 category 聚合，同时做 bbox 过滤
        #    cat_id -> {"type":"phrase", "text":cat_name, "mask":[rle, ...]}
        catid_to_group = {}

        img_area = width * height
        area_threshold = bbox_ratio_threshold * img_area

        filtered_cat = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h]
            if len(bbox) != 4:
                continue
            _, _, bw, bh = bbox
            bbox_area = bw * bh

            # 过滤 bbox 占比 > 50% 的标注
            if bbox_area > area_threshold:
                filtered_cat.append(ann["category_id"])
                continue

            ann_id_str = str(ann["id"])
            if ann_id_str not in seg_dict:
                filtered_cat.append(ann["category_id"])
                continue
            rle = seg_dict.get(ann_id_str, None)
            # mask = annToMask(rle, h=height, w=width)

            cat_id = ann["category_id"]
            cat_name = catid_to_name.get(cat_id, str(cat_id))

            if cat_id not in catid_to_group:
                catid_to_group[cat_id] = {
                    "type": "phrase",
                    "text": cat_name,
                    "mask": [],
                }

            catid_to_group[cat_id]["mask"].append(rle)
        if not flag:
            continue
        # 如果这一张图所有 annotation 都被过滤了，就不写入结果
        if not catid_to_group or len(catid_to_group)<=5:
            continue

        catid_to_group = {
            cid: group
            for cid, group in catid_to_group.items()
            if len(group["mask"]) <= 10 and cid not in filtered_cat
        }


        if len(catid_to_group) > 10:
            selected_cat_ids = random.sample(
                list(catid_to_group.keys()),
                10
            )
            catid_to_group = {cid: catid_to_group[cid] for cid in selected_cat_ids}



        # 6. 组装目标格式
        result_item = {
            "image": file_name,   # 你如果想用绝对路径，可在外面再 join 一下
            "height": height,
            "width": width,
            # "segmentation_file": seg_file_rel,
            "annotation": list(catid_to_group.values()),
        }

        results.append(result_item)

    # 7. 保存结果
    print(f"Saving converted dataset to {output_json_path} ...")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Total images in result:", len(results))


if __name__ == "__main__":
    # 根据你的实际路径修改
    anno_json_path = "/Users/yyq/Downloads/sam_obj365_train_1742k.json"
    mask_root_dir = "/Users/yyq/Downloads/sam_mask_json/sam_obj365_train_1742k"
    output_json_path = "/Users/yyq/Downloads/obj365_sam_train.json"

    convert_dataset(
        anno_json_path=anno_json_path,
        mask_root_dir=mask_root_dir,
        output_json_path=output_json_path,
        bbox_ratio_threshold=0.25,
    )
