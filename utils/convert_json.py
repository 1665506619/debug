import json
import os
from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


data_root = "/lustre/fs11/portfolios/llmservice/users/smajumdar/region/data/seg_train"
new_data_root = "/lustre/fs11/portfolios/llmservice/users/smajumdar/region/data/seg_train/new"
cache_dir = "/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/data/cache"

data_list = [
    # "refcoco.json",
    # "refcoco+.json",
    # "refcocog.json",
    # "grefcoco.json",
    # "grefcoco_single.json",
    # "coco_instances.json",
    # "coco_instances_train_with_none5.json",
    # "egobjectsv1_train.json",
    # "refclef.json",
    # "lvis_v1_train.json",
    # "coco_instances_train_with_none5.json",
    # "mapillary.json",
    # "paco_lvis.json",
    # "partimagenet.json",
    # "pascal_part.json",
    # "reason_seg_cat.json",
    # "roboafford_paco.json",
    # "obj365_phrase_sam_474k.json",
    # "obj365_phrase_final_mask_548k.json",
    # "obj365_cat_646k.json",
    # "roborefit.json",
    # "paco_ego4d_v1_train.json",
    # "icdar_2019.json",
    # "icdar_2019_task2.json",
    # "lisa_plus_coco_instances_merge.json"
    # "lisa_plus_cot.json",
    # "lisa_plus_instance.json",
    # "roi_train_convert.json",
    # "humanref_45k_mask.json",
    "sa1b_70k_sam_qa_filtered.json",
    "epic_kitchen_qa_no_none_filtered.json",
    "hd_epic_qa_no_none_filtered.json",
    "ego4d_text_qa_filtered.json",
    # "SA-1B_10k_qa.json",
    # "epic_kitchen_qa_no_none.json",
    # "hd_epic_qa_no_none.json",
    # "ego_ocr_gemnini_generate.json",
    # "vg_train_bbox_grounding.json",
]


def _split_list(data: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(data), n):
        yield data[i : i + n]


def _rewrite_one_record(d: dict, ensure_ascii: bool = False) -> Optional[dict]:
    """
    保持原逻辑：
    - 从 d 中取 annotations 或 annotation，并从 d 里 pop 掉
    - 为空则丢弃该样本
    - mask/bbox -> ann_type + ann(序列化列表) 并 pop 原字段
    - category/answer 缺失则补 None
    - 写回 d["annotation"] = annotation
    """
    annotation = d.pop("annotations", d.pop("annotation", None))
    if not annotation or len(annotation) == 0:
        return None

    for ann in annotation:
        if "mask" in ann:
            ann["ann_type"] = "mask"
            ann["ann"] = (
                [json.dumps(m, ensure_ascii=ensure_ascii) for m in ann["mask"]]
                if ann["mask"] is not None
                else None
            )
            ann.pop("mask", None)

        elif "bbox" in ann:
            ann["ann_type"] = "bbox"
            ann["ann"] = (
                [json.dumps(m, ensure_ascii=ensure_ascii) for m in ann["bbox"]]
                if ann["bbox"] is not None
                else None
            )
            ann.pop("bbox", None)

        ann.setdefault("category", "")
        ann.setdefault("answer", "")

    d["annotation"] = annotation
    return d

def _split_by_file_size(
    data: List[dict],
    max_bytes: int,
    ensure_ascii: bool = False,
):
    """
    按 JSON 序列化后的字节大小拆分
    """
    chunk = []
    chunk_bytes = 2  # for '[' and ']'

    for d in data:
        s = json.dumps(d, ensure_ascii=ensure_ascii)
        b = len(s.encode("utf-8")) + 1  # +1 for comma

        # 单条就超过 max_bytes，也必须单独成块
        if chunk and chunk_bytes + b > max_bytes:
            yield chunk
            chunk = []
            chunk_bytes = 2

        chunk.append(d)
        chunk_bytes += b

    if chunk:
        yield chunk

def rewrite_and_split_json_for_hf(
    src_json_path: str,
    out_dir: str,
    cache_dir: str,
    max_file_size_mb: int = 200,
    out_prefix: Optional[str] = None,
    ensure_ascii: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    max_bytes = max_file_size_mb * 1024 * 1024

    # 1) load 原始 json
    with open(src_json_path, "r") as f:
        data = json.load(f)

    # 2) rewrite
    processed: List[dict] = []
    for d in tqdm(data, desc=f"rewrite {os.path.basename(src_json_path)}"):
        if 'coco_instances' in src_json_path  or 'lvis_v1' in src_json_path or 'mapillary' in src_json_path:
            d['type'] = 'instseg'
        else:
            d['type'] = 'refseg'
        new_d = _rewrite_one_record(d, ensure_ascii=ensure_ascii)
        if new_d is not None:
            processed.append(new_d)

    base = out_prefix or os.path.splitext(os.path.basename(src_json_path))[0]

    chunk_paths: List[str] = []
    sub_datasets = []

    for idx, chunk in enumerate(
        _split_by_file_size(processed, max_bytes, ensure_ascii)
    ):
        chunk_path = os.path.join(out_dir, f"{base}_part{idx}.json")
        print(
            f"Writing chunk {idx} "
            f"({len(chunk)} samples, max {max_file_size_mb}MB) -> {chunk_path}"
        )

        with open(chunk_path, "w") as f:
            json.dump(chunk, f, ensure_ascii=ensure_ascii)

        chunk_paths.append(chunk_path)
        ds = load_dataset(
            "json", data_files=chunk_path, cache_dir=cache_dir
        )["train"]
        sub_datasets.append(ds)

    dataset = (
        concatenate_datasets(sub_datasets)
        if len(sub_datasets) > 1
        else sub_datasets[0]
    )
    return dataset, chunk_paths


def _load_json_dataset(json_path: str, cache_dir: str):
    return load_dataset("json", data_files=json_path, cache_dir=cache_dir)["train"]


def _rewrite_to_single_json_and_load(
    src_json_path: str,
    dst_json_path: str,
    cache_dir: str,
    ensure_ascii: bool = False,
):
    with open(src_json_path, "r") as f:
        data = json.load(f)

    new_data: List[dict] = []
    for d in tqdm(data, desc=f"rewrite {os.path.basename(src_json_path)}"):
        if 'coco_instances' in src_json_path  or 'lvis_v1' in src_json_path or 'mapillary' in src_json_path:
            d['type'] = 'instseg'
        else:
            d['type'] = 'refseg'
        new_d = _rewrite_one_record(d, ensure_ascii=ensure_ascii)
        if new_d is not None:
            new_data.append(new_d)

    os.makedirs(os.path.dirname(dst_json_path), exist_ok=True)
    with open(dst_json_path, "w") as f:
        json.dump(new_data, f, ensure_ascii=ensure_ascii, indent=2)

    return _load_json_dataset(dst_json_path, cache_dir=cache_dir)


data_list_new = []
concatenated = None  # 保持你末尾不断 concatenate 的行为

for filename in data_list:
    print(f"Processing {filename}...")

    src_path = os.path.join(data_root, filename)
    dst_path = os.path.join(new_data_root, filename)

    # 1) 优先直接读已经生成好的 new 文件
    try:
        dataset = _load_json_dataset(dst_path, cache_dir=cache_dir)

    except Exception as e1:
        # 2) fallback：从原始文件重写成单 json，再 load
        try:
            dataset = _rewrite_to_single_json_and_load(
                src_json_path=src_path,
                dst_json_path=dst_path,
                cache_dir=cache_dir,
                ensure_ascii=False,
            )

        except Exception as e2:
            # 3) fallback：再失败就分块
            print(f"Failed to process {filename}. single-file error: {e2}. Will try chunked rewrite...")

            try:
                dataset, _chunk_paths = rewrite_and_split_json_for_hf(
                    src_json_path=src_path,
                    out_dir=new_data_root,
                    cache_dir=cache_dir,
                    max_file_size_mb=500,   # 👈 推荐 100–500MB
                    out_prefix=os.path.splitext(filename)[0],
                    ensure_ascii=False,
                )
            except Exception as e3:
                print(f"Failed to process {filename} in chunked mode: {e3}")
                continue

    data_list_new.append(dataset)
    concatenated = concatenate_datasets(data_list_new)

# 最终结果在 concatenated
