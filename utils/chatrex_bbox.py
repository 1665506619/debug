import json
from PIL import Image
from rex_omni import RexOmniWrapper
from tqdm import tqdm
from rex_omni import RexOmniVisualize

def ensure_list_bbox(pred_list):
    """
    pred_list: [{"type":"box","coords":[x0,y0,x1,y1]}, ...]
    return: [[x0,y0,x1,y1], ...]
    """
    bboxes = []
    for p in pred_list or []:
        if isinstance(p, dict) and p.get("type") == "box" and "coords" in p:
            coords = p["coords"]
            if isinstance(coords, (list, tuple)) and len(coords) == 4:
                bboxes.append([float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])])
    return bboxes

def add_bbox_to_json(
    json_path: str,
    output_json_path: str,
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    image_root_fallback: str | None = None,  # 可选：当 image_path 不可用时，用 image_root_fallback + image 拼路径
):
    # 1) load json
    with open(json_path, "r", encoding="utf-8") as f:
        data_all = json.load(f)

    # 2) init model
    model = RexOmniWrapper(model_path=model_path, backend=backend)

    for data in tqdm(data_all):
        # 3) load image
        image_path = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/SA-1B/images/'+ data.get("image_path").split('/')[-1]
        if (not image_path) and image_root_fallback:
            image_name = data.get("image")
            if image_name:
                image_path = f"{image_root_fallback.rstrip('/')}/{image_name}"

        if not image_path:
            raise ValueError("找不到 image_path（也没有提供 image_root_fallback + image）。")
        data.pop('image_path')
        data['image'] = 'SA-1B/images/'+ data.get("image").split('/')[-1]
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"⚠️ Warning: 无法打开图片 {image_path}，跳过该图片。")
            continue

        # 4) loop annotations
        ann_list = data.get("annotation", [])
        for ann in ann_list:
            prompt = ann.get("prompt", "")
            if not prompt:
                ann["bbox"] = []
                continue

            # 注意：categories 是 list[str]
            results = model.inference(
                images=image,
                task="detection",
                categories=[prompt],
            )

            # results 是 list，每张图一个 result
            result = results[0] if results else {}

            vis = RexOmniVisualize(
                image=image,
                predictions=result["extracted_predictions"],
                font_size=20,
                draw_width=5,
                show_labels=True,
            )
            vis.save('vis/'+image_path.split('/')[-1].split('.')[0] + f'_vis_{prompt[:10].replace(" ","_")}.jpg')


            extracted = result.get("extracted_predictions", {}) or {}

            # extracted 的 key 一般会等于你的输入 category/prompt
            pred_list = extracted.get(prompt, None)

            # 有些实现可能会轻微改写 key（比如去空格、截断等）。
            # 兜底：如果 exact key 找不到且 extracted 只有一个 key，就取那个 key 的结果。
            if pred_list is None:
                if isinstance(extracted, dict) and len(extracted) == 1:
                    pred_list = next(iter(extracted.values()))
                else:
                    pred_list = []

            ann["bbox"] = ensure_list_bbox(pred_list)

    # 5) write output
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data_all, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to: {output_json_path}")

if __name__ == "__main__":
    add_bbox_to_json(
        json_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/sam_10k_merged.json",
        output_json_path="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/sam_10k_merged_with_bbox.json",
        model_path="IDEA-Research/Rex-Omni",
        backend="transformers",
        # image_root_fallback="/path/to/images"  # 可选
    )
