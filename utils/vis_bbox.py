import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
def _box_to_xyxy(box4, fmt="xywh"):
    """
    box4: [x,y,w,h] or [x1,y1,x2,y2]
    """
    x1, y1, a, b = box4
    if fmt == "xywh":
        x2, y2 = x1 + a, y1 + b
    elif fmt == "xyxy":
        x2, y2 = a, b
    else:
        raise ValueError("fmt must be 'xywh' or 'xyxy'")
    return x1, y1, x2, y2

def _clip_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    # 防止反向
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _color_from_text(text: str):
    # 稳定的伪随机颜色（不需要 random）
    h = abs(hash(text))
    return (50 + (h % 180), 50 + ((h // 3) % 180), 50 + ((h // 7) % 180))

def draw_bboxes_and_save(
    image_path: str,
    phrases: list,
    out_path: str,
    bbox_format: str = "xywh",   # 你的数据建议用 "xywh"
    draw_text: bool = True,
):
    """
    phrases: 形如：
      [
        {"type":"phrase","text":"xxx","bbox":[[x,y,w,h], [x,y,w,h]], "mask":[], "point":[]},
        ...
      ]
    """
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    # 线宽随图片尺寸自适应
    thickness = max(2, int(min(W, H) * 0.004))

    for item in phrases:
        text = str(item.get("text", ""))
        bboxes = item.get("bbox", []) or []

        color = _color_from_text(text)

        for b in bboxes:
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue

            x1, y1, x2, y2 = _box_to_xyxy(b, fmt=bbox_format)

            # float -> int，并裁剪到图像范围
            x1, y1, x2, y2 = _clip_xyxy(int(round(x1)), int(round(y1)),
                                        int(round(x2)), int(round(y2)), W, H)

            # 画框
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

            # 画文字（标签底色 + 字）
            if draw_text and text:
                label = text
                # 文本背景框
                tw, th = draw.textbbox((0, 0), label)[2:]
                pad = max(2, thickness)
                tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
                tx2, ty2 = x1 + tw + 2 * pad, ty1 + th + 2 * pad

                draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
                draw.text((tx1 + pad, ty1 + pad), label, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    data = json.load(open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k.json'))
    for d in tqdm(data):
        image = os.path.join('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data', d['image'])
        phrases = d['annotation']
        draw_bboxes_and_save(
            image_path=image,
            phrases=phrases,
            out_path=image.split('/')[-1].replace(".jpg", "_with_bbox.jpg"),
            bbox_format="xywh",  # 如果你的 bbox 是 [x1,y1,x2,y2] 就改成 "xyxy"
            draw_text=True
        )

