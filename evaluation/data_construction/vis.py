import os
import json
import cv2

# ===== 需要你根据实际情况修改的路径 =====
ANNOTATION_JSON = "/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoObjects/egobjectsv1_train.json"      # 标注文件路径
IMAGE_ROOT = "/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/"                          # 图片根目录（相对/绝对路径都可以）
OUTPUT_DIR = "./vis"           # 输出图片目录
# =========================================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def draw_bbox_on_image(entry, image_root, output_dir):

    img_rel_path = entry["image"]
    img_path = os.path.join(image_root, img_rel_path)

    if not os.path.exists(img_path):
        print(f"[WARN] 图像不存在: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 无法读取图像: {img_path}")
        return

    # 遍历所有标注
    for ann in entry.get("annotation", []):
        label = ann.get("text", "")
        bbox_list = ann.get("bbox", [])

        # 可能有多个 bbox
        for bbox in bbox_list:
            if len(bbox) != 4:
                print(f"[WARN] bbox 长度不是 4: {bbox}")
                continue

            x, y, w, h = bbox  # xyhw
            # 转成左上角(x1,y1)和右下角(x2,y2)
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))

            # 画矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 在框上方写文字（类别）
            if label:
                # 防止文字超出图像上边界
                text_org = (x1, max(y1 - 5, 0))
                cv2.putText(
                    img,
                    label,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA
                )

    # 保存到输出目录，保持相对路径结构
    out_path = os.path.join(output_dir, img_rel_path)
    out_dir = os.path.dirname(out_path)
    ensure_dir(out_dir)

    cv2.imwrite(out_path, img)
    print(f"[OK] 保存带 bbox 的图片: {out_path}")


def main():
    ensure_dir(OUTPUT_DIR)

    # 读取标注 JSON
    with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 如果你的文件是单个对象而不是 list，这里统一转成 list 处理
    if isinstance(data, dict):
        data = [data]

    print(f"共 {len(data)} 条标注，将开始批量处理…")

    for i, entry in enumerate(data):
        print(f"处理 {i + 1}/{len(data)}: {entry.get('image', 'UNKNOWN')}")
        draw_bbox_on_image(entry, IMAGE_ROOT, OUTPUT_DIR)
        if i > 10:
            break

    print("全部处理完成！")


if __name__ == "__main__":
    main()
