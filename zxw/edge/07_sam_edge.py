
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# 路径配置
base_dir = "output/unlabeled_detections"
image_dir = os.path.join(base_dir, "images")  # 图片路径
label_dir = os.path.join(base_dir, "labels")  # 标签路径
output_dir = os.path.join(base_dir, "sam")    # 结果保存路径
os.makedirs(output_dir, exist_ok=True)

# SAM初始化
sam_checkpoint = "sam_vit_b_01ec64.pth"  # 权重文件路径
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)


def yolo_to_bbox(norm_box, img_w, img_h):
    x_c, y_c, w, h = norm_box
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    return x1, y1, x2, y2


def get_valid_image_files(image_dir, label_dir):
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")}
    valid_files = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            if os.path.splitext(img_file)[0] in label_files:
                valid_files.append(img_file)
    return valid_files


image_files = get_valid_image_files(image_dir, label_dir)

if not image_files:
    print("[ERROR] No valid images with corresponding labels found")
    exit()

for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        tqdm.write(f"[ERROR] Failed to read image {img_path}")
        continue

    img_h, img_w = img.shape[:2]
    bboxes = []

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 6:
                try:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        norm_box = list(map(float, parts[1:5]))
                        bbox = yolo_to_bbox(norm_box, img_w, img_h)
                        bboxes.append(bbox)
                except:
                    continue

    if not bboxes:
        tqdm.write(f"[INFO] No class 0 bbox in {img_name}, skip.")
        continue

    try:
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for i, bbox in enumerate(bboxes):
            masks, _, _ = predictor.predict(
                box=np.array(bbox),
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8) * 255

            # 彩色mask 红色通道赋值
            mask_color = np.zeros_like(img)
            mask_color[:, :, 2] = mask

            # 拼接原图和mask
            composed = np.concatenate((img, mask_color), axis=1)

            # 保存拼接图
            out_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask_{i}_compose.png")
            cv2.imwrite(out_path, composed)
            tqdm.write(f"[INFO] Saved composed image: {out_path}")

    except Exception as e:
        tqdm.write(f"[ERROR] Failed processing {img_name}: {str(e)}")

