# -*- coding: utf-8 -*-
import os
import random
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

def get_unlabeled_images(image_dir, label_dir):
    """获取没有对应标注文件的图片路径列表"""
    unlabeled_images = []
    for img_file in tqdm(os.listdir(image_dir), desc="扫描图片"):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(label_dir, f"{img_name}.txt")
            if not os.path.exists(txt_path):
                unlabeled_images.append(os.path.join(image_dir, img_file))
    return unlabeled_images

def reduce_by_one_third(images):
    """随机减少约1/3的图片数量（向下取整后保留2/3）"""
    n = len(images)
    if n < 3:
        return images
    keep = n - n // 3
    return random.sample(images, keep)

def main():
    # —— 路径配置 ——
    base        = os.path.dirname(__file__)
    dataset_dir = os.path.join(base, "dataset")
    image_dir   = os.path.join(dataset_dir, "images")
    label_dir   = os.path.join(dataset_dir, "json2txt")
    weights     = os.path.join(base, "models", "train_custom", "weights", "last.pt")

    # —— 加载模型 ——
    print("加载模型中…")
    model = YOLO(weights)

    # —— 查找未标注图片 ——
    print("正在查找未标注的图片…")
    unlabeled = get_unlabeled_images(image_dir, label_dir)
    if not unlabeled:
        print("没有找到未标注的图片，退出。")
        return

    # —— 随机减少1/3 ——
    reduced = reduce_by_one_third(unlabeled)
    print(f"原始未标注: {len(unlabeled)} 张，实际处理: {len(reduced)} 张")

    # —— 单张预测 & 进度条 ——
    print("开始预测并保存结果…")
    for img_path in tqdm(reduced, desc="预测进度"):
        model.predict(
            source=img_path,
            imgsz=640,             # 保持原分辨率
            conf=0.5,
            save=True,
            save_txt=True,
            save_conf=True,
            project="output",
            name="unlabeled_detections",
            exist_ok=True,
            show=False
        )
        # 清理显存和窗口
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()

    print("预测完成！结果保存在 output/unlabeled_detections 文件夹中。")

if __name__ == "__main__":
    main()
