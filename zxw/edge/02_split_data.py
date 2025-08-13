import os
import shutil
import random

# 获取所有有标注的图片文件名列表
image_dir = "./dataset/images"
label_dir = "./dataset/json2txt"
image_names = []

# 获取有标签的图片文件
for img_file in os.listdir(image_dir):
    img_name, ext = os.path.splitext(img_file)
    if ext.lower() in ['.jpeg', '.jpg']:  # 确保是 .jpg 或 .jpeg 图片文件
        label_file = os.path.join(label_dir, img_name + '.txt')
        if os.path.isfile(label_file):  # 只有存在标签文件的图片才加入列表
            image_names.append(img_name)

# 打乱文件列表，确保数据集划分是随机的
random.shuffle(image_names)

# 训练集、测试集、验证集的比例
train_size = int(0.7 * len(image_names))  # 70% 用于训练
test_size = int(0.2 * len(image_names))  # 20% 用于测试
val_size = len(image_names) - train_size - test_size  # 剩余的 10% 用于验证

# 创建数据集文件夹（如果不存在的话）
os.makedirs('./dataset/train/labels', exist_ok=True)
os.makedirs('./dataset/train/images', exist_ok=True)
os.makedirs('./dataset/test/labels', exist_ok=True)
os.makedirs('./dataset/test/images', exist_ok=True)
os.makedirs('./dataset/val/labels', exist_ok=True)
os.makedirs('./dataset/val/images', exist_ok=True)


def prepare_data(main_txt_file, main_img_file, train_size, test_size, val_size):
    # 处理训练集
    for i in range(train_size):
        img_name = image_names[i]
        source_txt = os.path.join(main_txt_file, img_name + ".txt")
        source_img = os.path.join(main_img_file, img_name + ".jpg")  # 修改为 .jpg

        # 检查文件是否存在
        if not os.path.exists(source_txt):
            print(f"Warning: Label file for {img_name} not found, skipping...")
            continue
        if not os.path.exists(source_img):
            print(f"Warning: Image file for {img_name} not found, skipping...")
            continue

        train_destination_txt = os.path.join("./dataset/train/labels", img_name + ".txt")
        train_destination_jpg = os.path.join("./dataset/train/images", img_name + ".jpg")  # 修改为 .jpg
        shutil.copy(source_txt, train_destination_txt)
        shutil.copy(source_img, train_destination_jpg)

    # 处理测试集
    for i in range(train_size, train_size + test_size):
        img_name = image_names[i]
        source_txt = os.path.join(main_txt_file, img_name + ".txt")
        source_img = os.path.join(main_img_file, img_name + ".jpg")  # 修改为 .jpg

        # 检查文件是否存在
        if not os.path.exists(source_txt):
            print(f"Warning: Label file for {img_name} not found, skipping...")
            continue
        if not os.path.exists(source_img):
            print(f"Warning: Image file for {img_name} not found, skipping...")
            continue

        test_destination_txt = os.path.join("./dataset/test/labels", img_name + ".txt")
        test_destination_jpg = os.path.join("./dataset/test/images", img_name + ".jpg")  # 修改为 .jpg
        shutil.copy(source_txt, test_destination_txt)
        shutil.copy(source_img, test_destination_jpg)

    # 处理验证集
    for i in range(train_size + test_size, train_size + test_size + val_size):
        img_name = image_names[i]
        source_txt = os.path.join(main_txt_file, img_name + ".txt")
        source_img = os.path.join(main_img_file, img_name + ".jpg")  # 修改为 .jpg

        # 检查文件是否存在
        if not os.path.exists(source_txt):
            print(f"Warning: Label file for {img_name} not found, skipping...")
            continue
        if not os.path.exists(source_img):
            print(f"Warning: Image file for {img_name} not found, skipping...")
            continue

        val_destination_txt = os.path.join("./dataset/val/labels", img_name + ".txt")
        val_destination_jpg = os.path.join("./dataset/val/images", img_name + ".jpg")  # 修改为 .jpg
        shutil.copy(source_txt, val_destination_txt)
        shutil.copy(source_img, val_destination_jpg)


# 执行数据集准备
prepare_data(label_dir, image_dir, train_size, test_size, val_size)


