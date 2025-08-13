import os
import json
from PIL import Image

# 定义类别名称列表，对应YOLO格式的类别索引
class_names = ["edge"]

# 支持的图片文件扩展名列表
img_types = ['.JPEG', '.jpg']

def json_to_yolo(json_file, img_dir, output_dir):
    """
    将单个JSON标注文件转换为YOLO格式的TXT文件

    参数:
        json_file: JSON标注文件路径
        img_dir: 对应的图片目录路径
        output_dir: YOLO格式输出目录
    """
    # 1. 加载JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)  # 解析JSON内容

    # 2. 获取基础文件名（不带扩展名）
    img_path = data.get('imagePath', None)
    if img_path is None:
        print(f"Error: 'imagePath' key missing in {json_file}")
        return

    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    # 3. 查找对应图片文件（支持.jpg和.JPEG）
    img_file = next(
        (os.path.join(img_dir, base_filename + ext)
         for ext in img_types
         if os.path.isfile(os.path.join(img_dir, base_filename + ext))),
        None
    )

    # 4. 图片文件不存在时报错
    if img_file is None:
        raise FileNotFoundError(f"No image file found for {base_filename} with supported types {img_types}")

    # 5. 获取图片尺寸（用于坐标归一化）
    with Image.open(img_file) as img:
        img_width, img_height = img.size  # 图片的宽高

    # 6. 准备YOLO格式数据容器
    yolo_data = []

    # 7. 检查是否包含'shapes'键
    shapes = data.get('shapes', None)
    if shapes is None:
        print(f"Error: 'shapes' key missing in {json_file}")
        return

    # 8. 处理每个标注对象
    for shape in shapes:
        print(shape)  # 调试打印对象信息

        # 8.1 获取类别ID（从class_names列表中查找索引）
        if shape['label'] not in class_names:
            print(f"Error: Label '{shape['label']}' not in class names list.")
            continue
        class_id = class_names.index(shape['label'])

        # 8.2 计算矩形框的中心点坐标和宽高
        x1, y1 = shape['points'][0]  # 左上角坐标
        x2, y2 = shape['points'][1]  # 右下角坐标

        # 中心点坐标
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height

        # 宽度和高度
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height

        # 8.3 构建YOLO格式行：class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
        yolo_data.append(yolo_line)

    # 9. 写入YOLO格式文件
    yolo_file = os.path.join(output_dir, base_filename + '.txt')
    with open(yolo_file, 'w') as f:
        f.write('\n'.join(yolo_data))  # 每个对象一行


def convert_json_to_yolo(json_dir, img_dir, output_dir):
    """
    批量转换目录下的所有JSON文件为YOLO格式

    参数:
        json_dir: JSON文件所在目录
        img_dir: 对应的图片目录
        output_dir: 输出目录
    """
    # 1. 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 遍历JSON目录
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            try:
                # 3. 转换单个文件
                json_to_yolo(json_path, img_dir, output_dir)
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")


# 执行转换（训练集和测试集）
convert_json_to_yolo(
    json_dir='./dataset/labels',
    img_dir='./dataset/images',
    output_dir='./dataset/json2txt'
)



