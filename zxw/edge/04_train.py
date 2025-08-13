

from ultralytics import YOLO

import torch
torch.cuda.empty_cache()  # 清理 GPU 缓存

if __name__ == '__main__':
    model = YOLO('./yolo11n.pt')  # 确保是 YOLOv11 的权重
    results = model.train(
        data='./datasets.yaml',
        imgsz=640,
        epochs=40,
        batch=4,
        # name='yolov11n_custom',
        device='0', # 使用 GPU（如果是 'cpu' 则用 CPU）
        workers = 2 , # 减少数据加载线程
        project = './models',  # 保存根路径
        name = 'train_custom',  # 保存子文件夹
        exist_ok = True
    )
