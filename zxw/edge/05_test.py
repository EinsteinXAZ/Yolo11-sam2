# -*- coding: utf-8 -*-
from ultralytics import YOLO
if __name__ == '__main__':
# model = YOLO('./models/yolov8n.pt')
    model = YOLO('./models/train_custom/weights/best.pt')
# 进行评估
    metrics = model.val(data='datasets.yaml', split='test')
# print(metrics)
    model.predict(source='./dataset/test/images', **{'save': True})

