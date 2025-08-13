"""
不断地把像素(u,v) 映射到 (X,Y) mm  —— 相机/棋盘保持固定
"""

import cv2, yaml, numpy as np

# ------------ 文件 ------------
YAML_KD = "camera_params.yaml"
YAML_H  = "H.yaml"

# ------------ 读取参数 ------------
yml_kd = yaml.safe_load(open(YAML_KD))
K = np.asarray(yml_kd["camera_matrix"]["data"], dtype=np.float64)
D = np.asarray(yml_kd["distortion_coefficients"]["data"], dtype=np.float64)

H = np.asarray(yaml.safe_load(open(YAML_H))["H"])
Hinv = np.linalg.inv(H)

# ------------ 像素 → 世界函数 ------------
def pixel_to_world(u, v):
    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, D, P=K)[0,0]   # 理想像素
    w   = Hinv @ np.array([und[0], und[1], 1.0])
    w  /= w[2]
    return float(w[0]), float(w[1])   # 单位 mm

# ------------ 示例 ------------
if __name__ == "__main__":
    u, v = 1000, 600
    X, Y = pixel_to_world(u, v)
    print(f"(u,v)=({u},{v})  -->  (X,Y)=({X:.2f},{Y:.2f}) mm")
