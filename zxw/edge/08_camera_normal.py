

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zhang Zhengyou camera calibration with a chessboard.
-------------------------------------------
• 需要: pip install opencv-python numpy pyyaml
• 使用: python calibrate.py
"""

import glob
import yaml
import cv2
import numpy as np
from pathlib import Path

# ========= 1. 用户可修改的参数 =========
IMG_DIR      = "./chessboards"          # 棋盘格照片所在目录
IMG_PATTERN  = "*.bmp"                  # 文件匹配规则
CHKBD_SHAPE  = (8, 11)                  # 棋盘内部角点数 (列, 行) —— 9×6 角点 => 10×7 格子
SQUARE_SIZE  = 3.0                    # 单个方格实际边长，单位可自行决定 (mm / cm)
SAVE_YAML    = True                    # 是否把参数保存到 camera_params.yaml

# ========= 2. 准备世界坐标系中的角点 =========
objp = np.zeros((CHKBD_SHAPE[0] * CHKBD_SHAPE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHKBD_SHAPE[0], 0:CHKBD_SHAPE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # 3D 点 (世界坐标)
imgpoints = []   # 2D 点 (像素坐标)
image_size = None

# ========= 3. 依次读取并检测角点 =========
paths = sorted(glob.glob(str(Path(IMG_DIR) / IMG_PATTERN)))
if len(paths) == 0:
    raise RuntimeError(f"在 {IMG_DIR} 中找不到匹配 {IMG_PATTERN} 的图片")

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[警告] 读取失败: {path}")
        continue

    if image_size is None:
        image_size = (img.shape[1], img.shape[0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, CHKBD_SHAPE,
                                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ok:
        # 亚像素精细化
        corners_refined = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        )
        imgpoints.append(corners_refined)
        objpoints.append(objp)
        print(f"[✓] 棋盘检测成功: {path}")
    else:
        print(f"[×] 检测失败: {path}")

# ========= 4. 相机标定 =========
if len(objpoints) < 5:
    raise RuntimeError("有效棋盘图像少于 5 张，无法进行稳定标定")

ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None,
    flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3  # 常用配置，可按需调整
)

print("\n========= 标定结果 =========")
print(f"RMS 重投影误差: {ret:.4f} pixels\n")
print("相机矩阵 K:")
print(K)
print("\n畸变系数 [k1 k2 p1 p2 k3 k4 k5 k6]:")
print(distCoeffs.ravel())

# ========= 5. 逐张图像的外参 =========
for idx, (r, t) in enumerate(zip(rvecs, tvecs)):
    R, _ = cv2.Rodrigues(r)       # 转换为 3×3 旋转矩阵
    print(f"\n--- 图像 {idx}: {Path(paths[idx]).name} ---")
    print("旋转向量 rvec:")
    print(r.ravel())
    print("平移向量 tvec (mm 同 SQUARE_SIZE 单位):")
    print(t.ravel())

# ========= 6. 可选：保存到 YAML =========
if SAVE_YAML:
    data = dict(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=dict(data=K.tolist()),
        distortion_coefficients=dict(data=distCoeffs.tolist()),
        rms_error=float(ret),
        square_size=SQUARE_SIZE,
        board_shape=CHKBD_SHAPE,
    )
    with open("camera_params.yaml", "w") as f:
        yaml.safe_dump(data, f)
    print("\n已保存到 camera_params.yaml")

# ========= 7. 评估与可视化（选做） =========
# 计算每张图像重投影误差
total_err = 0
for i, (objp_i, imgp_i, rvec, tvec) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
    imgpoints2, _ = cv2.projectPoints(objp_i, rvec, tvec, K, distCoeffs)
    err = cv2.norm(imgp_i, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_err += err
    print(f"图 {i} 平均重投影误差: {err:.4f} px")
print(f"\n整体平均重投影误差: {total_err / len(objpoints):.4f} px")
