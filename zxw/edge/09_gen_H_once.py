"""
运行一次，生成 H.yaml
------------------------------------
需求:
  camera_params.yaml  相机内参 + 畸变
  full_board.bmp      一张完整棋盘照片
输出:
  H.yaml              单应性矩阵 (像素 ← 世界)  保存 9 个数
"""

import cv2, yaml, numpy as np

# ------------- 文件路径 -------------
YAML_IN  = "camera_params.yaml"
IMG_FULL = "full_board.bmp"
YAML_OUT = "H.yaml"

# ---------- 读取内参 ----------
with open(YAML_IN, "r") as f:
    yml = yaml.safe_load(f)
K  = np.asarray(yml["camera_matrix"]["data"], dtype=np.float64)
D  = np.asarray(yml["distortion_coefficients"]["data"], dtype=np.float64)
cols, rows = yml["board_shape"]          # (8,11)
sq   = float(yml["square_size"])         # 3.0 mm

# ---------- 世界坐标 (右下尖角系) ----------
MARGIN_X, MARGIN_Y = 11.5, 7.0           # mm
objp = np.zeros((rows*cols,3), np.float32)
for r in range(rows):
    for c in range(cols):
        idx = r*cols + c
        X = MARGIN_X + (rows-1 - r)*sq   # +X 向上
        Y = MARGIN_Y + (cols-1 - c)*sq   # +Y 向左
        objp[idx,:2] = [X, Y]

# ---------- 棋盘角点检测 ----------
img = cv2.imread(IMG_FULL)
if img is None:
    raise FileNotFoundError(IMG_FULL)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ok, corners = cv2.findChessboardCorners(gray, (cols,rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
if not ok:
    raise RuntimeError("完整棋盘角点检测失败，请检查 board_shape / 图像质量")

corners = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4))

# ---------- solvePnP → R,t ----------
_, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
R, _  = cv2.Rodrigues(rvec)
H     = K @ np.hstack([R[:,:2], tvec])   # 像素 ← 世界
H    /= H[2,2]

# ---------- 保存 H ----------
yaml.safe_dump({"H": H.tolist()}, open(YAML_OUT,"w"))
print("✅  H 已保存到", YAML_OUT)
