#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 交互 + TXT 区域自动框选
  · K：Savitzky-Golay 最平滑曲线
  · E：原 DP 折线
  · S：SAM2 分割
  · D：按 TXT 顺序切换检测框
  其它按键功能保持不变
"""

import os, sys, cv2, torch, numpy as np
from scipy.signal import savgol_filter

# ────────────────────────── SAM2 新接口 ──────────────────────────
import torch, numpy as np, cv2
from sam2.build_sam        import build_sam2          # 构造模型
from sam2.sam2_image_predictor import SAM2ImagePredictor
# ────────────────────────────────────────────────────────────────


# ======================== 配置区 ========================
IMAGE_PATH  = r"000350.jpg"                     # 输入图；留空则弹窗
CFG_PATH    = r"./checkpoints/sam2.1_hiera_t.yaml"  # ✎ 配置文件 (.yaml)
CKPT_PATH   = r"./checkpoints/sam2.1_hiera_tiny.pt" # ✎ 权重文件 (.pt)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR      = "./sam2_out"

VIEW_MAX_W, VIEW_MAX_H = 1600, 900            # 预览窗口最大宽高

EDGE_GRAD_MIN   = 10
SMOOTH_PENALTY  = 0.6
MAX_STEP        = 6
PRIOR_WEIGHT    = 0.05
POLARITY_WEIGHT = 0.40
RIGHT_BIAS      = 0.05
DRAW_THICKNESS  = 2
# =======================================================

# 防乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def pick_file(title, filetypes):
    """弹窗选文件；无法弹窗则命令行输入"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        return filedialog.askopenfilename(title=title, filetypes=filetypes)
    except Exception:
        return input(f"[提示] 无法弹窗，请输入 {title} 路径：").strip()


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def compute_view(shape, max_w, max_h):
    H, W = shape[:2]
    scale = min(max_w / float(W), max_h / float(H), 1.0)
    return scale, (int(round(W * scale)), int(round(H * scale)))


def draw_overlay_on_orig(base, pts, lbs, box, msk=None):
    vis = base.copy()
    for (x, y), l in zip(pts, lbs):
        cv2.circle(vis, (int(round(x)), int(round(y))),
                   4, (0, 255, 0) if l == 1 else (0, 0, 255), -1)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
    if msk is not None:
        mask_rgb = np.zeros_like(vis); mask_rgb[:, :, 1] = 255
        vis = np.where(msk[..., None] > 0, (vis // 2 + mask_rgb // 2), vis)
    return vis


def smooth_poly(poly, win=15, order=3):
    if len(poly) < 5:
        return poly.astype(np.float32)
    k = min(win | 1, len(poly) // 2 * 2 + 1)     # 保证奇数
    px = savgol_filter(poly[:, 0].astype(float), k, order)
    py = savgol_filter(poly[:, 1].astype(float), k, order)
    return np.vstack([px, py]).T.astype(np.float32)


def extract_edge_polyline_in_box(img_bgr, box,
                                 grad_min=EDGE_GRAD_MIN,
                                 smooth=SMOOTH_PENALTY,
                                 max_step=MAX_STEP,
                                 prior_w=PRIOR_WEIGHT,
                                 pol_w=POLARITY_WEIGHT,
                                 right_bias=RIGHT_BIAS,
                                 waypoints=None,
                                 points=None, labels=None):
    """原 DP 折线算法（未改动）"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = max(1, y1 - y0), max(1, x1 - x0)
    roi = img_bgr[y0:y1, x0:x1]
    if H < 10 or W < 10:
        return []
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, a = lab[:, :, 0], lab[:, :, 1]

    try:  # 若安装了 ximgproc，用导向滤波
        import cv2.ximgproc as xip
        Ls = xip.guidedFilter(L, L, 5, 1e-2)
        as_ = xip.guidedFilter(L, a, 5, 1e-2)
    except Exception:  # 退化为双边
        Ls = cv2.bilateralFilter(L, 9, 40, 40)
        as_ = cv2.bilateralFilter(a, 9, 40, 40)

    gx = cv2.Scharr(as_, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(as_, cv2.CV_32F, 0, 1)
    gm = cv2.magnitude(gx, gy)
    th = cv2.phase(gx, gy, angleInDegrees=False)
    g = (gm - gm.min()) / (gm.max() - gm.min() + 1e-6)
    cost = 1 - g + prior_w * (1 - np.cos(th))

    col0 = int(g.mean(axis=0).argmax())
    xs = np.arange(W, dtype=np.float32)
    cost += prior_w * ((xs - col0) ** 2)[None, :] / (W * W)

    def infer_pol(a_img):
        if points and labels:
            pos, neg = [], []
            for (p, l) in zip(points, labels):
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    px, py = int(p[0] - x0), int(p[1] - y0)
                    clip = a_img[max(0, py - 8):min(H, py + 9),
                                 max(0, px - 8):min(W, px + 9)]
                    (pos if l == 1 else neg).append(float(clip.mean()))
            if pos and neg:
                return 1 if np.mean(neg) - np.mean(pos) >= 0 else -1
        # fallback：左右均值比较
        return 1 if a_img[:, -1].mean() - a_img[:, 0].mean() >= 0 else -1

    desired = infer_pol(as_)
    gx_pos = np.maximum(gx * float(desired), 0)
    gx_pos /= (gx_pos.max() + 1e-6)
    cost += pol_w * (1 - gx_pos)

    can = cv2.Canny(Ls, 40, 80, L2gradient=True)
    dist = cv2.distanceTransform(255 - can, cv2.DIST_L2, 3)
    dist /= (dist.max() + 1e-6)
    cost += 0.25 * dist

    cost -= right_bias * (xs / max(1, W - 1))[None, :]
    cost -= cost.min(); cost /= (cost.max() + 1e-6)
    INF = 1e9

    def dp(mat, x_lock=None):
        Hc, Wc = mat.shape
        dp = np.full((Hc, Wc), INF, np.float32)
        prv = np.full((Hc, Wc), -1, np.int16)
        dp[0] = mat[0]
        if x_lock is not None:
            dp[0] = INF; dp[0, x_lock] = mat[0, x_lock]
        for y in range(1, Hc):
            for x in range(Wc):
                best, bpx = INF, -1
                for dx in range(-max_step, max_step + 1):
                    px = x + dx
                    if 0 <= px < Wc:
                        v = dp[y - 1, px] + mat[y, x] + smooth * abs(dx)
                        if v < best:
                            best, bpx = v, px
                dp[y, x] = best; prv[y, x] = bpx
            if gm[y].max() < grad_min:
                bx = int(dp[y - 1].argmin())
                dp[y] = dp[y - 1, bx] + 0.01; prv[y] = bx
        xe = int(dp[-1].argmin())
        path = [xe]
        for y in range(Hc - 1, 0, -1):
            path.append(int(prv[y, path[-1]]))
        return path[::-1]

    xsf = dp(cost)
    return [(x0 + xsf[i], y0 + i) for i in range(H)]


# 圆 / 尖 模式切换
ROUND_MODE = False
def set_params_for_mode(m):
    global SMOOTH_PENALTY, MAX_STEP, PRIOR_WEIGHT, EDGE_GRAD_MIN
    if m:
        SMOOTH_PENALTY, MAX_STEP, PRIOR_WEIGHT, EDGE_GRAD_MIN = 1.6, 2, 0.04, 8
    else:
        SMOOTH_PENALTY, MAX_STEP, PRIOR_WEIGHT, EDGE_GRAD_MIN = 0.6, 6, 0.05, 10
set_params_for_mode(ROUND_MODE)


def main():
    # ─────────────── 1. 读图 ───────────────
    img_path = IMAGE_PATH.strip() or pick_file("选择输入图像",
                                               [("Images", "*.jpg;*.png;*.bmp")])
    if not img_path or not os.path.isfile(img_path):
        raise FileNotFoundError("图像文件无效")
    img_bgr = cv2.imread(img_path); H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    scale, (vw, vh) = compute_view(img_bgr.shape, VIEW_MAX_W, VIEW_MAX_H)

    # ─────────────── 2. 初始化 SAM2 ───────────────
    cfg = CFG_PATH.strip()  or pick_file("选择 SAM2 配置(.yaml)", [("*.yaml", "*.yaml")])
    ckpt = CKPT_PATH.strip() or pick_file("选择 SAM2 权重(.pt)",   [("*.pt",   "*.pt")])

    model = build_sam2(cfg, ckpt).to(DEVICE).eval()
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(img_rgb)           # 内部归一化 + 自动缩放

    # ─────────────── 3. 读取 YOLOv11 输出的 .txt ───────────────
    txt_path = os.path.splitext(img_path)[0] + ".txt"
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"未找到同名 .txt: {txt_path}")

    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ps = ln.strip().split()
            if len(ps) < 5:
                continue
            _, cx, cy, bw, bh = map(float, ps[:5])
            x0 = int((cx - bw / 2) * W); y0 = int((cy - bh / 2) * H)
            x1 = int((cx + bw / 2) * W); y1 = int((cy + bh / 2) * H)
            boxes.append([max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)])
    if not boxes:
        print("⚠️ TXT 中未解析到任何区域，退出。"); return
    print(f"载入 {len(boxes)} 个区域，按 D 切换")

    # ─────────────── 4. 交互循环 ───────────────
    points, labels, box, last_mask = [], [], None, None
    det_idx = 0
    cv2.namedWindow("SAM2 Preview")

    while True:
        vis = draw_overlay_on_orig(img_bgr, points, labels, box, last_mask)
        disp = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA) if scale < 1 else vis
        cv2.imshow("SAM2 Preview", disp)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q'), ord('Q')):
            break
        elif k in (ord('c'), ord('C')):         # 清除点
            points, labels = [], []
        elif k in (ord('r'), ord('R')):         # 全部重置
            points, labels, box, last_mask = [], [], None, None
        elif k in (ord('j'), ord('J')):         # 圆 / 尖 模式
            global ROUND_MODE
            ROUND_MODE = not ROUND_MODE
            set_params_for_mode(ROUND_MODE)
            print("模式：", "圆角优先" if ROUND_MODE else "尖角优先")

        # D: 切换 TXT 框
        elif k in (ord('d'), ord('D')):
            box = boxes[det_idx]
            det_idx = (det_idx + 1) % len(boxes)
            print(f"当前区域[{det_idx}]: {box}")

        # 鼠标单击添加正 / 负样本点（左键=前景，右键=背景）
        elif k == 1:  # cv2.EVENT_LBUTTONDOWN == 1
            x, y = map(int, (cv2.getWindowImageRect("SAM2 Preview")[:2]))
            sx, sy = int((cv2.getMousePosition()[0] - x) / scale), int((cv2.getMousePosition()[1] - y) / scale)
            points.append([sx, sy]); labels.append(1)
        elif k == 2:  # cv2.EVENT_RBUTTONDOWN == 2
            x, y = map(int, (cv2.getWindowImageRect("SAM2 Preview")[:2]))
            sx, sy = int((cv2.getMousePosition()[0] - x) / scale), int((cv2.getMousePosition()[1] - y) / scale)
            points.append([sx, sy]); labels.append(0)

        # S：SAM2 分割
        elif k in (ord('s'), ord('S')):
            if box is None and not points:
                print("⚠️ 先按 D 选区域 或 手动标注"); continue
            pc = torch.tensor(points, dtype=torch.float32, device=DEVICE) if points else None
            pl = torch.tensor(labels, dtype=torch.int32,   device=DEVICE) if labels else None
            bx = torch.tensor([box], dtype=torch.float32,  device=DEVICE)  if box is not None else None

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords     = pc,
                    point_labels     = pl,
                    box              = bx,
                    multimask_output = True
                )
            idx = int(scores.argmax())
            last_mask = (masks[idx].astype(np.uint8) * 255)
            cv2.imshow("mask",
                       cv2.resize(last_mask, (vw, vh),
                                  interpolation=cv2.INTER_NEAREST) if scale < 1 else last_mask)

        # K：平滑曲线
        elif k in (ord('k'), ord('K')):
            if last_mask is None:
                print("⚠️ 先按 S 分割"); continue
            cnts, _ = cv2.findContours(last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                print("⚠️ 未找到轮廓"); continue
            cnt = max(cnts, key=lambda c: cv2.arcLength(c, True)).squeeze(1)
            curve = smooth_poly(cnt, win=17, order=3)
            vis2 = img_bgr.copy()
            cv2.polylines(vis2, [curve.astype(int)], True, (0, 255, 255), 2)
            ensure_dir(OUTDIR)
            cv2.imwrite(os.path.join(OUTDIR, "edge_curve_overlay.png"), vis2)
            np.savetxt(os.path.join(OUTDIR, "edge_curve.csv"), curve,
                       fmt="%.2f", delimiter=",", header="x,y", comments="")
            cv2.imshow("edge_curve",
                       cv2.resize(vis2, (vw, vh),
                                  interpolation=cv2.INTER_AREA) if scale < 1 else vis2)
            print("✅ 已保存 edge_curve_overlay.png & edge_curve.csv")

        # E：原 DP 折线
        elif k in (ord('e'), ord('E')):
            if box is None:
                print("⚠️ 先按 D 选区域"); continue
            anchors = [(int(px), int(py)) for (px, py), l in zip(points, labels)
                       if l == 1 and box[0] <= px <= box[2] and box[1] <= py <= box[3]]
            poly = extract_edge_polyline_in_box(
                img_bgr, box,
                EDGE_GRAD_MIN, SMOOTH_PENALTY, MAX_STEP,
                PRIOR_WEIGHT, POLARITY_WEIGHT, RIGHT_BIAS,
                anchors, points, labels
            )
            if not poly:
                print("⚠️ 未找到折线"); continue
            msk = np.zeros((H, W), np.uint8)
            cv2.polylines(msk, [np.array(poly, np.int32)],
                          False, 255, DRAW_THICKNESS)
            vis_l = img_bgr.copy()
            cv2.polylines(vis_l, [np.array(poly, np.int32)],
                          False, (0, 255, 255), DRAW_THICKNESS)
            ensure_dir(OUTDIR)
            np.savetxt(os.path.join(OUTDIR, "edge_polyline.csv"),
                       np.array(poly), fmt="%d", delimiter=",",
                       header="x,y", comments="")
            cv2.imwrite(os.path.join(OUTDIR, "edge_only_mask.png"), msk)
            cv2.imwrite(os.path.join(OUTDIR, "edge_only_overlay.png"), vis_l)
            cv2.imshow("edge_only",
                       cv2.resize(vis_l, (vw, vh),
                                  interpolation=cv2.INTER_AREA) if scale < 1 else vis_l)
            print("✅ 已保存 edge_polyline.csv & edge_only_*")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
