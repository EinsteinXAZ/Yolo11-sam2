import cv2
import os
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 配置参数
video_dir = './video'  # 视频文件夹路径
start_second = 5  # 从第5秒开始
sample_interval = 8  # 每8帧取1帧

# 创建保存目录（当前目录下的 dataset 文件夹）
os.makedirs('./dataset/images', exist_ok=True)
os.makedirs('./dataset/labels', exist_ok=True)

# 获取视频文件夹下的所有视频文件
video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.avi', '.mp4', '.mov'))]

# 用于命名保存图片的顺序号
image_counter = 1

for video_idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        continue

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_second * fps)

    # 跳转到指定时间（如第8秒）
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_count = 0

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames - start_frame, desc=f"处理视频 {video_idx + 1}", ncols=100) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= total_frames:
                break

            if frame_count % sample_interval == 0:
                # 按顺序号保存图片
                img_path = f'./dataset/images/{image_counter:06d}.jpg'
                cv2.imwrite(img_path, frame)
                image_counter += 1  # 增加顺序号
                saved_count += 1

            frame_count += 1
            pbar.update(1)  # 更新进度条

    cap.release()
    print(f'视频{video_idx + 1} [{video_path}] 提取完成，共保存 {saved_count} 张图片')

print('所有视频处理完毕')
