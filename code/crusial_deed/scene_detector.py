import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

# 获取视频的基本信息
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}，跳过该视频。")
        return None
    
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return info

# 计算帧间光流
def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    frame_info = []
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}，跳过该视频。")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))  # 减小分辨率加速处理

        if last_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                last_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            frame_info.append({
                "frame_number": frame_number,
                "optical_flow_mag": avg_magnitude
            })
        last_frame = gray

    cap.release()
    
    if len(frame_info) == 0:
        print(f"视频 {video_path} 没有有效帧，跳过该视频。")
        return None
    
    return frame_info

# K-Means 聚类帧
def cluster_frames_kmeans(frame_info, n_clusters=10):
    if len(frame_info) == 0:
        return None
    
    frame_numbers = np.array([f["frame_number"] for f in frame_info])
    optical_flows = np.array([f["optical_flow_mag"] for f in frame_info])

    # 特征矩阵：时间（帧号）和光流强度
    features = np.column_stack((frame_numbers, optical_flows))

    # 进行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)

    # 将帧按聚类标签分组
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(frame_info[idx])

    # 为每类选择中心点帧
    cluster_centers = kmeans.cluster_centers_
    representative_frames = []
    for cluster_id, center in enumerate(cluster_centers):
        closest_frame = min(
            clusters[cluster_id],
            key=lambda x: np.linalg.norm([x["frame_number"], x["optical_flow_mag"]] - center)
        )
        representative_frames.append(closest_frame)

    # 按时间顺序排序
    representative_frames.sort(key=lambda x: x["frame_number"])
    return representative_frames

# 保存帧为图片
def save_frames(video_path, output_dir, frames):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_number"])
        ret, img = cap.read()
        if ret:
            filename = f"frame_{frame['frame_number']:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), img)
            print(f"保存帧: {filename}")

    cap.release()

# 主处理流程
def process_video(video_path, output_dir=None, n_clusters=10, return_frames=False):
    print("获取视频信息...")
    video_info = get_video_info(video_path)
    if video_info is None:
        return []  # 跳过该视频，继续下一个

    print("计算帧间光流...")
    frame_info = calculate_optical_flow(video_path)
    if frame_info is None:
        return []  # 跳过该视频，继续下一个

    print(f"对帧进行 {n_clusters} 聚类...")
    representative_frames = cluster_frames_kmeans(frame_info, n_clusters)
    if representative_frames is None:
        print(f"视频 {video_path} 聚类失败，跳过该视频。")
        return []  # 跳过该视频，继续下一个

    if return_frames:
        print("处理完成，返回关键帧信息。")
        return representative_frames
    else:
        if output_dir is None:
            raise ValueError("保存模式下，output_dir 不能为空！")
        print("保存关键帧...")
        save_frames(video_path, output_dir, representative_frames)
        print("处理完成！")
