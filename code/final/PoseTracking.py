import glob
import os
import os
import time
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import einops
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy




# 读取视频路径
def get_video_paths(directory):
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_paths = []
    
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(directory, ext)))  # 获取视频路径
        
    # 替换路径中的反斜杠为正斜杠
    video_paths = [video.replace("\\", "/") for video in video_paths]
    
    return video_paths


# 获取视频的基本信息
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
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
    return frame_info

# K-Means 聚类帧
def cluster_frames_kmeans(frame_info, n_clusters=20):
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

# 获取视频的20个关键帧
def get_20_key_frames(video_path):
    # 获取视频的基本信息
    video_info = get_video_info(video_path)
    
    # 获取光流信息
    frame_info = calculate_optical_flow(video_path)
    
    # 进行K-Means聚类
    representative_frames = cluster_frames_kmeans(frame_info, n_clusters=20)
    
    # 从视频中提取关键帧图像
    key_frames = []
    cap = cv2.VideoCapture(video_path)
    
    for frame_info in representative_frames:
        frame_number = frame_info["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            key_frames.append(frame)
    
    cap.release()
    
    return key_frames

def extract_key_frames_from_videos(video_files):
    all_key_frames = []  # 用于存储所有视频的关键帧
    
    for video in video_files:
        key_frames = get_20_key_frames(video)  # 提取当前视频的关键帧
        all_key_frames.extend(key_frames)  # 将当前视频的关键帧合并到总列表
    
    return all_key_frames

# 启用 XLA
tf.config.optimizer.set_jit(True)
TARGET_FRAMES = 20  # 每个视频的关键帧数
HEIGHT, WIDTH = 224, 224  # 每帧的大小
class Conv2Plus1D(layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seq = tf.keras.Sequential([
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        # 确保返回所有参数，包括自定义的
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 通过从配置字典中解构来创建类实例
        return cls(**config)

class ResidualMain(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.seq = tf.keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def add_residual_block(input, filters, kernel_size):

  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input

  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class Project(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seq = tf.keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResizeVideo(layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t=old_shape['t'])
        return videos

    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

input_shape = (None, 20, HEIGHT, WIDTH, 3)  # 输入视频的形状，None 表示批次大小不固定
input = layers.Input(shape=(input_shape[1:]))  # 定义输入层，形状为 (时间步数, 高度, 宽度, 通道数)
x = input

# 初始卷积层：执行 2+1D 卷积操作（空间 + 时间分解）
x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)  # 批量归一化层，规范化每批次的特征
x = layers.ReLU()(x)  # 激活函数 ReLU
x = ResizeVideo(HEIGHT//2, WIDTH//4 )(x)  # 调整视频帧的尺寸到 (HEIGHT/2, WIDTH/2)

# Block 1: 添加第一个残差块并调整尺寸
x = add_residual_block(x, 16, (3, 3, 3))  # 添加残差块，过滤器数为 16，卷积核大小为 3x3x3
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)  # 调整尺寸到 (HEIGHT/4, WIDTH/4)

# Block 2: 添加第二个残差块并调整尺寸
x = add_residual_block(x, 32, (3, 3, 3))  # 过滤器数为 32
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)  # 调整尺寸到 (HEIGHT/8, WIDTH/8)

# Block 3: 添加第三个残差块并调整尺寸
x = add_residual_block(x, 64, (3, 3, 3))  # 过滤器数为 64
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)  # 调整尺寸到 (HEIGHT/16, WIDTH/16)

# Block 4: 添加第四个残差块
x = add_residual_block(x, 128, (3, 3, 3))  # 过滤器数为 128

# 全局平均池化和分类
x = layers.GlobalAveragePooling3D()(x)  # 对时间、空间维度进行全局平均池化，生成特征向量
x = layers.Flatten()(x)  # 展平为 1D 向量
x = layers.Dense(15)(x)  # 全连接层输出 10 个分类

# 定义模型
model = keras.Model(input, x)
# 修改损失函数的 reduction 参数
loss = SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size')
# 使用 Keras 提供的工具绘制模型结构
keras.utils.plot_model(
    model,               # 目标模型
    expand_nested=True,  # 展开嵌套的层，例如子模块或自定义层
    dpi=60,              # 设置图片分辨率
    show_shapes=True     # 显示每一层输出的形状
)



# 加载模型函数
def load_my_model(model_path):
    # 定义自定义层
    custom_objects = {
        'Conv2Plus1D': Conv2Plus1D,
        'ResizeVideo': ResizeVideo,
        'ResidualMain': ResidualMain,
        'Project': Project 
    }
    
    
    model = load_model(model_path, custom_objects=custom_objects)
    
    # 编译模型
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model



def preprocess_frames(frames, target_height, target_width, target_frames):
    """
    从帧数组加载关键帧并预处理为固定大小和数量。
    """
    processed_frames = []
    
    for frame in frames:
        # 调整大小
        image = Image.fromarray(frame)
        image = image.resize((target_width, target_height))  # 调整大小
        processed_frames.append(np.array(image) / 255.0)  # 归一化到 [0, 1]
    
    # 确保帧数量一致（补齐或裁剪）
    if len(processed_frames) > target_frames:
        processed_frames = processed_frames[:target_frames]
    elif len(processed_frames) < target_frames:
        padding = target_frames - len(processed_frames)
        processed_frames.extend([np.zeros((target_height, target_width, 3))] * padding)  # 补零帧
    
    return np.stack(processed_frames, axis=0)

def predict_from_frames(model, frames, target_height, target_width, target_frames):
    """
    用模型预测包含20帧的视频的标签概率。
    """
    # 预处理帧
    input_frames = preprocess_frames(frames, target_height, target_width, target_frames)
    input_frames = np.expand_dims(input_frames, axis=0)  # 添加批次维度
    
    # 预测
    probabilities = model.predict(input_frames)
    return probabilities

def sobel_custom(image_array):
    """
    使用自定义 Sobel 算子计算梯度。
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = np.zeros_like(image_array)
    grad_y = np.zeros_like(image_array)
    
    height, width = image_array.shape

    for i in range(1, height-1):
        for j in range(1, width-1):
            grad_x[i, j] = np.sum(image_array[i-1:i+2, j-1:j+2] * sobel_x)
            grad_y[i, j] = np.sum(image_array[i-1:i+2, j-1:j+2] * sobel_y)
    
    return grad_x, grad_y

def load_and_preprocess_frames_custom(frames, target_height, target_width, target_frames):
    """
    预处理关键帧，添加梯度特征。
    """
    processed_frames = []

    for frame in frames:
        # 转为灰度图
        image = Image.fromarray(frame).convert('L')
        image = image.resize((target_width, target_height))
        image_array = np.array(image, dtype=np.float32) / 255.0

        # 计算梯度
        grad_x, grad_y = sobel_custom(image_array)

        # 归一化梯度
        grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-6)
        grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min() + 1e-6)

        # 三通道特征
        three_channel_frame = np.stack([image_array, grad_x, grad_y], axis=-1)
        processed_frames.append(three_channel_frame)

    # 调整帧数量
    if len(processed_frames) > target_frames:
        processed_frames = processed_frames[:target_frames]
    elif len(processed_frames) < target_frames:
        padding = target_frames - len(processed_frames)
        padding_frame = np.zeros((target_height, target_width, 3))
        processed_frames.extend([padding_frame] * padding)

    return np.stack(processed_frames, axis=0)

def process_file(file_path, model1, model2, model3):
    # 获取文件名
    file_name = os.path.basename(file_path)


    key_frames = get_20_key_frames(file_path)
    processed_frames = load_and_preprocess_frames_custom(key_frames, 224, 224, 20)
    processed_frames = np.expand_dims(processed_frames, axis=0)

    probabilities = model1.predict(processed_frames)
    
    # 获取prediction1 (标签)
    
    prediction1 = np.argmax(model2.predict(probabilities))  # 得到标签
    
    # 获取prediction2 (分数)
    prediction2 = model3.predict(probabilities)[0][0]

    
    # 如果标签是14，设置prediction2为0
    if prediction1 == 14:
        prediction2 = 0
    
    return file_name, prediction1, prediction2



##主程序

model_path = '/home/service/competition/15209917996/model/200grey_model.h5'
model1 = load_my_model(model_path)
model2 = load_model("/home/service/competition/15209917996/model/final_model.h5")
model3=load_model("/home/service/competition/15209917996/model/best_model.h5")


# 全局变量，用来存储所有文件的结果
all_file_results = []
video_directory="/home/service/video"
# 多个文件路径
file_paths = get_video_paths(video_directory)

# 处理每个文件
for file_path in file_paths:
    start_time = time.time()  # 记录处理开始时间
    file_name, prediction1, prediction2 = process_file(file_path, model1, model2, model3)
    end_time = time.time()  # 记录处理结束时间
    
    # 计算时间差（单位为毫秒）
    time_taken_ms = (end_time - start_time) * 1000
    
    # 将结果添加到全局变量中
    all_file_results.append({
        "file_name": file_name,
        "prediction1": prediction1,
        "prediction2": prediction2,
        "time_taken_ms": time_taken_ms
    })

output_directory = "/home/service/result/15209917996/"
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "15209917996_submit.csv")

# 写入CSV文件
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(["视频文件唯一标识", "动作分类标签label", "动作标准度评分", "推理总耗时(ms)"])
    
    # 写入每个文件的结果
    for result in all_file_results:
        writer.writerow([
            result["file_name"],
            result["prediction1"],
            result["prediction2"],
            result["time_taken_ms"]
        ])