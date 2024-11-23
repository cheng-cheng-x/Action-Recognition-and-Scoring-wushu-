save_path = "C:/data/result/saved_dataset"
def load_dataset(save_path):
    """
    从指定路径加载 TensorFlow 数据集。
    
    Args:
        save_path: 数据集保存路径。
    Returns:
        加载的 tf.data.Dataset 对象。
    """
    dataset = tf.data.experimental.load(save_path)
    print(f"数据集已从 {save_path} 加载")
    return dataset

# 加载数据集
dataset = load_dataset(save_path)

# 假设 dataset 是已经处理好的 tf.data.Dataset 对象
# 将 dataset 分为训练集和验证集
train_split = 0.90  # 80% 数据用于训练
val_split = 1 - train_split

# 获取总样本数
total_samples = len(list(dataset))
train_size = int(total_samples * train_split)

# 分割数据集
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# 配置训练和验证集
train_ds = train_ds.shuffle(buffer_size=1000).batch(16).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)

del dataset
import gc
gc.collect()  # 强制进行垃圾回收

import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras import layers


# 定义生成的每一帧的尺寸
HEIGHT = 224  # 高度为224像素
WIDTH = 224   # 宽度为224像素
class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
        一个卷积层的组合，首先对空间维度进行卷积操作，
        然后对时间维度进行卷积操作。
        """
        super().__init__()
        self.seq = keras.Sequential([  
            # 空间维度分解卷积
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),  # 只对高度和宽度进行卷积
                          padding=padding),
            # 时间维度分解卷积
            layers.Conv3D(filters=filters, 
                          kernel_size=(kernel_size[0], 1, 1),  # 只对时间步长进行卷积
                          padding=padding)
        ])

    def call(self, x):
        # 执行序列化卷积操作
        return self.seq(x)
class ResidualMain(keras.layers.Layer):
    """
    模型中的残差模块，包含卷积、层归一化和 ReLU 激活函数。
    """
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            # 第一个卷积层
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            # 层归一化
            layers.LayerNormalization(),
            # ReLU 激活函数
            layers.ReLU(),
            # 第二个卷积层
            Conv2Plus1D(filters=filters, 
                        kernel_size=kernel_size,
                        padding='same'),
            # 层归一化
            layers.LayerNormalization()
        ])

    def call(self, x):
        # 执行序列化的操作
        return self.seq(x)
class Project(keras.layers.Layer):
    """
    通过不同大小的过滤器和下采样，对张量的某些维度进行投影处理。
    """
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            # 全连接层（投影操作）
            layers.Dense(units), 
            # 层归一化
            layers.LayerNormalization()
        ])

    def call(self, x):
        # 执行顺序操作
        return self.seq(x)
def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        """
        初始化视频尺寸调整层。

        Args:
            height: 调整后的高度。
            width: 调整后的宽度。
        """
        super().__init__()
        self.height = height
        self.width = width
        # 使用 Keras 的 Resizing 层来调整尺寸
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
        调整视频张量的尺寸。

        Args:
            video: 表示视频的张量，形状为 (batch, time, height, width, channels)。

        Returns:
            调整为新高度和宽度的视频张量。
        """
        # 解析视频的原始形状：b 表示批次大小，t 表示时间步，h 和 w 表示高度和宽度，c 表示通道数
        old_shape = einops.parse_shape(video, 'b t h w c')

        # 将视频重新排列为单张图像的形式，合并批次和时间维度
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')

        # 调整每一帧的尺寸
        images = self.resizing_layer(images)

        # 将调整后的图像重新排列为视频的形式，分离批次和时间维度
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t']
        )
        return videos

input_shape = (None, 20, HEIGHT, WIDTH, 3)  # 输入视频的形状，None 表示批次大小不固定
input = layers.Input(shape=(input_shape[1:]))  # 定义输入层，形状为 (时间步数, 高度, 宽度, 通道数)
x = input

# 初始卷积层：执行 2+1D 卷积操作（空间 + 时间分解）
x = Conv2Plus1D(filters=16, kernel_size=(6, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)  # 批量归一化层，规范化每批次的特征
x = layers.ReLU()(x)  # 激活函数 ReLU
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)  # 调整视频帧的尺寸到 (HEIGHT/2, WIDTH/2)

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

# 从训练数据集中获取一个批次的数据
frames, label = next(iter(train_ds))

# 通过 build 方法将模型与输入张量关联，用于可视化或调试
model.build(frames)

# 使用 Keras 提供的工具绘制模型结构
keras.utils.plot_model(
    model,               # 目标模型
    expand_nested=True,  # 展开嵌套的层，例如子模块或自定义层
    dpi=60,              # 设置图片分辨率
    show_shapes=True     # 显示每一层输出的形状
)


# 编译模型
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 使用稀疏分类交叉熵作为损失函数
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),              # 优化器为 Adam，学习率设置为 0.0001
    metrics=['accuracy']                                               # 评估指标为准确率
)

history = model.fit(
    x=train_ds,            # 使用训练数据集
    epochs=50,             # 训练 50 轮
    validation_data=val_ds # 使用验证数据集
)

# 假设您已有一个训练好的模型 'model'
save_path = r'C:\data\result\model\best_model.h5'

# 保存整个模型
model.save(save_path)

print(f"模型已保存到：{save_path}")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import einops

# 定义自定义的 Conv2Plus1D 层，修改以接受 trainable 等参数
class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super().__init__(**kwargs)  # 接受所有传递给父类的参数，包括 trainable
        self.seq = keras.Sequential([  
            # 空间维度卷积
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # 时间维度卷积
            layers.Conv3D(filters=filters, 
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

# 定义残差块的主要模块
class ResidualMain(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)  # 接受所有传递给父类的参数，包括 trainable
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

# 定义用于投影的层
class Project(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)  # 接受所有传递给父类的参数，包括 trainable
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

# 定义 ResizeVideo 层，确保在加载时也能恢复参数
class ResizeVideo(keras.layers.Layer):
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
        config = super(ResizeVideo, self).get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config

# 加载模型函数
# 加载模型函数
def load_my_model(model_path):
    # 定义自定义层
    custom_objects = {
        'Conv2Plus1D': Conv2Plus1D,
        'ResizeVideo': ResizeVideo,
        'ResidualMain': ResidualMain,
        'Project': Project  # 确保 Project 也被添加到 custom_objects 中
    }
    
    # 使用 Keras 加载模型
    model = load_model(model_path, custom_objects=custom_objects)
    
    # 编译模型
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

# 调用加载模型的函数，假设您给的路径是 model_path
model_path = r'C:\data\result\model\best_model.h5'  # 替换为您实际的模型文件路径
model = load_my_model(model_path)

# 可选：打印模型摘要
model.summary()
