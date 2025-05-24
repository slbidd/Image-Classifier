# 图像分类器 - Image Classifier

一个基于 EfficientNet-B0 的简易通用深度学习图像分类项目。该项目提供了完整的训练、验证和推理流程，可以轻松适配到各种图像分类任务。

A simple general-purpose deep learning image classification project based on EfficientNet-B0. This project provides a complete training, validation, and inference pipeline that can be easily adapted to various image classification tasks.

## 项目特点 Features

- 高效模型: 基于预训练的 EfficientNet-B0，兼顾性能与效率
- 完整流程: 包含数据预处理、模型训练、验证和测试的完整pipeline
- 可视化: 自动生成训练过程的损失和准确率曲线
- 灵活推理: 支持单张图片预测和批量测试
- 自动保存: 自动保存最佳模型和训练指标
- 高准确率: 在验证集上可达到90%+的准确率
- 易于定制: 可轻松修改类别数量和标签

## 环境要求 Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）

## 安装 Installation

### 步骤1: 克隆项目
```text
git clone https://github.com/slbidd/Image-Classifier
cd Image-Classifier
```
### 步骤2: 创建虚拟环境（推荐）
```text
python -m venv .venv
```
### 步骤3: 激活虚拟环境
# Windows:
```text
.venv\Scripts\activate
```
# Linux/Mac:
```text
source .venv/bin/activate
```
### 步骤4: 安装依赖
```text
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
**注意：如果你的 CUDA 版本不是 12.1，请访问[ PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应 CUDA 版本的安装命令。**
## 快速开始 Quick Start

### 1. 准备数据集

在 data/raw_images 目录下按类别组织你的图片：
```text
data/raw_images/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   └── ...
└── 类别3/
    └── ...
```
### 2. 划分数据集
```text
cd scripts
python split_dataset.py
```
这将自动将数据按8:2的比例划分为训练集和验证集。

### 3. 训练模型
```text
python train.py
```
训练完成后，最佳模型将保存在 outputs/model_best.pth。

### 4. 测试模型

单张图片预测：
```text
python test_single_image.py
```
批量测试：
```text
python test.py
```
## 项目结构 Project Structure
```text
anime_classifier/
├── data/                    # 数据目录
│   ├── raw_images/         # 原始图片
│   ├── train/              # 训练集
│   └── val/                # 验证集
├── model/                   # 模型定义
│   └── efficientnet_b0.py  # EfficientNet-B0模型
├── scripts/                 # 脚本文件
│   ├── train.py            # 训练脚本
│   ├── test.py             # 批量测试
│   ├── test_single_image.py # 单图预测
│   ├── split_dataset.py    # 数据集划分
│   ├── plot_metrics.py     # 绘制训练曲线
│   └── check_class_indices.py # 检查类别索引
├── outputs/                 # 输出目录
│   ├── model_best.pth      # 最佳模型
│   ├── metrics.json        # 训练指标
│   └── training_plot.png   # 训练曲线图
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```
## 自定义配置 Customization

### 修改类别数量

1. 在 model/efficientnet_b0.py 中修改 num_classes 参数
2. 在 scripts/test_single_image.py 中更新 label_map 字典
3. 重新组织数据集目录结构

### 调整训练参数

在 scripts/train.py 中可以修改：
- batch_size: 批次大小
- lr: 学习率
- epochs: 训练轮数
- 数据增强策略

### 更换模型

可以在 model/efficientnet_b0.py 中替换为其他预训练模型，如：
- ResNet系列
- VGG系列
- MobileNet系列

## 性能监控 Performance Monitoring

项目会自动：
- 保存训练过程中的损失和准确率
- 生成可视化的训练曲线
- 在验证集上评估模型性能
- 保存最佳模型权重


## 许可证 License

本项目采用 MIT 许可证。

## 致谢 Acknowledgments

- EfficientNet - 高效的卷积神经网络架构
- PyTorch - 深度学习框架
- torchvision - 计算机视觉工具包