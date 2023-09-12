# YOLOv3 Object Detection from Scratch

<img src="https://pjreddie.com/media/image/yologo_2.png" alt="drawing" width="100"/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This project aims to create the YOLOv3 (You Only Look Once version 3) Object Detection from scratch. YOLO is real-time object detection system that can detect multiple objects in a single frame with impressive accuracy.

In this project, we provide step-by-step instructions, code, and resources to guide you through the process of implementing YOLOv3 from the ground up. Whether you are new to object detection or looking to deepen your understanding of YOLO, this repository will help you achieve your goals.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training](#training)
   - [Data Preparation](#data-preparation)
   - [Model Configuration](#model-configuration)
   - [Training Process](#training-process)
5. [Inference](#inference)
6. [Performance Tuning](#performance-tuning)

## 1. Introduction

Object detection is a fundamental task in computer vision, and YOLOv3 is one of the most popular and effective approaches for real-time object detection. This project aims to teach you how YOLOv3 works by building it from scratch. By following the steps outlined here, you will gain a deep understanding of the architecture, training process, and inference pipeline of YOLOv3.

## 2. Getting Started

### 2.1. Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Python
- Pytorch
- NumPy
- OpenCV
- [Pascal VOC Dataset](https://cocodataset.org/#download)

### 2.2. Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/arief25ramadhan/yolo-v3-from-scratch.git
   ```

2. Navigate to the project directory:

   ```bash
   cd yolo-v3-from-scratch
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## 3. Project Structure

```
/
|-- data/                    # Data-related files (datasets, annotations, etc.)
|-- model/                   # YOLOv3 model architecture and checkpoints
|-- notebooks/               # Jupyter notebooks for experimenting
|-- src/                     # Source code for YOLOv3 implementation
|-- utils/                   # Utility functions
|-- images/                  # Images used in README and documentation
|-- README.md                # Project README (you are here)
|-- requirements.txt         # List of Python dependencies
|-- LICENSE                  # License information
```

## 4. Training

Training YOLOv3 from scratch requires data preparation, model configuration, and the training process itself.

### 4.1. Data Preparation

1. Download the COCO dataset and annotations from [here](https://cocodataset.org/#download). You will need the train and validation sets.

2. Organize your dataset in the following directory structure:

   ```
   /data/
   |-- coco/
       |-- train2017/          # Training images
       |-- val2017/            # Validation images
       |-- annotations/
           |-- instances_train2017.json  # Training annotations
           |-- instances_val2017.json    # Validation annotations
   ```

3. Convert the annotations to YOLO format using the provided script in the `src/` directory.

### 4.2. Model Configuration

Configure the YOLOv3 model architecture in the `src/yolov3_model.py` file. You can customize the number of classes, anchors, and other hyperparameters to match your dataset.

### 4.3. Training Process

Run the training script to start training:

```bash
python src/train.py
```

You can monitor training progress using TensorBoard.

## 5. Inference

After training your YOLOv3 model, you can perform inference on images.

```bash
python inference.py --image path/to/your/image.jpg --model path/to/your/model_weights.h5
```

## 6. Performance Results

### 6.1. Accuracy
We train the Yolo V3 from the checkpoint created by Aladdin Persson. This project is only for learning. So, creating the most accurate model, which requires a lot of tuning and training, is not our priority.

### 6.2. Memory Consumption

To improve the efficiency of our model, we convert it to JIT format. This conversion results in a slimmer and faster model. Table 2 compares the performance of the original model and the scripted model.
