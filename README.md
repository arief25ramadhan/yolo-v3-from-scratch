# YOLOv3 Object Detection from Scratch

<img src="https://pjreddie.com/media/image/yologo_2.png" alt="drawing" width="100"/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This project aims to create the YOLOv3 (You Only Look Once version 3) Object Detection from scratch. YOLO is real-time object detection system that can detect multiple objects in a single frame with impressive accuracy.

We willl provide the step-by-step instructions, code, and resources to guide you through the process of implementing YOLOv3 from the ground up. Whether you are new to object detection or looking to deepen your understanding of YOLO,  we hope this repository will help you achieve your goals.

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

Object detection is a fundamental task in computer vision, and YOLO is one of the most popular and effective approaches. This project aims to teach you how YOLOv3 works by building it from scratch. By following the steps outlined here, you will gain a deep understanding of the architecture, training process, and inference pipeline of YOLOv3.

### 1.1. A brief explanation of the Model

Yolo v3 used a Feature Pyramid Network
<img src="https://miro.medium.com/v2/resize:fit:1200/1*d4Eg17IVJ0L41e7CTWLLSg.png" alt="drawing" width="800"/>




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
|-- dataset/                 # Folder containing data files
|-- model/                   # Folder containing model files
|-- train.py                 # Training script
|-- dataset.py               # Script to load dataset
|-- loss.py                  # Script to load loss function
|-- utils.py                 # Script containing helper functions
|-- config.py                # Script containing hyperparameters
|-- model.py                 # Script to load model
|-- inference.py             # Script to perform inference
```

## 4. Training

Training YOLOv3 from scratch requires data preparation, model configuration, and the training process itself.

### 4.1. Data Preparation

1. Download the PASCAL VOC dataset and annotations from [here](https://cocodataset.org/#download). The PASCAL VOC is a dataset of 20 class 

2. Organize your dataset in the following directory structure:

   ```
   dataset/
   |-- PASCAL_VOC/
       |-- images/          # Contain images
       |-- labels/          # Contain labels
       |-- train.csv        # Define image belongs to train set
       |-- test.csv         # Define image belongs to test set
   ```

### 4.2. Model Configuration

Configure the YOLOv3 model architecture in the `src/yolov3_model.py` file. You can customize the number of classes, anchors, and other hyperparameters to match your dataset.

### 4.3. Training Process

Run the training script to start training:

```bash
python train.py
```

You can monitor training progress using TensorBoard.

## 5. Inference

After training your YOLOv3 model, you can perform inference on images.

```bash
python inference.py --image path/to/your/image.jpg --model path/to/your/model_weights.h5
```

## 6. Performance Results

Figure 1 shows some of the inference results of our Yolo V3 model. We can see that the model does mispredicted.

### 6.1. Mean Average Precision
We train the Yolo V3 from the checkpoint created by Aladdin Persson. This project is only for learning. So, creating the most accurate model, which requires a lot of tuning and training, is not our priority.

| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 78.2              |
| YOLOv3 (MS-COCO)        | Will probably train on this at some point      |


### 6.2. Memory Consumption

To improve the efficiency of our model, we convert it to JIT format. This conversion results in a slimmer and faster model. Table 2 compares the performance of the original model and the scripted model.

## 7. Acknowledgement

### 7.1. The Original Paper
The implementation is based on the [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) by Joseph Redmon and Ali Farhadi.

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

### 7.2. The Code 

This project is for learning purposes and made by following the tutorial by Aladdin Persson in his [Youtube channel](https://www.youtube.com/watch?v=Grir6TZbc1M). The original code is also available in [his repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3).
