# Import libraries
import os
import config
import torch
import torch.optim as optim
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint, 
    load_checkpoint, 
    check_class_accuracy,
    get_loaders, 
    plot_couple_examples, 
    non_max_suppression,
    plot_image,
    draw_and_save_image
)


## 1. Data Transform
test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)


## 2. Load model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

model_path = 'model/yolov3_pascal_78.1map.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

## 3. Inference function
def inference_image(image_path, model, image_transform, device='cuda'):

    model = model.to(device)
    
    img = np.array(Image.open(image_path).convert("RGB"))
    # get normalized image
    img_normalized = image_transform(image=img)
    img_normalized = img_normalized['image'].unsqueeze(0).to(device)
    print(img_normalized.shape)

    iou_threshold=config.NMS_IOU_THRESH
    anchors=config.ANCHORS
    S = config.S
    threshold=config.CONF_THRESHOLD
    box_format="midpoint"
    all_pred_boxes = []

    model.eval()
    
    with torch.no_grad():

        predictions = model(img_normalized)
    
    batch_size = img_normalized.shape[0]
    bboxes = [[] for _ in range(batch_size)]
   
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(
            predictions[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append(nms_box)

    print("Plot Image")
    file_name = os.path.basename(image_path)
    print('file_name: ', file_name)
    save_path = 'media/' + file_name.replace('.jpg', '_predicted.jpg')
    print('save_path: ', save_path)
    draw_and_save_image(img, all_pred_boxes, save_path=save_path)


## 4. Inference
image_path = 'dataset/PASCAL_VOC/images/000004.jpg'
inference_image(image_path, model, test_transform)