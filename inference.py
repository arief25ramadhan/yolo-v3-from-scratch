# Import libraries
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
    plot_couple_examples
)

print(config.IMAGE_SIZE)

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
    # bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


## 2. Load model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

model_path = 'model/yolov3_pascal_78.1map.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])

# # Inference function
def inference_image(image_path, model, image_transform, device='cuda'):

    model = model.to(device)
    
    img = np.array(Image.open(image_path).convert("RGB"))
    # get normalized image
    img_normalized = image_transform(image=img)
    img_normalized = img_normalized['image'].unsqueeze(0).to(device)
    print(img_normalized.shape)

    model.eval()
    
    with torch.no_grad():

        preds = model(img_normalized)
        print('preds: ', len(preds))
        print('preds: ', len(preds[0][0]))
        

image_path = 'dataset/PASCAL_VOC/images/000015.jpg'
inference_image(image_path, model, test_transform)