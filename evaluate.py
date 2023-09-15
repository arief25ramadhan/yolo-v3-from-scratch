import config
import torch
import torch.optim as optim

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

from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def main():
    
    ## Load model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

    model_path = 'model/yolov3_pascal_78.1map.pth.tar'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
   
    _, test_loader, _ = get_loaders(
        train_csv_path = config.DATASET + '/train.csv',
        test_csv_path = config.DATASET + '/test.csv',
    )

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1,3,2)
    ).to(config.DEVICE)
      
    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")
        

if __name__ == "__main__":
    main()