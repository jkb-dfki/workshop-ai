# System Imports
import sys
import argparse

# Third-Party Imports
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms



COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3))
DETECTION_THRESHOLD = 0.8



def predict(image):
    
    transform = transforms.Compose([transforms.ToTensor()])
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    print(outputs)
    pred_classes = [COCO_NAMES[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= DETECTION_THRESHOLD].astype(np.int32)

    return boxes, pred_classes, outputs[0]['labels']



def draw_boxes(boxes, classes, labels, image):
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

    for i, box in enumerate(boxes):

        color = COCO_COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)

    return image



image = Image.open(sys.argv[1])
boxes, classes, labels = predict(image)
image = draw_boxes(boxes, classes, labels, image)
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
