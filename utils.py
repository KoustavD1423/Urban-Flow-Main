import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDHead, det_utils
from torchvision.models import efficientnet_b0
import torchvision.transforms.functional as tf
import base64
from io import BytesIO
from PIL import Image

root_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(root_dir, "weights")

def classify_img(model, img):
    model.eval()  # Set the model to evaluation mode
    img = tf.to_tensor(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predict = model(img)
        predict = nn.functional.softmax(predict, dim=1)  # Ensure this is correct for your model output
        label = torch.argmax(predict, dim=1).item()
        probability = torch.max(predict).item()
        return label, probability

def detection_img(model, img, classes, conf_threshold, iou_threshold):
    model.eval()  # Set the model to evaluation mode
    img_tensor = tf.to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
        prediction = preprocess_bbox(prediction, conf_threshold, iou_threshold)
        img_with_boxes = show_bbox(img_tensor[0], prediction, classes)
        
        # Convert image with bounding boxes to base64
        img_with_boxes = (img_with_boxes * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_with_boxes)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return img_base64

# Utility function to convert a tensor image to a format that can be displayed in HTML
def show_bbox(img, target, classes, color=(0, 255, 0)):
    img = np.transpose(img.numpy(), (1, 2, 0))
    boxes = target["boxes"].numpy().astype("int")
    labels = target["labels"].numpy()
    scores = target["scores"].numpy()
    img = img.copy()
    for i, box in enumerate(boxes):
        text = f"{classes[labels[i]]} {scores[i]:.2f}"
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
        cv2.putText(img, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def preprocess_bbox(prediction, conf_threshold, iou_threshold):
    processed_bbox = {}
    boxes = prediction["boxes"][prediction["scores"] >= conf_threshold]
    scores = prediction["scores"][prediction["scores"] >= conf_threshold]
    labels = prediction["labels"][prediction["scores"] >= conf_threshold]

    nms = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)

    processed_bbox["boxes"] = boxes[nms]
    processed_bbox["scores"] = scores[nms]
    processed_bbox["labels"] = labels[nms]
    return processed_bbox

def get_density_model():
    model = efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=5)
    weights_path = os.path.join(weights_dir, "traffic_density.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    return model

def get_pothole_model():
    model = ssdlite320_mobilenet_v3_large(pretrained=False)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (480, 480))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=2)
    weights_path = os.path.join(weights_dir, "pothole_model.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    return model

def get_category_model():
    model = ssdlite320_mobilenet_v3_large(pretrained=False)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=9)
    weights_path = os.path.join(weights_dir, "vehicle_categorization.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    return model
