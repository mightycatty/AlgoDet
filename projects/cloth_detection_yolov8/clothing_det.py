"""clothing detection with yolov8
# installation
# supported categories:
"""
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np
def nms(bounding_boxes, confidence_score, labels, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_label
class ClothDet:
    def __init__(self, model_dir, device='cuda', post_processing_fn=None):
        if (not torch.cuda.is_available()) and device == 'cuda':
            print('gpu not found, fall back to cpu')
            device = 'cpu'
        self.model = YOLO(model_dir)  # pretrained YOLOv8n model
        self._device = device
        self.post_processing_fn = post_processing_fn

    def __call__(self, image_rgb: Image.Image, nms_iou=0.5, score_thershold=0.5, show=False, save=False, **kwargs):
        results = self.model.predict(source=image_rgb, device=self._device, nms=True, iou=nms_iou, conf=score_thershold,**kwargs)
        assert len(results) == 1, 'only support 1 image input'
        results = results[0].cpu().numpy()

        scores = np.float32(results.boxes.conf)
        bboxes = np.int16(results.boxes.xyxy)[scores > score_thershold].tolist()
        labels = np.int16(results.boxes.cls)[scores > score_thershold].tolist()
        scores = scores.tolist()
        classes_list = results.names
        labels = [classes_list[item] for item in labels]
        ret = (bboxes, scores, labels)
        ret = nms(*ret, nms_iou)
        if self.post_processing_fn is not None:
            ret = self.post_processing_fn(ret)
        return ret


def post_processing_fn(results):
    bboxes, scores, labels = results
    class_project = {
        't-shirt': ['top, t-shirt, sweatshirt', 'vest','shorts'],
        'coat': ['cardigan', 'jacket', 'coat', 'cape'],
        'shirt':['shirt, blouse'],
        'pants': ['pants'],
        'one-piece dress':['dress', 'jumpsuit'],
        'skirt': ['skirt']
    }
    labels_new = []
    for label_item in labels:
        for key, value in class_project.items():
            if label_item in value:
                labels_new.append(key)
                break
    return bboxes, scores, labels_new


if __name__ == '__main__':
    from algo_qol.utils.file_utils import get_file_recursively
    from algo_qol.algo.detection.det_utils import det_vis
    # Load a model
    model_dir = r'/home/justsomeone/code/AlgoQOL/algo_qol/algo/detection/ultralytics_train/runs/detect/train5/weights/best.pt'
    model = ClothDet(model_dir, post_processing_fn=post_processing_fn)

    test_f = r'/home/justsomeone/Downloads/tb'
    img_list = get_file_recursively(test_f)
    for item in img_list:
        name = os.path.basename(item)
        item = Image.open(item).convert('RGB')
        result = model(item)
        det_result = det_vis(np.array(item), *result)
        plt.imsave(name, det_result)
        # plt.imshow(det_result)
        # plt.show()
