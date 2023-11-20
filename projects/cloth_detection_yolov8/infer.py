import os

import cv2
from ultralytics import YOLO
from algo_qol.utils.file_utils import get_file_recursively
from algo_qol.algo.detection.det_utils import det_vis
# Load a model
test_f = r'/home/justsomeone/data/ur/ur'
img_list = get_file_recursively(test_f)

model = YOLO('/algo_qol/algo/detection/ultralytics_train/runs/detect/train3/weights/last.pt')  # pretrained YOLOv8n model
for item in img_list:
    results = model.predict(source=item, show=False, save=True, nms=True, iou=0.2, conf=0.5, device='cpu')
# Run batched inference on a list of images

# Process results list
# for img_name, result in zip(img_list, results):
#     result = result.cpu().numpy()
#     names = result.names
#     class_idx = result.boxes.cls
#     class_idx = [names[item] for item in class_idx]
#     scores = result.boxes.conf
#     bboxes = result.boxes.xyxy
#     vis_result = det_vis(result.orig_img, bboxes, scores, class_idx)
#     base_name = os.path.basename(img_name)
#     cv2.imwrite(base_name, vis_result)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', vis_result)
    # cv2.waitKey(0)
    # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
