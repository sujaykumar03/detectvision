import cv2
import numpy as np
import os

# Paths
MODEL_PATH = 'model/frozen_inference_graph.pb'
CONFIG_PATH = 'model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
LABELS_PATH = 'model/labels.txt'
INPUT_FOLDER = 'data'
OUTPUT_FOLDER = 'outputs'

# Load labels
with open(LABELS_PATH, 'r') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Load the model
net = cv2.dnn_DetectionModel(MODEL_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Make sure outputs folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Loop over all images in data/
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                label = f"{class_names[classId - 1]}: {round(confidence * 100, 1)}%"
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, img)
        print(f"Processed: {filename} → saved to {output_path}")

