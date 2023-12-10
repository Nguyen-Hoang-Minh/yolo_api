from ultralytics import YOLO
import json
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse

#Create an argument parser 
parser = argparse.ArgumentParser(description="YOLO object detection")

#Add an argument for image file path
parser.add_argument("image_path", type=str, help="Path to the image for detection")

#Parse the command-line argument
args = parser.parse_args()

# Load the YOLO model
model = YOLO('yolov8n-seg.pt')


    # Mount the directory containing the image you want to detect to the container
    #image_dir = '/path/to/image/directory'
    #os.environ['IMAGE_DIR'] = image_dir

# Get the path to the image you want to detect
image_path = args.image_path 

# Perform object detection on the image
results = model.predict(image_path)
json_results = []
image = cv2.imread(image_path)
for result in results:
    json_result = result.tojson()
    json_results.append(json_result)

for result in results:
    annotator = Annotator(image)    
    boxes = result.boxes
    for box in boxes:       
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])

image=annotator.result() 
cv2.imwrite('resulted.jpg', image)
# Save the JSON object to a file
with open('results.json', 'w') as f:
    json.dump(json_results, f, indent=4)


    