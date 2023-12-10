from flask import Flask, redirect, request, render_template, url_for
from ultralytics import YOLO
import json
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse
import socket
from PIL import Image

model = YOLO('yolov8n-seg.pt')




app = Flask(__name__)
#api = Api(app)

# class Docker(Resource):
#     def get(self):
#         docker_command = f'docker run -v {os.getcwd()}/street.jpg:/app/image.jpg minhhoang283/docker_yolo python /app/yolo.py /app/image.jpg'
#         output = subprocess.check_output(docker_command, shell=True)
#         return {"data": "Docker is running"}

@app.route("/")
def test():
    address = socket.gethostbyname(socket.gethostname())
    return {"data": address}

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    results = model.predict(img)
    return results[0].tojson()



if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)