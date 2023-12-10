from flask import Flask, redirect, request, render_template, url_for
from ultralytics import YOLO
import json
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse
import socket
import base64
import io
from io import BytesIO
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

def decode_image(request):
  """
  Decodes the base64 encoded image data from the request body.

  Args:
    request: The HTTP request object.

  Returns:
    A PIL Image object containing the decoded image data.
  """
  image_data = request.form.get('image')
  if image_data is None:
    raise Exception('Missing image data in request')
  image_bytes = base64.b64decode(image_data)
  image_stream = BytesIO(image_bytes)
  image = Image.open(image_stream)
  return image

@app.route("/predict", methods=['POST'])
def predict():
    # Decode the image data
    image = decode_image(request)
    # Get the encoded image from the request
    # encoded_image = request.files['image'].read()

    # Decode the image
    # image_bytes = base64.b64decode(encoded_image)
    # image = Image.open(io.BytesIO(image_bytes))
    #file = request.files['image']
    # img = Image.open(image_data.stream)
    results = model.predict(image)
    return results[0].tojson()



if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)