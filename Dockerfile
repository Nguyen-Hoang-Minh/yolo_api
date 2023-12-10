FROM python:3.11.5
#FROM nvidia/cuda:latest
RUN pip install ultralytics

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install flask
COPY . /app

WORKDIR /app

EXPOSE 5000 

#RUN chmod  +x yolo.py

CMD ["python", "server.py"]