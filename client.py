import requests

BASE = "http://localhost:5000"

my_img = {'image': open('street.jpg', 'rb')}

reponse = requests.post(url = BASE+"/predict", files=my_img)

print(reponse.json())