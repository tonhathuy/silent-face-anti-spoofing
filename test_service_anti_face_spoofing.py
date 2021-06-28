import requests
import json


URL = 'http://service.aiclub.cs.uit.edu.vn/face_anti_spoofing/'

img_path = 'image_F1.jpg'


response = requests.post(URL, files={"file": (
    "filename", open(img_path, "rb"), "image/jpeg")}).json()

print(response)


