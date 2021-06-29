import base64
import urllib.parse
import requests
import json
import timeit
import sys
import io
import cv2
import numpy as np
import time
start_time = time.time()
url = 'http://service.aiclub.cs.uit.edu.vn/face_anti_spoofing/predict_binary'
####################################
image_path = "test_face.png"
####################################
f = {'file': open(image_path, 'rb')}
####################################
response = requests.post(url, files = f)
response = response.json()
print(response)
print('time', time.time()-start_time)
