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
#url = 'http://0.0.0.0:2341/predict'
url = 'https://aiclub.uit.edu.vn/gpu/service/craft_ocr_fastapi/predict_binary'
####################################
image_path = "text_detection_test_image.png"
####################################
f = {'file': open(image_path, 'rb')}
####################################
response = requests.post(url, files = f)
response = response.json()
print(response)
print('time', time.time()-start_time)
