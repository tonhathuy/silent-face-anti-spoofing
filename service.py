import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
import uvicorn
import cv2
import traceback
import asyncio
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from configparser import ConfigParser
import rcode
import warnings

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')
now = datetime.datetime.now()
#######################################
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))
WORKER_NUM = str(config.get('main', 'WORKER_NUM'))
MODEL = str(config.get('main', 'MODEL'))
DEVICE_ID = int(config.get('main', 'DEVICE_ID'))
#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None, example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None, example=None, description='List of base64 encoded images')
#######################################
model_test = AntiSpoofPredict(DEVICE_ID)
for model_name in os.listdir(MODEL):
    print(model_name)
    model_test._load_model(os.path.join(MODEL, model_name))
def test_face_croped(image):
    image_cropper = CropImage()
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(MODEL):
        
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": None,
            "scale": None,
            "out_w": w_input,
            "out_h": h_input,
            "crop": False,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img)
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    return label, value
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("WORKER_NUM", WORKER_NUM)
print("API READY")
#######################################
@app.post('/predict')
async def predict(data: PredictData):
    ###################
    #####
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            images = jsonable_encoder(data.images)
        except Exception as e:
            print(str(e))
            print(str(traceback.print_exc()))
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        for image in images:
            image_decoded = base64.b64decode(image)
            jpg_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
            process_image = cv2.imdecode(jpg_as_np, flags=1)
            label, value = test_face_croped(process_image)
            label = -1 if (label != 1) else 1
            predicts = [{"fake" : label, "score": value}]
        return_result = {'code': '1000', 'status': rcode.code_1000, 'data': {'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time, 'WORKER_NUM': WORKER_NUM, 'return': '1: real, -1: fake'}}
    except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_binary')
async def predict_binary(file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            img_file = await file.read()
        except Exception as e:
            print(str(e))
            print(str(traceback.print_exc()))
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(img_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        label, value = test_face_croped(process_image)
        label = -1 if (label != 1) else 1
        predicts = [{"fake" : label, "score": value}]
        return_result = {'code': '1000', 'status': rcode.code_1000, 'data': {'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time, 'WORKER_NUM': WORKER_NUM, 'return': '1: real, -1: fake'}}
    except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)


