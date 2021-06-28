from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from utils.parser import get_config
import logging
import traceback

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

from time import gmtime, strftime
import random
warnings.filterwarnings('ignore')

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

MODEL_DIR = cfg.SERVICE.MODEL
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.PORT
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
DEVICE_ID = cfg.SERVICE.DEVICE_ID
BACKUP_DIR = cfg.SERVICE.BACKUP_DIR

# create logging
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = FastAPI()

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

model_test = AntiSpoofPredict(DEVICE_ID)

def test(image_name, image):
    
    image_cropper = CropImage()
    result = check_image(image)
    # if result is False:
    #     print("check")
    #     return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    # if label == 1:
    #     print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
    #     result_text = "RealFace Score: {:.2f}".format(value)
    #     color = (255, 0, 0)
    # else:
    #     print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
    #     result_text = "FakeFace Score: {:.2f}".format(value)
    #     color = (0, 0, 255)
    # print("Prediction cost {:.2f} s".format(test_speed))
    # cv2.rectangle(
    #     image,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color, 2)
    # cv2.putText(
    #     image,
    #     result_text,
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(BACKUP_DIR + result_image_name, image)
    return label, value

class Prediction(BaseModel):
    fake: str 
    score: str

@app.post('/', response_model=Prediction)
async def predict_color(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = time + '_' + number + '.jpg'

        label, value = test(img_name, image)

        result = {"fake" : str(label), "score": str(value)}

        print(result)
        return result
        # return jsonify(result = result)


    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app_v2:app", host=HOST, port=PORT)