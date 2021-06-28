from flask import Flask, render_template, request, jsonify

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

app = Flask(__name__)

device_id = 0
model_dir = "./resources/anti_spoof_models"
SAMPLE_IMAGE_PATH = "./OUTPUT/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

model_test = AntiSpoofPredict(device_id)

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
    for model_name in os.listdir(model_dir):
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
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    return label, value

@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = time + '_' + number + '.jpg'

        label, value = test(img_name, image)

        result = {"fake" : str(label), "score": str(value)}

        print(result)
        return jsonify(result = result)
        # return result
        # return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)