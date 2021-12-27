#
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from logging import getLogger, Formatter, StreamHandler, FileHandler
import base64
from flask import Flask, render_template, request, send_file

import queue
import threading
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pi_resnet50
from tensorflow.keras.applications.xception import Xception as xception
from tensorflow.keras.applications.xception import preprocess_input as pi_xception
from tensorflow.keras.applications.imagenet_utils import decode_predictions

LOG_FILE = 'logs/inference.log'

logger = getLogger("Inference")
logger.setLevel(logging.DEBUG)
handler_format = Formatter('%(asctime)s [%(levelname)s] %(message)s')
stream_handler = StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(handler_format)
file_handler = FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

request_queue = queue.Queue()

class Classifier:
    def __init__(self, request_queue, model_type):
        self.request_queue = request_queue
        self.model_type = model_type

    def run(self):
        t = threading.Thread(target=self.classify)
        t.start()

    def classify(self):
        tf.disable_eager_execution()
        if self.model_type == "xception":
            logger.info('loading Xception model')
            model = xception(weights='imagenet')
        else:
            logger.info('loading ResNet50 model')
            model = resnet50(weights='imagenet')

        while True:
            response_queue, img = self.request_queue.get()
            logger.info('classify start')
            if self.model_type == "xception":
                pil_img = Image.open(img).resize((299,299))
                preprocess = pi_xception
            else:
                pil_img = Image.open(img).resize((224,224))
                preprocess = pi_resnet50
            x = image.img_to_array(pil_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess(x)
            preds = model.predict(x)
            res = decode_predictions(preds, top=5)
            cls = res[0][0][1]
            conf = res[0][0][2]
            logger.info('classify end (cls={} conf={})'.format(cls, str(conf)))
            response_queue.put((cls, conf))

cfr=Classifier(request_queue, "resnet50")
#cfr=Classifier(request_queue, "xception")
cfr.run()

app = Flask(__name__)

@app.route('/')
@app.route('/inference')
def inference():
    logger.info('inference')
    return render_template('inference.html', inf_img="", inf_class="", inf_conf="")

@app.route('/upload', methods=['POST'])
def upload():
    logger.info('upload')
    response_queue = queue.Queue()
    img = request.files['image']
    request_queue.put((response_queue, img))
    res_cls, res_conf = response_queue.get()
    img.seek(0)
    img_str = 'data:image/jpeg;base64,' + base64.b64encode(img.read()).decode('utf-8')
    return render_template('inference.html', inf_img=img_str, 
        inf_class=res_cls, inf_conf=str(res_conf))

@app.route('/log')
def log():
    with open(LOG_FILE, 'r') as f:
        log = f.readlines()
    return render_template('log.html', log_name=LOG_FILE, log_content=log)

@app.route('/download_log', methods=['POST'])
def download_log():
    logger.info('download_log')
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
