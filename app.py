#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# deploy the service to the website
import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER  
CAPTCHA_LEN = 4            
CAPTCHA_HEIGHT = 60        
CAPTCHA_WIDTH = 160       


MODEL_FILE = './model/train_demo/captcha_adam_binary_crossentropy_bs_100_epochs_4.h5'

def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text

def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B 
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def customerAcc(y_true, y_pred):
    batch_size = y_true.shape[0]
    y_pred = y_pred.reshape(batch_size, 4, 10)
    y_true = y_true.reshape(batch_size, 4, 10)
    
    true_pred = K.sum(K.round(K.clip(y_pred, 0 ,1)), -1)
    true_positive = K.sum(K.round(K.clip(y_pred * y_true, 0 ,1)), -1)
    true_y = K.sum(K.round(K.clip(y_true,0,1)), -1)
    accuracy = K.sum((true_positive == true_y and true_pred == true_y), -1)
    accuracy = K.mean(accuracy==4)
    return accuracy
    


app = Flask(__name__) 


@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'captcha service start'

@app.route('/predict', methods=['POST'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image= False
    if request.method == 'POST':
        if request.files.get('image'): 
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json(): 
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            #with graph.as_default():
            pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)

model = load_model(MODEL_FILE) 
#graph = tf.get_default_graph() 
# export FLASK_ENV=development && flask run --host=0.0.0.0
# curl -X POST -F image=@0155.png 'http://localhost:5000/predict'

