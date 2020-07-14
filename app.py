# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:35:45 2020

@author: Vivek
"""
from flask import Flask, render_template
import cv2
from tensorflow.keras.models import load_model         

myapp = Flask(__name__)

@myapp.route('/')
def fun1():
    return render_template('index.html')
    
@myapp.route('/predict',methods = ['GET','POST'])
def fun2():
    img = cv2.imread('0.jpg')
    model = load_model('CNNmodel.h5')
    img = cv2.resize(img,(28,28))
    img = img[:,:,0].reshape(1,28,28,1)
    return model.predict(img).argmax()

if __name__ == '__main__':
    myapp.run(debug=True)