from flask import Flask, render_template, request
import os
from utils import classify_img, detection_img, get_density_model, get_pothole_model, get_category_model
from PIL import Image
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load models
density_model = get_density_model()
pothole_model = get_pothole_model()
category_model = get_category_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/traffic_density', methods=['GET', 'POST'])
def traffic_density():
    result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = Image.open(file.stream)
            img = np.array(img)
            img = cv2.resize(img, (480, 480))
            label, probability = classify_img(density_model, img)
            classes = ['Empty', 'High', 'Low', 'Medium', 'Traffic Jam']
            result = f"Predicted Class is {classes[label]} with probability {probability:.4f}"
        else:
            result = "No file uploaded."
    return render_template('traffic_density.html', result=result)

@app.route('/vehicle_category', methods=['GET', 'POST'])
def vehicle_category():
    img_base64 = None
    if request.method == 'POST':
        file = request.files.get('file')
        conf_threshold = float(request.form.get('conf_threshold', 0.2))
        iou_threshold = float(request.form.get('iou_threshold', 0.6))
        if file:
            img = Image.open(file.stream)
            img = np.array(img)
            img = cv2.resize(img, (480, 480))
            img_base64 = detection_img(category_model, img, ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"], conf_threshold, iou_threshold)
    return render_template('vehicle_category.html', img=img_base64)

@app.route('/pothole', methods=['GET', 'POST'])
def pothole():
    img_base64 = None
    if request.method == 'POST':
        file = request.files.get('file')
        conf_threshold = float(request.form.get('conf_threshold', 0.2))
        iou_threshold = float(request.form.get('iou_threshold', 0.7))
        if file:
            img = Image.open(file.stream)
            img = np.array(img)
            img = cv2.resize(img, (480, 480))
            img_base64 = detection_img(pothole_model, img, ["Background", "Pothole"], conf_threshold, iou_threshold)
    return render_template('pothole.html', img=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
