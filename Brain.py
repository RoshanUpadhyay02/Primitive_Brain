import cv2 as cv
import shutil
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import subprocess
import tkinter as tk

os.chdir(r'G:\Design of Artificial Intelligence Products\Project')
folder_path = r'G:\Design of Artificial Intelligence Products\Project\yolov5\runs\detect'
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path) and os.listdir(item_path):
        shutil.rmtree(item_path)

model_pokemon = load_model('Pokemon.h5')
model_lukemia = load_model('Lukemia.h5')
model_num = load_model('mnist.h5')

model_number = 0
input_text = ''

def apply_model():
    global model_number
    global input_text
    model_number = int(model_entry.get())
    input_text = input_entry.get()
    # use the model_number and input_text variables in your code here
    root.destroy()

root = tk.Tk()
root.geometry("500x300")

# create label for choosing model
choose_label = tk.Label(root, text="Choose a model from below:")
choose_label.pack()

# print options for choosing model
options_label = tk.Label(root, text='''1 : Pokemon
2 : Lukemia Cells
3 : Handwritten Digits
4 : Car''')
options_label.pack()

# create input field for model number
model_label = tk.Label(root, text="Enter which model you want to use:")
model_label.pack()
model_entry = tk.Entry(root)
model_entry.pack()


input_label = tk.Label(root, text="Enter path of image(url for model 1):")
input_label.pack()
input_entry = tk.Entry(root)
input_entry.pack()

# create button to apply the model
apply_button = tk.Button(root, text="Apply", command=apply_model)
apply_button.pack()

root.mainloop()

n = model_number

class Pokemon:
    def __init__(self, url, model, labels):
        self.url = url
        self.model = model
        self.labels = labels
        
    def preprocess_image(self):
        r = requests.get(self.url, stream=True).raw
        image = np.asarray(bytearray(r.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        image = cv.resize(image, (96, 96))
        image = image.reshape(-1, 96, 96, 3) / 255.0
        return image
        
    def predict_image(self):
        image = self.preprocess_image()
        preds = self.model.predict(image)
        pred_class = np.argmax(preds)
        return pred_class, preds[0][pred_class]
    
    def display_image(self):
        pred_class, confidence = self.predict_image()
        true_label = self.labels[pred_class]
        pred_label = f'Predicted: {true_label} {round(confidence * 100, 2)}%'
        
        r = requests.get(self.url, stream=True).raw
        image = np.asarray(bytearray(r.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        
        plt.imshow(image[:, :, ::-1])
        plt.title(pred_label)
        plt.axis('off')
        plt.show()
        
class LeukemiaImage:
    def __init__(self, image_path, m):
        self.image_path = image_path
        self.model = m
        
    def preprocess_image(self):
        image = cv.imread(self.image_path)
        image = cv.resize(image, (227, 227))
        image = image.reshape(-1, 227, 227, 3) / 255.0
        return image
        
    def predict_image(self):
        image = self.preprocess_image()
        preds = self.model.predict(image)
        pred_class = np.argmax(preds)
        confidence = preds[0][pred_class]
        if pred_class == 0:
            return 'ALL', confidence
        else:
            return 'AML', confidence
    
    def display_image(self):
        predicted_class, confidence = self.predict_image()
        plt.imshow(cv.imread(self.image_path))
        plt.title(f'Predicted: {predicted_class} {round(confidence * 100, 2)}%')
        plt.axis('off')
        plt.show()
class Hd:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def load_image(self):
        img = load_img(self.filename, grayscale=True, target_size=(28, 28))
        img = img_to_array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32')
        img = img / 255.0
        return img

    def run_example(self):
        img = self.load_image()
        predict_value = self.model.predict(img)
        digit = argmax(predict_value)
        
        root = tk.Tk()
        root.title("Large Font Text")
        root.geometry("600x400")
        options_label = tk.Label(root, text='''Predicted Number''')
        options_label.pack()

        label = tk.Label(root, text=digit, font=("Arial", 100))
        x = root.winfo_reqwidth() // 2
        y = root.winfo_reqheight() // 2
        label.place(relx=0.5, rely=0.5, anchor="center")
        root.mainloop()

class YOLOv5Detector:
    def __init__(self, weights_path, source_path):
        self.weights_path = weights_path
        self.source_path = source_path

    def detect(self):
        cmd = f"python detect.py --weights {self.weights_path} --source {self.source_path}"
        output = subprocess.check_output(cmd, shell=True)
        return output
    
if n == 1:
    path = input_text
    labels = ['mewtwo', 'pikachu', 'charmander', 'bulbasaur', 'squirtle']
    my_test_image = Pokemon(path, model_pokemon, labels)
    my_test_image.display_image()
    
elif n == 2:
    path = input_text
    my_leukemia_image = LeukemiaImage(path, model_lukemia)
    my_leukemia_image.display_image()

elif n == 3:
    path = input_text
    my_hd = Hd(path, model_num)
    my_hd.run_example()
    
elif n == 4:
    path = input_text
    os.chdir(r'G:\Design of Artificial Intelligence Products\Project\yolov5')
    detector = YOLOv5Detector('car.pt', path)
    detector.detect()
    folder_path = r"G:\Design of Artificial Intelligence Products\Project\yolov5\runs\detect\exp"
    p = ''
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            p = image_path
    im = Image.open(p)
    plt.imshow(im)
    plt.show()
    folder_path = r'G:\Design of Artificial Intelligence Products\Project\yolov5\runs\detect'
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and os.listdir(item_path):
            shutil.rmtree(item_path)
os.chdir(r'G:\Design of Artificial Intelligence Products\Project')