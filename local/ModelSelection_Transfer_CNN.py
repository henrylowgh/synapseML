# ModelSelection_Transfer_CNN
# Scripts by Henry Low

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog
import csv # for importing binary labels file

import yaml

# Load configurations from YAML file
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Accessing configurations in the program
image_resize = tuple(config['data']['image_resize'])
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
base_model_choice = config['model']['base_model']

def select_transfer_model(input_shape):
    while True:
        print("Select the transfer learning model you want to use:")
        print("1. VGG16")
        print("2. ResNet50")
        #print("3. U-Net (semantic segmentation)")
        
        choice = input("Enter the number corresponding to your choice: ")

        if choice == "1":
            return build_model_vgg16(input_shape)
        elif choice == "2":
            return build_model_resnet(input_shape)
        elif choice == "3":
            return build_unet_model(input_shape)
        else:
            print("Invalid choice. Please select again.")

# Function to load images using a dialog box
def load_images_from_directory():
    root = tk.Tk()
    root.withdraw()

    directory = filedialog.askdirectory(title="Select directory containing synapse images")
    if not directory:
        raise Exception("No directory selected")

    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    images = [np.array(Image.open(os.path.join(directory, file)).convert('RGB')) for file in image_files]

    return np.array(images) / 255.0

# Transfer Learning with VGG16 for Synapse Quantification
def build_model_vgg16(input_shape): # VGG16 expects 3-channel images
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False  # Freeze layers to retain pre-trained weights
    
    model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')  # Binary output
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to load labels using a dialog box
def load_labels_from_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select the labels CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_path:
        raise Exception("No file selected")

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        labels = [int(row[0]) for row in reader]

    return np.array(labels)

# Load data
X = load_images_from_directory()

# If images are not 64x64, resize
X = np.array([np.array(Image.fromarray((img*255).astype(np.uint8)).resize((64, 64))) for img in X])

# y contains labels (0 for no synapse, 1 for synapse)
y = load_labels_from_file()

# Split data into training and validation sets (80-20 split)
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Initialize and train the model
model = select_transfer_model((64, 64, 3))  
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

