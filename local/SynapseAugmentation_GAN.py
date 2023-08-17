# GAN pipeline tool for augmenting synaptic training data (WORKSPACE)
# Scripts by Henry Low

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import filedialog

def load_images_from_directory():
    # Create main window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog
    directory = filedialog.askdirectory(title="Select directory containing synapse images")
    
    if not directory:
        raise Exception("No directory selected")

    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    images = [np.array(Image.open(os.path.join(directory, file)).convert('L')) for file in image_files]

    # Convert list of images to numpy array and normalize them
    return np.array(images) / 255.0
    
# Assumes images are 64x64 and grayscale
img_shape = (64, 64, 1)
latent_dim = 128  # Latent space dimension

# Generator Model
def build_generator():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_dim=latent_dim),
        keras.layers.Reshape((8, 8, 2)),
        keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')  # Output grayscale image
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile models
optimizer = keras.optimizers.Adam(0.0002, 0.5)

discriminator = build_discriminator()
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
z = keras.layers.Input(shape=(latent_dim,))
img = generator(z)

discriminator.trainable = False  # Only trains generator in the combined GAN model
valid = discriminator(img)
combined = keras.Model(z, valid)
combined.compile(optimizer=optimizer, loss='binary_crossentropy')

# Training GAN
def train_gan(epochs, batch_size, save_interval):
    # training images normalized between 0 and 1
    # Load images using dialog function
    X_train = load_images_from_directory()

    # If images aren't 64x64, resize them:
    X_train = np.array([np.array(Image.fromarray((img*255).astype(np.uint8)).resize((64, 64))) for img in X_train])

    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    
    valid_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, valid_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid_labels)
        
        # Print progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
        
        # Save generated images
        if epoch % save_interval == 0:
            save_imgs(epoch)

# Save Images
def save_imgs(epoch):
    rows, cols = 5, 5
    noise = np.random.normal(0, 1, (rows * cols, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to 0 - 1

    fig, axs = plt.subplots(rows, cols)
    count = 0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            count += 1
    fig.savefig(f"images/epoch_{epoch}.png")
    plt.close()
    
# Start training
train_gan(epochs=10000, batch_size=32, save_interval=200)