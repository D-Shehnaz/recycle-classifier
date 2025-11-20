# utils.py
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

def ensure_dirs():
    os.makedirs("saved_models", exist_ok=True)

def load_and_prep_image(path, img_size=(180,180)):
    img = Image.open(path).convert("RGB").resize(img_size)
    arr = np.array(img)/255.0
    return arr
