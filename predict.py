import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to your trained model
MODEL_PATH = "saved_models/recycle_mobilenetv2.keras"
IMG_SIZE = (180, 180)

def predict(img_path):
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    # Load model
    model = load_model(MODEL_PATH)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]  # Sigmoid output
    if prediction > 0.5:
        class_name = "recyclable"
        confidence = prediction
    else:
        class_name = "non_recyclable"
        confidence = 1 - prediction

    print(f"Prediction: {class_name}  Confidence: {confidence:.2%}")

    # Show image with label
    plt.imshow(img)
    plt.title(f"{class_name} ({confidence:.2%})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found at {args.image}")
        exit(1)

    predict(args.image)
