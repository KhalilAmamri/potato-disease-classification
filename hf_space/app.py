# app.py - Gradio demo for the potato disease classifier
import os
import json
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

# The Space will run this file from its repo root. Place potato_disease_model.h5 and class_names.json
# in the same folder in the Space (or adjust the paths below).
H5_PATH = "potato_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"

# Load model
if os.path.exists(H5_PATH):
    model = tf.keras.models.load_model(H5_PATH)
else:
    raise FileNotFoundError(f"Model file not found: {H5_PATH}. Upload potato_disease_model.h5 to the Space repository.")

# Load class names
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    # fallback
    class_names = ["Early Blight", "Late Blight", "Healthy"]

# Preprocess helper
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((256, 256))
    arr = np.array(img).astype(np.float32)
    # if your model includes a Rescaling layer, send raw 0..255 values; otherwise divide by 255.
    batch = np.expand_dims(arr, axis=0)
    return batch

# Prediction function used by Gradio
def predict(image: Image.Image):
    batch = preprocess(image)
    preds = model.predict(batch)[0]
    # return mapping label->score
    out = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return out

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Potato disease classifier",
    description="Upload a potato leaf image (256x256 will be used)."
)

if __name__ == "__main__":
    iface.launch()
