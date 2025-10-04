from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Get the absolute path to the model
model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "potato_disease_model.h5")
try:
    MODEL = tf.keras.models.load_model(model_path)
except Exception as e:
    # Defer failure until first request with clear message
    MODEL = None
    load_error = e
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return {"Hello, I am alive!"}

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB')
    image = np.array(img).astype(np.float32) / 255.0
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Create a batch
    if MODEL is None:
        return {"error": "model failed to load", "detail": str(load_error)}
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)