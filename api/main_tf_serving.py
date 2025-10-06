from fastapi import FastAPI, File, UploadFile
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import json

app = FastAPI()
TF_SERVING_URL = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB').resize((256,256))
    image = np.array(img).astype(np.float32)
    # do NOT normalize here if the SavedModel expects 0-255 input and includes a Rescaling layer
    return image.tolist()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    payload = {"instances": [image]}
    resp = requests.post(TF_SERVING_URL, json=payload)
    if resp.status_code != 200:
        return {"error": f"TF Serving returned {resp.status_code}", "detail": resp.text}
    data = resp.json()
    predictions = data.get('predictions')
    if not predictions:
        return {"error": "no predictions in TF Serving response", "detail": data}
    preds = np.array(predictions)
    predicted_class = CLASS_NAMES[np.argmax(preds[0])]
    confidence = float(np.max(preds[0]))
    return {"class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=False)
