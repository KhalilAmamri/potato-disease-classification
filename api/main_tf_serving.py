from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import json
import os
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TF Serving endpoints
TF_SERVING_BASE = os.environ.get('TF_SERVING_BASE', 'http://localhost:8501/v1/models/potatoes_model')
TF_SERVING_URL = TF_SERVING_BASE + ':predict'

# Try to load class names saved by the training notebook
CLASS_NAMES = None
try:
    classes_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'potatoes_model', 'class_names.json')
    classes_path = os.path.abspath(classes_path)
    if os.path.exists(classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            CLASS_NAMES = json.load(f)
        logger.info('Loaded class names from %s: %s', classes_path, CLASS_NAMES)
except Exception as e:
    logger.warning('Could not load class_names.json: %s', e)

if CLASS_NAMES is None:
    CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    """Open image bytes, resize to 256x256 and return float32 numpy array.
    We keep raw pixel values (0..255) because the exported SavedModel includes a Rescaling layer.
    """
    try:
        img = Image.open(BytesIO(data)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Unable to read image file: {e}')
    img = img.resize((256, 256))
    image = np.array(img).astype(np.float32)
    return image.tolist()


@app.get('/ping')
async def ping():
    """Health check: verifies TF Serving reachable and model names loaded."""
    # Check TF Serving model endpoint
    try:
        status_resp = requests.get(TF_SERVING_BASE, timeout=2)
        ts = status_resp.status_code
        ok = (ts == 200)
    except Exception as e:
        ok = False
        status_resp = None
    return {
        'api': 'ok',
        'tf_serving_reachable': ok,
        'tf_serving_status_code': status_resp.status_code if status_resp is not None else None,
        'class_names_count': len(CLASS_NAMES)
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Basic content-type check
    if not file.content_type or not file.content_type.startswith('image'):
        raise HTTPException(status_code=400, detail='Uploaded file is not an image')

    image = read_file_as_image(await file.read())
    payload = {'instances': [image]}

    try:
        resp = requests.post(TF_SERVING_URL, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error('Error calling TF Serving: %s', e)
        raise HTTPException(status_code=502, detail=f'Could not reach TF Serving: {e}')

    if resp.status_code != 200:
        logger.error('TF Serving returned %s: %s', resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail=f'TF Serving returned {resp.status_code}: {resp.text}')

    data = resp.json()
    predictions = data.get('predictions')
    if not predictions:
        logger.error('No predictions in TF Serving response: %s', data)
        raise HTTPException(status_code=502, detail='No predictions in TF Serving response')

    preds = np.array(predictions)
    # safety check shape
    if preds.ndim == 1:
        probs = preds
    else:
        probs = preds[0]

    idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    confidence = float(np.max(probs))
    return {'class': predicted_class, 'confidence': confidence}


if __name__ == '__main__':
    import uvicorn
    logger.info('Starting TF-Serving proxy API on http://localhost:8000, TF Serving URL=%s', TF_SERVING_URL)
    uvicorn.run(app, host='localhost', port=8000, reload=False)
