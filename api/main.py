from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Helper: determine whether a Keras model contains a Rescaling layer
def model_needs_raw_input(model):
    """
    Return True if the model expects raw 0-255 pixel values (i.e. it contains a Rescaling layer
    that rescales to 0-1). We inspect model.layers for a Rescaling layer; if model is None or
    cannot be inspected we conservatively return False (so we normalize the input to 0-1).
    """
    try:
        if model is None:
            return False
        for layer in getattr(model, 'layers', [])[:8]:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
    except Exception:
        pass
    return False

# Load model: prefer SavedModel directory (TF Serving style) then fall back to H5
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAVEDMODEL_DIR = os.path.join(BASE_DIR, 'models', 'potatoes_model', '1')
H5_PATH = os.path.join(BASE_DIR, 'saved_models', 'model_final.h5')

MODEL = None
load_error = None
try:
    # Prefer H5 for direct Keras loading (Keras 3 supports .h5 and .keras formats).
    if os.path.exists(H5_PATH):
        MODEL = tf.keras.models.load_model(H5_PATH)
        print(f"Loaded Keras H5 from {H5_PATH}")
    elif os.path.isdir(SAVEDMODEL_DIR):
        # Keras 3 cannot load legacy SavedModel with load_model(); attempt to wrap with TFSMLayer
        try:
            MODEL = tf.keras.models.load_model(SAVEDMODEL_DIR)
            print(f"Loaded SavedModel from {SAVEDMODEL_DIR}")
        except Exception as e_inner:
            # Fallback: create a small Keras model that calls the SavedModel using TFSMLayer
            try:
                print('SavedModel exists but cannot be loaded directly by Keras. Loading with tf.saved_model.load() and wrapping...')
                sm = tf.saved_model.load(SAVEDMODEL_DIR)
                # choose a signature
                sig = None
                if hasattr(sm, 'signatures') and 'serve' in sm.signatures:
                    sig = sm.signatures['serve']
                elif hasattr(sm, 'signatures') and 'serving_default' in sm.signatures:
                    sig = sm.signatures['serving_default']
                else:
                    # try to get any available signature
                    try:
                        sig = list(sm.signatures.values())[0]
                    except Exception:
                        sig = None

                class SavedModelWrapper:
                    def __init__(self, module, signature):
                        self.module = module
                        self.signature = signature

                    def predict(self, np_batch):
                        # convert to tensor
                        t = tf.convert_to_tensor(np_batch, dtype=tf.float32)
                        if self.signature is not None:
                            out = self.signature(t)
                            # signature may return dict of outputs
                            if isinstance(out, dict):
                                # pick the first tensor value
                                first = list(out.values())[0]
                                return first.numpy()
                            else:
                                return out.numpy()
                        else:
                            # try calling the module directly
                            out = self.module(t)
                            if isinstance(out, dict):
                                return list(out.values())[0].numpy()
                            return out.numpy()

                MODEL = SavedModelWrapper(sm, sig)
                # The training notebook included a Rescaling layer (1./255) inside the model.
                # SavedModel therefore expects raw 0-255 input; set the flag so we do not normalize again.
                _MODEL_ACCEPTS_RAW = True
                print('Wrapped SavedModel for inference:', SAVEDMODEL_DIR)
            except Exception as e_wrap:
                print('Failed to wrap SavedModel:', e_wrap)
                raise e_inner
    else:
        raise FileNotFoundError(f"No model found at {SAVEDMODEL_DIR} or {H5_PATH}")
except Exception as e:
    MODEL = None
    load_error = e

# Determine whether loaded model expects raw 0-255 input or already rescales internally.
# We compute this after loading the model because SavedModelWrapper is not a Keras Model.
try:
    if MODEL is None:
        _MODEL_ACCEPTS_RAW = False
    else:
        # Keras Model: inspect layers for Rescaling
        if isinstance(MODEL, tf.keras.Model):
            _MODEL_ACCEPTS_RAW = model_needs_raw_input(MODEL)
        else:
            # If we wrapped a SavedModel (SavedModelWrapper), assume it expects raw 0-255
            # because training notebook included a Rescaling layer inside the exported model.
            try:
                if MODEL.__class__.__name__ == 'SavedModelWrapper':
                    _MODEL_ACCEPTS_RAW = True
                else:
                    _MODEL_ACCEPTS_RAW = False
            except Exception:
                _MODEL_ACCEPTS_RAW = False
except Exception:
    _MODEL_ACCEPTS_RAW = False

# CLASS_NAMES must match the training dataset class ordering
# Try to load class names saved by the training notebook. If not present, fall back to a reasonable default.
CLASS_NAMES = None
try:
    classes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'potatoes_model', 'class_names.json'))
    if os.path.exists(classes_path):
        import json
        with open(classes_path, 'r', encoding='utf-8') as f:
            CLASS_NAMES = json.load(f)
        print(f"Loaded class names from {classes_path}: {CLASS_NAMES}")
except Exception as e:
    print('Could not load class_names.json:', e)

if CLASS_NAMES is None:
    # fallback label names (user-friendly). If your training used a different order, please update class_names.json
    CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# _MODEL_ACCEPTS_RAW is computed after model loading further below

@app.get("/ping")
async def ping():
    return {"Hello, I am alive!"}

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB')
    # resize to the model input size used during training (256x256)
    img = img.resize((256, 256))
    image = np.array(img).astype(np.float32)
    # If model does not include an internal Rescaling layer, normalize to 0-1 here
    if not _MODEL_ACCEPTS_RAW:
        image = image / 255.0
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Create a batch
    if MODEL is None:
        return {"error": "model failed to load", "detail": str(load_error)}
    # If we wrapped a SavedModel with TFSMLayer, the wrapper may expect float32 input in 0-255 range
    try:
        predictions = MODEL.predict(img_batch)
    except Exception as e_predict:
        # Try converting to 0-255 ints or to 0-1 depending on model expectations
        try:
            # second attempt: normalize to 0-1 if not already
            predictions = MODEL.predict(img_batch / 255.0)
        except Exception:
            # as last resort, raise original error
            raise e_predict
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)