from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client, handle_file
import os
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If your Space is private, set HF_TOKEN env var
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    client = Client("Khalil-Amamri/potato-space", hf_token=hf_token)
else:
    client = Client("Khalil-Amamri/potato-space")

class ImgUrl(BaseModel):
    url: str

@app.get("/health")
async def health():
    return {"status": "ok"}


# Accept both /predict and /api/predict for compatibility with mobile/web clients
# This version accepts file uploads (multipart/form-data)
@app.post("/predict")
@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        logger.info("Received file upload: %s", file.filename)
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info("Forwarding file to Gradio Space: %s", tmp_path)
        res = client.predict(image=handle_file(tmp_path), api_name="/predict")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return res
    except Exception as e:
        logger.exception("Error calling gradio client")
        raise HTTPException(status_code=502, detail=str(e))

# Keep URL endpoint for testing with public URLs
@app.post("/predict_url")
async def predict_url(payload: ImgUrl):
    try:
        logger.info("Forwarding URL to Gradio Space: %s", payload.url)
        res = client.predict(image=handle_file(payload.url), api_name="/predict")
        return res
    except Exception as e:
        logger.exception("Error calling gradio client")
        raise HTTPException(status_code=502, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8081, log_level="info")
