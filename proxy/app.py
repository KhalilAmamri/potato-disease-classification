from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client, handle_file
import os
import logging

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
@app.post("/predict")
@app.post("/api/predict")
async def predict_url(payload: ImgUrl):
    try:
        logger.info("Forwarding request to Gradio Space: %s", payload.url)
        res = client.predict(image=handle_file(payload.url), api_name="/predict")
        return res
    except Exception as e:
        logger.exception("Error calling gradio client")
        raise HTTPException(status_code=502, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8081, log_level="info")
