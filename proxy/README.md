# Proxy for Hugging Face Space

This small FastAPI proxy forwards image-URL requests to the Gradio Space using `gradio_client`.

Run locally (use the repository venv):

```powershell
Set-Location 'C:\Learn Programming\Machine Learning\potato_disease\proxy'
& 'C:/Learn Programming/Machine Learning/potato_disease/myenv/Scripts/python.exe' -m uvicorn app:app --host 0.0.0.0 --port 8081
```

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8081/health -Method GET
```

Predict example (public image URL):

```powershell
$body = @{ url = "https://upload.wikimedia.org/wikipedia/commons/3/39/Healthy_potato_leaf.jpg" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8081/predict -Method POST -Body $body -ContentType 'application/json'
```

When using the Android emulator, set the Flutter `HF_RUNTIME` to `http://10.0.2.2:8081` so the emulator can reach your host machine.
