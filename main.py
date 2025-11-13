from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    # Load the small model (fits Railway free plan)
    model = whisper.load_model("small")

    # Transcribe the file
    result = model.transcribe(temp_path)

    # Remove temp file
    os.remove(temp_path)

    return {"text": result["text"]}
