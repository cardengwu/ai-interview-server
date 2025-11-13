from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# CORS for frontend usage (very important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # Load tiny model (fastest + smallest)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # Transcribe audio
    segments, info = model.transcribe(audio_path)

    # Combine text segments
    text = "".join([seg.text for seg in segments])

    # Clean up
    os.remove(audio_path)

    return {"text": text}
