from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# -------------------------
# Load model ONCE at startup
# -------------------------
model_size = "tiny"   # tiny = 75MB, works on Render free tier
model = WhisperModel(model_size, device="cpu", compute_type="int8")


@app.get("/")
def root():
    return {"status": "server OK", "message": "AI Transcription ready."}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Save temporary audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # -------------------------
    # Transcribe with faster-whisper
    # -------------------------
    segments, info = model.transcribe(audio_path, beam_size=1)

    # Concatenate text
    text_result = "".join([seg.text for seg in segments])

    os.remove(audio_path)

    return {
        "text": text_result,
        "lang": info.language,
        "duration": info.duration
    }
