from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # save temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # load tiny model (smallest, works on Railway)
    model = WhisperModel("tiny", device="cpu")
    segments, info = model.transcribe(audio_path)

    # join transcript
    text = " ".join([seg.text for seg in segments])

    os.remove(audio_path)

    return {"text": text}
