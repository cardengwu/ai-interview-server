from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# Load model at startup (tiny 模型最適合 Railway)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # transcribe
    segments, info = model.transcribe(audio_path)

    # collect text
    text = "".join([segment.text for segment in segments])

    os.remove(audio_path)
    return {"text": text}
