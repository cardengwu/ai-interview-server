from fastapi import FastAPI, UploadFile, File
import whisper_timestamped as whisper
import tempfile
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # save uploaded file temporary
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # load tiny model (fastest, works on Railway)
    model = whisper.load_model("tiny", device="cpu")

    # transcribe audio
    result = whisper.transcribe(model, audio_path)

    os.remove(audio_path)

    # result["text"] contains transcription
    return {"text": result["text"]}
