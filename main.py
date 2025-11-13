from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.get("/")
async def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Read uploaded audio file
    audio_bytes = await file.read()

    # Call OpenAI Whisper API
    response = client.audio.transcriptions.create(
        file=(file.filename, audio_bytes),
        model="gpt-4o-transcribe"
    )

    return {"text": response.text}
