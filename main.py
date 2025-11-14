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

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # Load the tiny model (fastest and smallest)
    model = whisper.load_model("tiny", device="cpu")

    # Run transcription
    result = whisper.transcribe(model, audio_path)

    # Cleanup
    os.remove(audio_path)

    return {"text": result["text"]}
