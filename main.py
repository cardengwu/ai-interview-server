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
    # save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    # load tiny model
    model = whisper.load_model("tiny", device="cpu")

    # transcribe
    result = whisper.transcribe(model, audio_path)

    # clean up
    os.remove(audio_path)

    return {"text": result["text"]}
