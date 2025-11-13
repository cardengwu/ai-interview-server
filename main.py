from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper_timestamped as whisper
import tempfile
import os

app = FastAPI()

# --- 新增 CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允許任何前端來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio_path)

    os.remove(audio_path)
    return {"text": result["text"]}
