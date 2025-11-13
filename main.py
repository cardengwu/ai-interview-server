import os
import base64
import json
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

# CORS: Allow desktop APP & mobile sync
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 1. Whisper Transcription (Groq Whisper)
# ================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

async def groq_whisper_transcribe(audio_bytes):
    """Send audio bytes to Groq Whisper"""
    async with httpx.AsyncClient(timeout=20) as client:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "whisper-large-v3-turbo", "response_format": "json"}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

        resp = await client.post(GROQ_WHISPER_URL, files=files, data=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["text"]


@app.post("/transcribe")
async def transcribe_api(file: UploadFile):
    audio_bytes = await file.read()
    text = await groq_whisper_transcribe(audio_bytes)
    return {"text": text}


# ================================
# 2. LLM Auto-Selector (GPT / DeepSeek / Groq)
# ================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"

async def llm_generate(prompt):
    """
    Auto-selection:
    1. DeepSeek (free → cheap → fast)
    2. Groq Llama 3.1 (super fast)
    3. OpenAI GPT-4o (high accuracy)
    """

    async with httpx.AsyncClient(timeout=20) as client:

        # 1. Try DeepSeek
        if DEEPSEEK_API_KEY:
            try:
                resp = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except:
                pass

        # 2. Try Groq Llama3.1
        if GROQ_API_KEY:
            try:
                resp = await client.post(
                    GROQ_LLM_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.1-70b-versatile", "messages": [{"role": "user", "content": prompt}]}
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except:
                pass

        # 3. Fallback → OpenAI GPT-4o
        if OPENAI_API_KEY:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
            )
            return resp.json()["choices"][0]["message"]["content"]

    return "LLM error."


@app.post("/generate")
async def generate_api(prompt: str = Form(...)):
    reply = await llm_generate(prompt)
    return {"reply": reply}


# ================================
# 3. Real-Time WebSocket
# ================================
@app.websocket("/stream")
async def stream_ws(ws: WebSocket):
    await ws.accept()

    print("Client connected to WebSocket.")

    try:
        while True:
            # Receive audio chunk (base64)
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg["type"] == "audio":
                audio_bytes = base64.b64decode(msg["data"])

                # Transcribe
                text = await groq_whisper_transcribe(audio_bytes)

                # Generate suggestion from LLM
                reply = await llm_generate(text)

                # Send back result
                await ws.send_text(json.dumps({
                    "type": "result",
                    "asr": text,
                    "ai": reply
                }))

    except Exception as e:
        print("WebSocket closed:", e)


# ================================
# Root test
# ================================
@app.get("/")
def root():
    return {"status": "server OK", "message": "Real-Time AI active."}
