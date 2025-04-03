#git clone https://github.com/petewarden/openai-whisper-webapp
from fastapi import FastAPI, File, UploadFile
import whisper
import edge_tts
import asyncio
import os
from fastapi.responses import FileResponse
import google.generativeai as genai
import gradio as gr
import requests

# Initialize FastAPI
app = FastAPI()

# Configure Gemini API
GEMINI_API_KEY = "YOUR KEY"
genai.configure(api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel("gemini-1.5-flash")

# Load Whisper model (choose a smaller model for speed)
asr_model = whisper.load_model("base")
# Folder for saving temporary files
os.makedirs("temp", exist_ok=True)

@app.post("/convert/")
async def convert_speech_to_speech(file: UploadFile = File(...)):
    # Save uploaded audio file
    audio_path = f"temp/{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Convert speech to text using Whisper
    result = asr_model.transcribe(audio_path)
    transcribed_text = result["text"]

    # Send text to Gemini and get response
    response = genai.generate_text(genai_model, prompt=transcribed_text)
    ai_response = response.text  # Extract text response

    # Convert AI response to speech using edge-tts
    tts_output = f"temp/output.mp3"
    tts = edge_tts.Communicate(ai_response, "en-US-JennyNeural")
    await tts.save(tts_output)

    # Return generated speech file
    return FileResponse(tts_output, media_type="audio/mpeg", filename="response.mp3")

def process_audio(file):
    response = requests.post("http://127.0.0.1:8000/convert/", files={"file": open(file, "rb")})
    output_path = "response.mp3"  # Save response file locally
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path  # Return the file path

gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Audio(type="filepath"),  # Fix: Use 'filepath' instead of 'file'
    title="AI Voice Assistant"
).launch()
# def transcribe(audio):
#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)
#
#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#
#     # decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#     return result.text
