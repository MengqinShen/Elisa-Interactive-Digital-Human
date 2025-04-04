import os
import tempfile
import whisper
import edge_tts
import asyncio
import google.generativeai as genai
import gradio as gr
from scipy.io.wavfile import write
import sounddevice as sd
import subprocess
import keyboard
import time
import pyaudio

'''
gemini
ASR: openai-whisper
TTS:edge-tts
'''

# ======= CONFIG =======
GEMINI_API_KEY = "Your Key"  # <-- Put your key here
VOICE = "en-US-AriaNeural"
RECORD_SECONDS = 15
SAMPLE_RATE = 16000
# 设备初始化
P = pyaudio.PyAudio()

genai.configure(api_key=GEMINI_API_KEY)
whisper_model = whisper.load_model("tiny", device="cpu")

def record_audio(duration=RECORD_SECONDS):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recording complete.")
    return audio


def save_wav(audio):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result["text"]

def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

async def speak_text(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    # Generate speech with edge-tts
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(mp3_path)

    # Play the generated audio
    subprocess.run(["afplay", mp3_path])

    # Optional: remove the file after playing
    os.remove(mp3_path)

# Main Loop
async def main():
    while True:
        print("""
        ===== Gemini Multi-Conversation =====
        Space : start/stop
        ESC  : quit
        ======================
        """)
        audio = record_audio()
        audio_path = save_wav(audio)
        text = transcribe(audio_path)
        # text = transcribe("./data/sun.wav")
        os.remove(audio_path)

        print(f"📝 You said: {text}")
        reply = ask_gemini(text)
        print(f"🤖 Gemini: {reply}")

        await speak_text(reply)

        # 键盘事件检测
        if keyboard.is_pressed('space'):
            # self.toggle_recording()
            time.sleep(0.5)  # 防抖处理
        elif keyboard.is_pressed('esc'):
            P.terminate()
            break

        time.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())

# import pyaudio
# import wave
#
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# RECORD_SECONDS = 5
# OUTPUT_FILENAME = "output.wav"
#
# p = pyaudio.PyAudio()
#
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)
#
# print("🎙️ Recording...")
# frames = []
# for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# print("✅ Done recording.")
#
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# wf = wave.open(OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()
