import asyncio
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv

load_dotenv()
# Load YAML configuration

# WebSocket server URL
WEBSOCKET_URL = (
    f"ws://{os.getenv('HOST', 'localhost')}:{int(os.getenv('PORT', 5001))}/ws/stt"
)


# Function to generate dummy audio data (sine wave)
def generate_dummy_audio(sample_rate=16000, duration_seconds=2):
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    return (audio_data * 32767).astype(np.int16)


async def send_audio_message():
    # Generate dummy audio data
    sample_rate = 16000
    # audio_data = generate_dummy_audio(sample_rate=sample_rate)
    audio_path = Path(__file__).parent / "test.wav"
    audio_data, sample_rate = sf.read(audio_path, dtype="int16")
    try:
        audio_data, sample_rate = sf.read(audio_path, dtype="int16")
        if sample_rate != 16000:
            print(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
            # Convert to float32 for resampling
            audio_float = audio_data.astype(np.float32) / 32767.0
            # Resample to 16000 Hz
            audio_float_resampled = librosa.resample(
                audio_float, orig_sr=sample_rate, target_sr=16000
            )
            # Convert back to int16
            audio_data = (audio_float_resampled * 32767.0).astype(np.int16)
            sample_rate = 16000  # Update sample rate
        if audio_data.ndim > 1:
            print("Converting stereo audio to mono (using first channel)")
            audio_data = audio_data[:, 0]  # Use first channel for stereo audio
    except Exception as e:
        print(f"Failed to load audio file '{audio_path}': {e}")
        return
    # Connect to the WebSocket server
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        print("Connected to WebSocket server")

        # Prepare the message with audio data
        message = {
            "type": "audio.append",
            "sample_rate": sample_rate,
            "audio": audio_data.tolist(),  # Convert numpy array to list for JSON serialization
        }

        # Send the audio data to the server
        await websocket.send(json.dumps(message))
        print("Sent audio message to server")

        # Receive and print the response
        response = await websocket.recv()
        response_data = json.loads(response)
        print(f"Received response: {response_data}")

        # Validate the response
        if response_data.get("type") == "response.audio_transcript.done":
            print(f"Transcription: {response_data.get('transcript')}")
            print(f"Item ID: {response_data.get('item_id')}")
        elif response_data.get("type") == "error":
            print(f"Error from server: {response_data.get('message')}")


# Run the client
if __name__ == "__main__":
    asyncio.run(send_audio_message())
