import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketDisconnect

load_dotenv()


class ServerConfig:

    # Faster Whisper Model Configuration
    WHISPER_MODEL_SIZE: str = "large-v3"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"

    SAMPLE_RATE: int = 16000  # get from config.yaml
    HPF_CUTOFF_FREQ_HZ: int = 295  # get from config.yaml
    HPF_ORDER: int = 2  # get from config.yaml
    NORMALIZATION_TARGET_PEAK: float = 0.95  # get from config.yaml


class ModelManager:
    """Manages the lifecycle and access to all AI models."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.whisper_model: Optional[WhisperModel] = None
        self.b_hpf, self.a_hpf = None, None
        self.stt_language: Optional[str] = None

    async def initialize(self):
        """Loads all models and designs the audio filter."""

        self._load_whisper_model()
        self._design_hpf()

    async def warmup(self):
        """Performs warm-up inferences for all models."""
        if self.is_audio_processing_ready():
            self._warmup_audio_models()

    def is_initialized(self) -> bool:
        """Checks if all essential models and components are ready."""
        return all([self.whisper_model, self.b_hpf is not None, self.a_hpf is not None])

    def is_audio_processing_ready(self) -> bool:
        """Checks if Whisper and filter are ready."""
        return all([self.whisper_model, self.b_hpf is not None, self.a_hpf is not None])

    def _load_whisper_model(self):
        print(f"Loading Faster Whisper model '{self.config.WHISPER_MODEL_SIZE}'...")
        try:
            self.whisper_model = WhisperModel(
                self.config.WHISPER_MODEL_SIZE,
                device=self.config.WHISPER_DEVICE,
                compute_type=self.config.WHISPER_COMPUTE_TYPE,
            )
            print("Faster Whisper model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Faster Whisper model: {e}")
            self.whisper_model = None

    def _design_hpf(self):
        print("Designing High-Pass Filter...")
        try:
            nyquist = 0.5 * self.config.SAMPLE_RATE
            normal_cutoff = self.config.HPF_CUTOFF_FREQ_HZ / nyquist
            self.b_hpf, self.a_hpf = scipy.signal.butter(
                self.config.HPF_ORDER, normal_cutoff, btype="high", analog=False
            )
            print(
                f"Designed HPF: Cutoff={self.config.HPF_CUTOFF_FREQ_HZ} Hz, Order={self.config.HPF_ORDER}"
            )
        except Exception as e:
            print(f"Failed to design High-Pass Filter: {e}")
            self.b_hpf, self.a_hpf = None, None

    def _warmup_audio_models(self):
        print("Performing Faster Whisper, and Filter warm-up...")
        warmup_audio_path = "./voice.wav"
        self._create_dummy_voice_wav(warmup_audio_path, self.config.SAMPLE_RATE)
        warmup_audio_np_int16 = self._load_warmup_audio(warmup_audio_path)
        if warmup_audio_np_int16 is None:
            print("Skipping audio warm-up: No valid audio data or audio too short.")
            return

        processed_audio = self._process_audio_for_asr(warmup_audio_np_int16)
        segments, _ = self.whisper_model.transcribe(
            processed_audio, beam_size=5, language="en"
        )
        transcription = "".join([s.text for s in segments]).strip()
        print(
            f"Faster Whisper warm-up completed. Transcription: '{transcription[:50]}...'"
        )

    def _process_audio_for_asr(self, audio_data_int16: np.ndarray) -> np.ndarray:
        """Applies HPF and normalization to raw audio data."""
        audio_float32 = audio_data_int16.astype(np.float32)
        filtered_audio = scipy.signal.filtfilt(self.b_hpf, self.a_hpf, audio_float32)
        peak = np.max(np.abs(filtered_audio))
        if peak > 1e-6:
            return filtered_audio / peak * self.config.NORMALIZATION_TARGET_PEAK
        return filtered_audio

    async def transcribe_audio(self, audio_data_int16: np.ndarray) -> Optional[str]:
        """Processes and transcribes an audio segment."""
        try:
            processed_audio = await asyncio.to_thread(
                self._process_audio_for_asr, audio_data_int16
            )
            segments, info = await asyncio.to_thread(
                self.whisper_model.transcribe,
                processed_audio,
                beam_size=10,
                without_timestamps=True,
                vad_filter=False,
                language="en",
                hotwords="Fadi",
            )
            transcription = "".join([s.text for s in segments]).strip()
            self.stt_language = info.language if info.language else "unknown"
            print(f"Transcription: '{transcription}' (Lang: {self.stt_language})")
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def _create_dummy_voice_wav(
        self, file_path: str, sample_rate: int, duration_seconds: int = 2
    ):
        if os.path.exists(file_path):
            return
        print(f"Creating dummy warm-up audio '{file_path}'...")
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(file_path, (audio_data * 32767).astype(np.int16), sample_rate)

    def _load_warmup_audio(self, file_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(file_path):
            return None
        data, samplerate = sf.read(file_path, dtype="int16")
        if samplerate != self.config.SAMPLE_RATE:
            print(
                f"Warning: Warm-up audio SR mismatch. Expected {self.config.SAMPLE_RATE}Hz, got {samplerate}Hz."
            )
        return data[:, 0] if data.ndim > 1 else data


class WebSocketSession:
    """
    Manages the state and logic for a single client WebSocket connection.
    this for call orchestration
    """

    def __init__(
        self, websocket: WebSocket, model_manager: ModelManager, config: ServerConfig
    ):
        self.websocket = websocket
        self.model_manager = model_manager
        self.config = config
        self.current_response_item_id: Optional[str] = None

    async def transcribe_speech_segment(self, audio_buffer) -> Optional[str]:
        full_buffer_np = np.array(audio_buffer, dtype=np.int16)
        transcription = await self.model_manager.transcribe_audio(full_buffer_np)
        # for example: transcription = "When Fadi has a free time?"
        if transcription:
            await self.websocket.send_json(
                {
                    "type": "response.audio_transcript.done",
                    "transcript": transcription,
                    "item_id": str(uuid.uuid4()),
                }
            )


# Load server configuration from environment variables or defaults
config = ServerConfig()
model_manager = ModelManager(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    await model_manager.initialize()
    await model_manager.warmup()
    yield
    print("Server shutting down.")


app = FastAPI(lifespan=lifespan)


@app.websocket(f"/ws/stt")
async def openai_realtime_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"Client connected: {websocket.client}")

    if not model_manager.is_initialized():
        await websocket.send_json(
            {"type": "error", "message": "Server models not initialized."}
        )
        await websocket.close(code=1011)
        return

    try:
        while True:
            message = await websocket.receive_json()
            audio = message.get("audio")
            session = WebSocketSession(websocket, model_manager, config)
            session.config.SAMPLE_RATE = message.get("sample_rate", 16000)
            await session.transcribe_speech_segment(audio)
    except WebSocketDisconnect as e:
        print(
            f"Client disconnected: {websocket.client}, code={e.code}, reason={e.reason or 'none'}"
        )
    except Exception as e:
        print(f"Unexpected error in WebSocket: {e}")
        await websocket.send_json(
            {"type": "error", "message": f"Server error: {str(e)}"}
        )
        await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "5001")),
        log_level="info",
    )
