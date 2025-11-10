import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import redis.asyncio as redis
import scipy.signal
from agent_architect.datatype_abstraction import AudioFeatures, Features, TextFeatures
from agent_architect.models_abstraction import (
    AbstractAsyncModelInference,
    AbstractInferenceServer,
    AbstractQueueManagerServer,
    DynamicBatchManager,
)
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import go_next_service
from fastapi import FastAPI
from faster_whisper import BatchedInferencePipeline, WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerConfig:
    HPF_CUTOFF_FREQ_HZ: int = 295  # get from config.yaml
    HPF_ORDER: int = 2  # get from config.yaml
    NORMALIZATION_TARGET_PEAK: float = 0.95  # get from config.yaml
    SAMPLE_RATE: int = 16000  # get from config.yaml


class WhisperAsyncBatchInference(AbstractAsyncModelInference):
    """
    Asynchronous Whisper inference with dynamic batching using faster-whisper's BatchedInferencePipeline.
    Accepts VAD speech segments as base64-encoded int16 PCM at 16kHz.
    """

    def __init__(
        self, model_name: str = "large-v3", max_worker: int = 4, config=ServerConfig
    ):
        super().__init__(max_worker=max_worker)
        print("Loading Whisper model...")
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        # whisper_model = WhisperModel(model_name, device="cuda", compute_type="float16")
        # self.model = BatchedInferencePipeline(model=whisper_model)
        self.config = config
        self.b_hpf, self.a_hpf = None, None
        self._design_hpf()
        print("Whisper model loaded successfully.")

    def _design_hpf(self):
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

    def _process_audio_for_asr(self, audio_data_int16: np.ndarray) -> np.ndarray:
        """Applies HPF and normalization to raw audio data."""
        audio_float32 = audio_data_int16.astype(np.float32)
        filtered_audio = scipy.signal.filtfilt(self.b_hpf, self.a_hpf, audio_float32)
        peak = np.max(np.abs(filtered_audio))
        if peak > 1e-6:
            return filtered_audio / peak * self.config.NORMALIZATION_TARGET_PEAK
        return filtered_audio

    async def process_batch(self, batch: List[AudioFeatures]) -> List[TextFeatures]:
        """Process a batch of requests asynchronously."""
        try:
            loop = __import__("asyncio").get_event_loop()
            batch_outputs = await loop.run_in_executor(
                self.thread_pool, self._run_model_inference, batch
            )
            return batch_outputs
        except Exception as e:
            return await self._handle_batch_error(batch, e)

    async def _run_model_inference(
        self, prepared_inputs: List[AudioFeatures]
    ) -> List[TextFeatures]:
        results = []
        for audio_object in prepared_inputs:
            audio_object.sample_rate
            if audio_object.sample_rate != 16000:
                raise ValueError(
                    f"Request {audio_object.sid}: Expected 16kHz audio, got {audio_object.sample_rate}"
                )
            audio_bytes = base64.b64decode(audio_object.audio)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            processed_audio = await asyncio.to_thread(
                self._process_audio_for_asr, audio_int16
            )
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                processed_audio,
                beam_size=10,
                without_timestamps=True,
                vad_filter=False,
                language="en",
                hotwords="Fadi",
            )

            output_text = "".join([s.text for s in segments]).strip()
            print(f"Transcription info: {output_text}")
            output_text = output_text.replace("NOTHING", "")
            print(f"Result for {audio_object.sid}: {output_text}")
            if len(output_text):
                results.append(
                    TextFeatures(
                        sid=audio_object.sid,
                        agent_type=audio_object.agent_type,
                        text=output_text,
                        priority=audio_object.priority,
                        created_at=None,
                        is_final=audio_object.is_final,
                    )
                )
        return results


class RedisQueueManager(AbstractQueueManagerServer):
    """
    Manages Redis-based async queue for inference requests
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        service_name: str = "STT",
        priorities: List[str] = ["high", "low"],
    ):
        self.redis_url = redis_url
        self.priorities = priorities
        self.service_name = service_name
        self.redis_client = None
        self.pubsub = None
        self.active_sessions_key = f"active_sessions"
        self.input_channels: List = [
            f"{self.service_name}:high",
            f"{self.service_name}:low",
        ]

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)

    async def get_status_object(self, req: Features) -> AgentSessions:
        raw = await self.redis_client.hget(
            f"{req.agent_type}:{self.active_sessions_key}", req.sid
        )
        if raw is None:
            return None
        return AgentSessions.from_json(raw)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(req)
        if status_obj is None:
            return False
        # change status of the session to 'stop' if the session expired
        print(f"🎃 Session {req.sid} status: {status_obj.status}")
        if status_obj.is_expired():
            print(f"🤖 Session {req.sid} status: {status_obj.status}")
            status_obj.status = SessionStatus.STOP
            await self.redis_client.hset(
                f"{req.agent_type}:{self.active_sessions_key}",
                req.sid,
                status_obj.to_json(),
            )
            return False
        elif status_obj.status == SessionStatus.INTERRUPT:
            print(f"🎉 Session {req.sid} status: {status_obj.status}")
            return False
        return True

    async def get_data_batch(
        self, max_batch_size: int = 8, max_wait_time: float = 0.1
    ) -> List[AudioFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break

            for input_channel in self.input_channels:
                result = await self.redis_client.brpop(input_channel, timeout=0.01)
                if result:
                    break

            if result:
                _, request_json = result
                try:
                    req = AudioFeatures.from_json(request_json)
                    if not await self.is_session_active(req):
                        logger.info(f"Skipped request for stopped session: {req.sid}")
                        continue
                    batch.append(req)
                except Exception as e:
                    logger.error(f"Error in get_data_batch: {e}", exc_info=True)
            else:
                await asyncio.sleep(0.01)

        return batch

    async def push_result(self, result: TextFeatures):
        """Push inference result back to Redis pub/sub"""
        if not await self.is_session_active(result):
            logger.info(f"Not pushing result for inactive session: {result.sid}")
            return
        status_obj = await self.get_status_object(result)
        await self.redis_client.hset(
            f"{result.agent_type}:{self.active_sessions_key}",
            result.sid,
            status_obj.to_json(),
        )
        # calculate next service and queue name
        next_service = go_next_service(
            current_stage_name=self.service_name,
            service_names=status_obj.service_names,
            channels_steps=status_obj.channels_steps,
            last_channel=status_obj.last_channel,
            prioriry=result.priority,
        )
        await self.redis_client.lpush(next_service, result.to_json())
        logger.info(f"Result pushed for request {result.sid}, to {next_service}")


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        service_name: str = "STT",
    ):
        super().__init__()
        self.service_name = service_name
        self.queue_manager = RedisQueueManager(
            redis_url, service_name=self.service_name
        )
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = WhisperAsyncBatchInference(max_worker=max_worker)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(req)

    async def _initialize_components(self):
        await self.queue_manager.initialize()

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_batches_loop())

    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        while self.is_running:
            try:
                batch = await self.queue_manager.get_data_batch(
                    max_batch_size=self.batch_manager.max_batch_size,
                    max_wait_time=self.batch_manager.max_wait_time,
                )
                if batch:
                    start_time = time.time()
                    batch_results = await self.inference_engine._run_model_inference(
                        batch
                    )
                    processing_time = time.time() - start_time
                    for result_object in batch_results:
                        # update create_at time of the session
                        await self.queue_manager.push_result(result=result_object)

                    self.batch_manager.update_metrics(len(batch), processing_time)
                    logger.info(
                        f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                    )
                else:
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, service
    logging.info("Application startup: Initializing LLM Manager...")

    service = InferenceService()
    await service.start()
    logging.info("InferenceService started.")

    yield
    # Shutdown logic
    if service:
        service.is_running = False
        if service.processing_task:
            service.processing_task.cancel()
            try:
                await service.processing_task
            except asyncio.CancelledError:
                logging.info("Processing loop cancelled.")
        logging.info("InferenceService stopped.")
    logging.info("Application shutdown...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "LLM RAG server is running."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8102)
