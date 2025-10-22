import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_VAD.models.abstract_models import *


import base64
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
import base64
import logging
import numpy as np
import time
import redis.asyncio as redis
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dataclasses import dataclass

@dataclass
class SttFeatures(Features):
    payload: Dict[str, Any]


import base64
import numpy as np
import time
from typing import Any, Dict, List
from faster_whisper import WhisperModel, BatchedInferencePipeline



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  
call_channels = ChannelNames(input_channel=["VAD:high", "VAD:low"], output_channel=["STT:high", "STT:low"])
    
    
class WhisperAsyncBatchInference(AbstractAsyncModelInference):
    """
    Asynchronous Whisper inference with dynamic batching using faster-whisper's BatchedInferencePipeline.
    Accepts VAD speech segments as base64-encoded int16 PCM at 16kHz.
    """

    def __init__(self, model_name: str = "large-v3", max_worker: int = 4):
        super().__init__(max_worker=max_worker)
        print('Loading Whisper model...')
        whisper_model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self.model = BatchedInferencePipeline(model=whisper_model)
        print('Whisper model loaded successfully.')

    async def process_batch(self, batch: List[AudioFeatures]) -> List[TextFeatures]:
        """Process a batch of requests asynchronously."""
        try:
            loop = __import__('asyncio').get_event_loop()
            batch_outputs = await loop.run_in_executor(
                self.thread_pool,
                self._run_model_inference,
                batch
            )
            return batch_outputs
        except Exception as e:
            return await self._handle_batch_error(batch, e)

    def _run_model_inference(self, prepared_inputs: List[AudioFeatures]) -> List[TextFeatures]:
        results = []
        for audio_object in prepared_inputs:
            audio_object.sample_rate
            if audio_object.sample_rate != 16000:
                raise ValueError(f"Request {audio_object.sid}: Expected 16kHz audio, got {audio_object.sample_rate}")

            audio_bytes = base64.b64decode(audio_object.audio)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            segments, info = self.model.transcribe(
                audio=audio_float32,
                beam_size=5,
                language='en')
            output_text = "".join([s.text for s in segments]).strip()
            if len(output_text):
                results.append(TextFeatures(sid=audio_object.sid, text=output_text, priority=audio_object.priority, created_at=None, timeout=None))
        return results
                
class RedisQueueManager(AbstractQueueManagerServer):
    """
    Manages Redis-based async queue for inference requests
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "call_agent"):
        self.redis_url = redis_url
        self.channels_name = call_channels.get_all_channels()
        self.redis_client = None
        self.pubsub = None
        self.queue_name = queue_name
        self.active_sessions_key = f"{queue_name}:active_sessions"
        self.priorities = get_high_low(call_channels.output_channel)
        self.in_priorities = get_high_low(call_channels.input_channel)

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
        self.pubsub = self.redis_client.pubsub()
        for channel_name in self.channels_name:
            await self.pubsub.subscribe(channel_name)
        logger.info(f"Redis queue manager initialized for queue: {self.queue_name}")
        
    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        raw = await self.redis_client.hget(self.active_sessions_key, req.sid)
        if raw is None:
            return False
        status_obj = SessionStatus.from_json(raw)
        # change status of the session to 'stop' if the session expired
        if status_obj.is_exired():
            status_obj.status = "stop"
            await self.redis_client.hset(self.active_sessions_key, req.sid, status_obj.to_json())
            return False
        
        # update create_at time of the session
        status_obj.refresh_time()
        await self.redis_client.hset(self.active_sessions_key, req.sid, status_obj.to_json())
        return True

    async def get_data_batch(self, max_batch_size: int = 8, max_wait_time: float = 0.1) -> List[AudioFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break
            
            result = await self.redis_client.brpop(self.in_priorities["high"], timeout=0.01)
            if not result:
                result = await self.redis_client.brpop(self.in_priorities["low"], timeout=0.01)
            
            if result:
                _, request_json = result
                try:
                    req = AudioFeatures.from_json(request_json)
                    if not await self.is_session_active(req):
                        logger.info(f"Skipped request for stopped session: {req.sid}")
                        continue
                    req.priority = transform_priority_name(self.priorities, req.priority)
                    batch.append(req)
                except Exception as e:
                    logger.error(f"Error in get_data_batch: {e}", exc_info=True)
            else:
                await asyncio.sleep(0.01)

        return batch
    
    async def push_result(self, result: TextFeatures, channel_name:str, error: str = None):
        """Push inference result back to Redis pub/sub"""
        await self.redis_client.lpush(channel_name, result.to_json())
        logger.info(f"Result pushed for request {result.sid}")
   
   
class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        queue_name: str = "call_agent"
    ):
        super().__init__()
        self.queue_manager = RedisQueueManager(redis_url, queue_name=queue_name)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = WhisperAsyncBatchInference(max_worker=max_worker)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(req)
    
    async def _initialize_components(self):
        await self.queue_manager.initialize()
        # await self.inference_engine.model_manager.initialize()
    
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
                    max_wait_time=self.batch_manager.max_wait_time
                )
                if batch:
                    start_time = time.time()
                    batch_results = await self.inference_engine.process_batch(batch)
                    processing_time = time.time() - start_time
                    for result_object in batch_results:
                        if self.is_session_active(result_object):
                            await self.queue_manager.push_result(result=result_object, channel_name=result_object.priority)
                        
                    self.batch_manager.update_metrics(len(batch), processing_time)
                    logger.info(f"Processed batch of {len(batch)} requests in {processing_time:.3f}s")
                else:
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)

  