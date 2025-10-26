#!/usr/bin/env python3
import os
import asyncio
import base64
import logging
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
import time
import redis.asyncio as redis
from pydub import AudioSegment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Adjust this import to match your project structure
# You may need to add your project root to sys.path
from agent_architect.models_abstraction import AbstractAsyncModelInference, AbstractQueueManagerServer, AbstractInferenceServer, DynamicBatchManager
from agent_architect.datatype_abstraction import AudioFeatures, Features
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import go_next_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis channels (must match your service)
INPUT_CHANNELS = ["VAD:high", "VAD:low"]
OUTPUT_CHANNELS = ["STT:high", "STT:low"]


SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", 16000))
AGENT_NAME = "call"
SERVICE_NAMES = ["VAD","STT","RAG","TTS"]
CHANNEL_STEPS = {"VAD":["input"],"STT":["high", "low"], "RAG":["high", "low"],"TTS":["high","low"]}
INPUT_CHANNEL =f"{SERVICE_NAMES[0]}:{CHANNEL_STEPS[SERVICE_NAMES[0]][0]}"
OUTPUT_CHANNEL = f"{AGENT_NAME.lower()}:output"



# Session timeout key (must match your service)
ACTIVE_SESSIONS_KEY = f"{AGENT_NAME}:active_sessions"


async def load_wav_as_int16_base64(file_path: Path) -> str:
    """Load a WAV file and return base64-encoded int16 PCM at 16kHz."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        audio = await loop.run_in_executor(pool, AudioSegment.from_wav, str(file_path))
        # Ensure mono and 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)  # 16-bit
        raw_bytes = audio.raw_data  # this is int16 PCM
        b64 = base64.b64encode(raw_bytes).decode('utf-8')
        return b64


async def publish_requests(redis_client, wav_files: List[Path], num_sessions: int = 5):
    """Publish simulated VAD segments to Redis input channels."""
    sids = [f"test_sid_{i}" for i in range(num_sessions)]

    # Mark sessions as active
    for sid in sids:
        agent_session = AgentSessions(sid=sid,
                      owner_id="",
                      kb_id=[],
                      kb_limit=5,
                      agent_name=AGENT_NAME, 
                      service_names=SERVICE_NAMES,
                      channels_steps=CHANNEL_STEPS,
                      status=SessionStatus.ACTIVE,
                      first_channel=INPUT_CHANNEL,
                      last_channel=OUTPUT_CHANNEL,
                      timeout=3000)
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, agent_session.to_json())

    tasks = []
    for idx, wav_file in enumerate(wav_files):
        sid = sids[idx % len(sids)]
        priority = "high" if idx % 2 == 0 else "low"
        audio_b64 = await load_wav_as_int16_base64(wav_file)
        
        audio_feat = AudioFeatures(
            sid=sid,
            agent_name=AGENT_NAME,
            audio=audio_b64,
            sample_rate=16000,
            priority= f"{priority}",
            created_at=None
        )
        channel = f"STT:{priority}"
        logger.info(f"Publishing {wav_file.name} (sid={sid}, priority={priority}) to {channel}, create_at {audio_feat.created_at}")
        tasks.append(redis_client.lpush(channel, audio_feat.to_json()))

    await asyncio.gather(*tasks)
    logger.info(f"Published {len(wav_files)} requests.")


async def listen_for_results(redis_client, expected_count: int, timeout: int = 60):
    """Listen on output channels and collect results."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(*OUTPUT_CHANNELS)

    results = []
    start_time = asyncio.get_event_loop().time()

    logger.info("Listening for STT results...")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"].decode()
        data = message["data"]
        logger.info(f"Received result on {channel}")
        results.append(data)

        if len(results) >= expected_count:
            break

        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning("Timeout reached while waiting for results.")
            break

    await pubsub.unsubscribe(*OUTPUT_CHANNELS)
    return results


async def main(timeout: int = 60):
    input_dir = "/home/mehdi/Documents/projects/tts/test/test_wav"
    # input_dir = "/home/mehdi/Documents/projects/tts/tmp/corrupted"
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    wav_files = list(input_path.glob("*.wav"))
    num_sessions = len(wav_files)
    if not wav_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    logger.info(f"Found {len(wav_files)} WAV files.")

    redis_client = await redis.from_url("redis://localhost:6379", decode_responses=False)

    # Start listener first (to avoid missing fast results)
    listener_task = asyncio.create_task(listen_for_results(redis_client, len(wav_files), timeout))

    # Publish requests
    await publish_requests(redis_client, wav_files, num_sessions)

    # Wait for results
    try:
        results = await asyncio.wait_for(listener_task, timeout=timeout)
        logger.info(f"✅ Received {len(results)} / {len(wav_files)} results.")
    except asyncio.TimeoutError:
        logger.error("❌ Timeout: Not all results received.")
        results = listener_task.result() if listener_task.done() else []

    # Cleanup: mark sessions as stopped
    sids = [f"test_sid_{i}" for i in range(num_sessions)]
    for sid in sids:
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, "stop")

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())