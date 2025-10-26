import asyncio
import os
import tempfile
from pathlib import Path
import shutil
# --- Mock TtsFeatures (normally from abstract_models) ---
from dataclasses import dataclass

# from mode1 import AsyncModelInference, InferenceService
from main_service import *

async def main():
    service = InferenceService()
    await service.start()
    print("✅ Service started – running for 10 seconds...")

    # Keep the service alive to observe queue behavior
    await asyncio.sleep(10)

    # await service.stop()
    print("✅ Service stopped")

if __name__ == "__main__":
    import torch
    asyncio.run(main())