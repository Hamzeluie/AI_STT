import asyncio
import json
import websockets
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP, Context

load_dotenv()
MCP_WS_URL = f"ws://{os.getenv("HOST", "localhost")}:{os.getenv("PORT", "localhost")}/ws/stt"

mcp = FastMCP("speech_to_text_tool")


async def call_stt_server(audio, sample_rate=16000):
    async with websockets.connect(MCP_WS_URL) as ws:
        # send message
        await ws.send(json.dumps({
            "audio": audio,
            "sample_rate": sample_rate
        }))
        # wait for transcript
        response = await ws.recv()
        return json.loads(response)


@mcp.tool()
async def transcribe_audio(ctx: Context, audio: list[int], sample_rate: int = 16000):
    """
    Send audio to FastAPI WebSocket STT server and return transcription.
    """
    result = await call_stt_server(audio, sample_rate)
    return result


if __name__ == "__main__":
    mcp.run()
