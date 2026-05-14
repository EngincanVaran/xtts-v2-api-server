"""
ws/stream.py — WebSocket endpoint for real-time PCM audio streaming.

Protocol
--------
  1. Client connects to  ws://<host>/v1/stream
  2. Client sends a JSON text frame with the synthesis parameters:
       {
         "text": "...",
         "language": "tr",          // optional, defaults to DEFAULT_LANGUAGE
         "speaker_name": "alice",   // or supply raw embeddings below
         "gpt_cond_latent": [...],  // optional, alternative to speaker_name
         "speaker_embedding": [...] // optional, alternative to speaker_name
       }
  3. Server responds with binary frames — raw float32 PCM at 24 000 Hz (mono).
     Each frame is one synthesis chunk as produced by model.inference_stream().
  4. Server sends a final text frame:  {"status": "done"}
     On error it sends:               {"status": "error", "detail": "..."}
  5. Connection closes after the final frame.

Why raw float32 PCM?
--------------------
  Sending an encoded container format (WAV, MP3) over a stream would require
  buffering all chunks first so the header can be written.  Raw float32 lets
  the client start playing back the first chunk (~200-400 ms latency) without
  waiting for the full audio.  The client is responsible for knowing the
  sample rate (24 000 Hz) and converting to its preferred format.

Backpressure
------------
  The stream request enters the same QueueManager as regular jobs.  If all
  workers are busy the WebSocket connection waits until a slot opens rather
  than being dropped immediately.  MAX_QUEUE_SIZE still applies — if the
  queue is full the connection is closed with a 1013 (try again later) code.
"""

import asyncio
import contextlib
import json
import multiprocessing
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from pydantic import BaseModel, ValidationError

from config import SUPPORTED_LANGUAGES
from logging_config import get_logger
from queue_manager import QueueFullError
from routers.tts import _resolve_speaker
from worker import StreamSynthesisRequest, SynthesisChunk, SynthesisStreamEnd

logger = get_logger(__name__)

router = APIRouter(tags=["stream"])


# ---------------------------------------------------------------------------
# Incoming message schema (validated after receiving the JSON frame)
# ---------------------------------------------------------------------------


class StreamRequest(BaseModel):
    text: str
    language: str | None = None
    speaker_name: str | None = None
    gpt_cond_latent: list[list[list[float]]] | None = None
    speaker_embedding: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/v1/stream")
async def stream_tts(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    settings = state.settings
    job_id = str(uuid.uuid4())

    logger.info("WebSocket connected | job_id=%s | client=%s", job_id, websocket.client)

    try:
        # ---- Step 1: receive synthesis parameters -------------------
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        except TimeoutError:
            await _close_error(websocket, job_id, "Timed out waiting for request JSON.")
            return

        try:
            params = StreamRequest.model_validate_json(raw)
        except ValidationError as exc:
            await _close_error(websocket, job_id, f"Invalid request: {exc}")
            return

        # ---- Step 2: validate text length ---------------------------
        if len(params.text) > settings.MAX_TEXT_LENGTH:
            await _close_error(
                websocket,
                job_id,
                f"Text length {len(params.text)} exceeds MAX_TEXT_LENGTH "
                f"({settings.MAX_TEXT_LENGTH}).",
            )
            return

        # ---- Step 3: resolve language --------------------------------
        language = params.language or settings.DEFAULT_LANGUAGE
        if language not in SUPPORTED_LANGUAGES:
            await _close_error(websocket, job_id, f"Unsupported language: '{language}'")
            return
        if language != settings.DEFAULT_LANGUAGE:
            logger.warning("Stream job %s — non-default language: %s", job_id, language)

        # ---- Step 4: resolve speaker ---------------------------------
        try:
            gpt_cond_latent, speaker_embedding, speaker_id = _resolve_speaker(params, state)
        except Exception as exc:
            # _resolve_speaker raises HTTPException; extract detail for WS response.
            detail = getattr(exc, "detail", str(exc))
            await _close_error(websocket, job_id, detail)
            return

        logger.info(
            "Stream synthesis | job_id=%s | lang=%s | speaker=%s | text_len=%d",
            job_id,
            language,
            speaker_id,
            len(params.text),
        )

        # ---- Step 5: build request and enqueue ----------------------
        # Use dispatcher's queue factory to match the spawn context of workers.
        result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
        stream_request = StreamSynthesisRequest(
            job_id=job_id,
            text=params.text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            result_queue=result_queue,
        )

        try:
            worker_future = await state.queue_manager.submit_stream(stream_request)
        except QueueFullError as exc:
            await _close_error(websocket, job_id, str(exc))
            return

        # ---- Step 6: wait until dispatched, then stream chunks ------
        loop = asyncio.get_running_loop()

        # Block (async) until the drain loop assigns a worker.  This is when
        # the request leaves the asyncio queue and enters a worker process.
        worker = await worker_future

        chunk_count = 0

        try:
            # try/finally ensures the worker slot is always released, even
            # if the client disconnects mid-stream or an error occurs.
            while True:
                # Run the blocking queue.get() in a thread-pool executor so the
                # event loop remains free to handle other connections.
                item = await loop.run_in_executor(None, result_queue.get)

                if isinstance(item, SynthesisChunk):
                    # Send raw float32 bytes — client knows sample rate is 24 000 Hz.
                    await websocket.send_bytes(item.chunk.astype(np.float32).tobytes())
                    chunk_count += 1

                elif isinstance(item, SynthesisStreamEnd):
                    if item.error:
                        logger.error(
                            "Stream error | job_id=%s | error=%s",
                            job_id,
                            item.error[:300],
                        )
                        await _close_error(websocket, job_id, item.error[:300])
                    else:
                        logger.info("Stream done | job_id=%s | chunks=%d", job_id, chunk_count)
                        await websocket.send_text(json.dumps({"status": "done"}))
                        await websocket.close()
                    break
        finally:
            # Always release the worker slot — covers normal completion,
            # client disconnect (WebSocketDisconnect), and unexpected errors.
            await state.dispatcher.release(worker.worker_id, elapsed_ms=0.0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client | job_id=%s", job_id)
    except Exception as exc:
        logger.exception("Unexpected WebSocket error | job_id=%s | %s", job_id, exc)
        with contextlib.suppress(Exception):
            await _close_error(websocket, job_id, "Internal server error.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _close_error(websocket: WebSocket, job_id: str, detail: str) -> None:
    """Send an error JSON frame and close the connection."""
    logger.warning("Stream closing with error | job_id=%s | detail=%s", job_id, detail)
    with contextlib.suppress(Exception):
        await websocket.send_text(json.dumps({"status": "error", "detail": detail}))
        await websocket.close(code=1013)  # 1013 = Try Again Later
