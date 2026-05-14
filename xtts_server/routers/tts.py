"""
routers/tts.py — TTS synthesis endpoints.

POST /v1/tts (async fire-and-poll)
----
  1. Validate request (text length, language, resolve speaker).
  2. Create a PENDING job in the job store.
  3. Build a SynthesisRequest with a fresh multiprocessing result queue.
  4. Register an on_complete callback that writes the audio file and
     transitions the job to DONE or FAILED.
  5. Submit to QueueManager — returns immediately with job_id.

POST /v1/tts/sync (synchronous — waits and returns audio directly)
----
  Same validation and queue path as the async endpoint (backpressure preserved),
  but bridges on_complete to an asyncio.Future so the handler can await the
  result and return the encoded audio in the HTTP response body.  No job record
  is written; the client simply waits until synthesis completes.

GET /v1/tts/{job_id}/audio
    Stream the finished audio file back to the client.
    Returns 404 if the job does not exist, 409 if it is not yet DONE.

Speaker resolution order
------------------------
  1. speaker_embedding (raw numpy arrays in request body) — used as-is
  2. speaker_name — loaded from SpeakerStore (disk → RAM cache)
  One of the two must be present; missing both → 422.

Language handling
-----------------
  Defaults to DEFAULT_LANGUAGE when omitted.
  A WARNING is logged when the request language differs from DEFAULT_LANGUAGE,
  but the request is never rejected on that basis.
"""

import asyncio
import multiprocessing
import os
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
import numpy as np
from pydantic import BaseModel, Field, field_validator

from audio import (
    SUPPORTED_FORMATS,
    AudioFormat,
    audio_to_bytes,
    mime_type,
    output_filename,
    save_audio,
)
from config import SUPPORTED_LANGUAGES
from dispatcher import WorkerHandle
from job_store import JobStatus, JobStore
from logging_config import get_logger
from queue_manager import QueueFullError
from speakers import SpeakerNotFoundError
from worker import SynthesisRequest, SynthesisResult

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/tts", tags=["tts"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TtsRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise.")
    language: str | None = Field(
        default=None,
        description="BCP-47 language code. Defaults to server DEFAULT_LANGUAGE.",
    )
    speaker_name: str | None = Field(
        default=None,
        description="Name of a pre-registered speaker in the speaker store.",
    )
    # Raw conditioning arrays — alternative to speaker_name for one-off requests.
    gpt_cond_latent: list[list[list[float]]] | None = Field(
        default=None,
        description="Pre-computed GPT conditioning latent (shape 1xTx1024).",
    )
    speaker_embedding: list[list[float]] | None = Field(
        default=None,
        description="Pre-computed speaker embedding (shape 1x512).",
    )
    format: AudioFormat = Field(
        default="wav",
        description="Output audio format: wav, mp3, ogg, or flac.",
    )

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{v}'. Choose from {SUPPORTED_FORMATS}.")
        return v

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str | None) -> str | None:
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'. Supported: {SUPPORTED_LANGUAGES}")
        return v


class TtsJobResponse(BaseModel):
    job_id: str
    status: str  # always "pending" on creation
    poll_url: str  # convenience URL for the client to poll


# ---------------------------------------------------------------------------
# POST /v1/tts — submit job
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TtsJobResponse,
    status_code=202,
    summary="Submit a TTS synthesis job",
    description=(
        "Enqueues a synthesis request and returns a job_id immediately. "
        "Poll GET /v1/jobs/{job_id} for status, then fetch audio from "
        "GET /v1/tts/{job_id}/audio when status is 'done'."
    ),
)
async def submit_tts(body: TtsRequest, request: Request) -> TtsJobResponse:
    state = request.app.state
    settings = state.settings
    job_store: JobStore = state.job_store

    # ---- Text length guard -------------------------------------------
    if len(body.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text length {len(body.text)} exceeds MAX_TEXT_LENGTH "
                f"({settings.MAX_TEXT_LENGTH})."
            ),
        )

    # ---- Language resolution -----------------------------------------
    language = body.language or settings.DEFAULT_LANGUAGE
    if language != settings.DEFAULT_LANGUAGE:
        logger.warning(
            "Non-default language requested | lang=%s | default=%s",
            language,
            settings.DEFAULT_LANGUAGE,
        )

    # ---- Speaker resolution ------------------------------------------
    gpt_cond_latent, speaker_embedding, speaker_id = _resolve_speaker(body, state)

    logger.info(
        "TTS request | lang=%s | speaker=%s | text_len=%d | format=%s",
        language,
        speaker_id,
        len(body.text),
        body.format,
    )

    # ---- Create job --------------------------------------------------
    job = await job_store.create(
        text=body.text,
        language=language,
        speaker_id=speaker_id,
    )

    # ---- Build worker request ----------------------------------------
    # Use the dispatcher's queue factory so all queues share the same
    # spawn context as the worker processes (avoids incompatible pipe handles).
    result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
    synth_request = SynthesisRequest(
        job_id=job.job_id,
        text=body.text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        result_queue=result_queue,
    )

    # ---- Register on_complete callback -------------------------------
    # This closure is called by QueueManager._collect_result once the
    # worker puts a SynthesisResult on result_queue.
    fmt = body.format
    outputs_dir = settings.OUTPUTS_DIR

    async def on_complete(worker: WorkerHandle, result: SynthesisResult) -> None:
        await job_store.mark_running(job.job_id, worker.worker_id, worker.gpu_index)
        if result.error:
            await job_store.mark_failed(job.job_id, result.error)
            return

        # save_audio is synchronous blocking I/O — run in thread pool so we
        # don't stall the event loop while encoding/writing the audio file.
        filename = output_filename(job.job_id, fmt)
        audio_path = os.path.join(outputs_dir, filename)
        loop = asyncio.get_running_loop()
        size_bytes = await loop.run_in_executor(
            None, save_audio, result.audio, audio_path, fmt, result.sample_rate
        )

        await job_store.mark_done(job.job_id, audio_path)
        logger.info(
            "Response ready | job_id=%s | format=%s | size=%d bytes",
            job.job_id,
            fmt,
            size_bytes,
        )

    # ---- Enqueue -------------------------------------------------
    try:
        await state.queue_manager.submit_job(synth_request, on_complete)
    except QueueFullError as exc:
        # Job was created but never dispatched — mark it failed and surface 503.
        await job_store.mark_failed(job.job_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TtsJobResponse(
        job_id=job.job_id,
        status="pending",
        poll_url=f"/v1/jobs/{job.job_id}",
    )


# ---------------------------------------------------------------------------
# POST /v1/tts/sync — synthesise and return audio immediately
# ---------------------------------------------------------------------------


@router.post(
    "/sync",
    summary="Synthesise audio and return the file immediately",
    description=(
        "Enqueues a synthesis request through the same worker pool as POST /v1/tts "
        "(backpressure and least-loaded routing are preserved), but waits for the "
        "result and returns the encoded audio file directly in the response body. "
        "No job record is created. The client blocks until synthesis completes."
    ),
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}, "audio/mpeg": {}, "audio/ogg": {}, "audio/flac": {}}},
        503: {"description": "Request queue is full — try again later"},
    },
)
async def synthesise_sync(body: TtsRequest, request: Request) -> Response:
    state = request.app.state
    settings = state.settings

    if len(body.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text length {len(body.text)} exceeds MAX_TEXT_LENGTH "
                f"({settings.MAX_TEXT_LENGTH})."
            ),
        )

    language = body.language or settings.DEFAULT_LANGUAGE
    if language != settings.DEFAULT_LANGUAGE:
        logger.warning(
            "Non-default language requested | lang=%s | default=%s",
            language,
            settings.DEFAULT_LANGUAGE,
        )

    gpt_cond_latent, speaker_embedding, speaker_id = _resolve_speaker(body, state)

    job_id = str(uuid.uuid4())
    logger.info(
        "TTS sync request | job_id=%s | lang=%s | speaker=%s | text_len=%d | format=%s",
        job_id,
        language,
        speaker_id,
        len(body.text),
        body.format,
    )

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[SynthesisResult] = loop.create_future()

    result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
    synth_request = SynthesisRequest(
        job_id=job_id,
        text=body.text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        result_queue=result_queue,
    )

    async def on_complete(worker: WorkerHandle, result: SynthesisResult) -> None:
        if not result_future.done():
            result_future.set_result(result)

    try:
        await state.queue_manager.submit_job(synth_request, on_complete)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result: SynthesisResult = await result_future

    if result.error:
        logger.error("TTS sync failed | job_id=%s | error=%s", job_id, result.error[:300])
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {result.error[:300]}",
        )

    fmt = body.format
    audio_data = audio_to_bytes(result.audio, fmt, result.sample_rate)

    logger.info(
        "TTS sync done | job_id=%s | format=%s | size=%d bytes",
        job_id,
        fmt,
        len(audio_data),
    )

    return Response(
        content=audio_data,
        media_type=mime_type(fmt),
        headers={
            "Content-Disposition": f'attachment; filename="speech.{fmt}"',
            "X-Job-Id": job_id,
        },
    )


# ---------------------------------------------------------------------------
# GET /v1/tts/{job_id}/audio — download finished audio
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/audio",
    summary="Download synthesised audio",
    description=(
        "Returns the audio file for a completed job. "
        "Returns 404 if the job does not exist, 409 if synthesis is not yet done."
    ),
)
async def get_audio(job_id: str, request: Request):
    job = await request.app.state.job_store.get(job_id)

    if job is None:
        logger.warning("Audio download — job not found: %s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Job '{job_id}' failed: {job.error}",
        )

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not done yet (status: {job.status.value}).",
        )

    if not job.audio_path or not os.path.isfile(job.audio_path):
        logger.error("Audio file missing on disk for job %s: %s", job_id, job.audio_path)
        raise HTTPException(status_code=500, detail="Audio file not found on disk.")

    # Derive the format from the file extension so we set the correct MIME type.
    ext = os.path.splitext(job.audio_path)[1].lstrip(".")
    media_type = mime_type(ext) if ext in SUPPORTED_FORMATS else "application/octet-stream"

    logger.info("Serving audio | job_id=%s | path=%s", job_id, job.audio_path)
    return FileResponse(
        path=job.audio_path,
        media_type=media_type,
        filename=os.path.basename(job.audio_path),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _resolve_speaker(
    body: TtsRequest,
    state,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return (gpt_cond_latent, speaker_embedding, speaker_id_label).

    Priority:
      1. Raw arrays from the request body (gpt_cond_latent + speaker_embedding).
      2. speaker_name looked up in the SpeakerStore.
    Raises HTTP 422 if neither is provided, 404 if speaker_name is unknown.
    """
    # Path 1 — raw arrays supplied directly in the request.
    if body.gpt_cond_latent is not None and body.speaker_embedding is not None:
        gpt = np.array(body.gpt_cond_latent, dtype=np.float32)
        emb = np.array(body.speaker_embedding, dtype=np.float32)
        return gpt, emb, "inline"

    # Path 2 — named speaker from the store.
    if body.speaker_name is not None:
        try:
            record = state.speaker_store.get(body.speaker_name)
        except SpeakerNotFoundError as exc:
            logger.warning("Speaker not found: %s", body.speaker_name)
            raise HTTPException(
                status_code=404,
                detail=f"Speaker '{body.speaker_name}' not found.",
            ) from exc
        return record.gpt_cond_latent, record.speaker_embedding, body.speaker_name

    # Neither provided.
    raise HTTPException(
        status_code=422,
        detail=(
            "Provide either 'speaker_name' (a registered speaker) or both "
            "'gpt_cond_latent' and 'speaker_embedding'."
        ),
    )
