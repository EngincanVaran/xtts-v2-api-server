# XTTS-v2 Inference Server

A production-grade, multi-GPU text-to-speech API server built on [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) and FastAPI.

## Features

- **Multi-GPU worker pool** — spawn multiple XTTS-v2 processes across any number of GPUs, each configured independently
- **Async fire-and-poll jobs** — `POST /v1/tts` returns a `job_id` instantly; clients poll until done
- **WebSocket streaming** — receive raw PCM chunks as they are generated (~200–400 ms to first audio)
- **Voice cloning** — upload a reference audio clip, register a speaker name, reuse it forever
- **Batch synthesis** — submit up to 50 TTS items in a single request
- **Request queue** — clients wait instead of receiving 503 (only returned when the queue itself is full)
- **Verbose structured logging** — every request lifecycle event, GPU VRAM, worker stats, RTF

---

## Architecture

```
Client
  │
  ├── POST /v1/tts          ──►  QueueManager (asyncio.Queue)
  ├── POST /v1/batch        ──►        │
  ├── WS   /v1/stream       ──►        ▼
  │                              Dispatcher
  │                         (least-loaded routing)
  │                          ┌────┬────┬────┐
  │                          W0   W1   W2  ...   ← multiprocessing.Process
  │                         GPU0 GPU0 GPU1        each owns one XTTS-v2 model
  │
  ├── GET  /v1/jobs/{id}    ──►  JobStore (in-memory, TTL cleanup)
  ├── GET  /v1/tts/{id}/audio ►  OUTPUTS_DIR/{job_id}.{fmt}
  ├── POST /v1/clone        ──►  SpeakerStore (disk + RAM cache)
  ├── GET  /v1/speakers     ──►  SpeakerStore
  └── GET  /v1/system/info  ──►  GPU VRAM + worker stats + queue depth
```

### Why `inference()` instead of `synthesize()`

`model.synthesize()` re-encodes the reference audio on every call (100–500 ms overhead). This server pre-computes `gpt_cond_latent` and `speaker_embedding` once at speaker registration time (stored as `.npz`) and calls `model.inference()` directly, skipping that cost on every request.

For WebSocket streaming, `model.inference_stream()` is used instead — it yields audio chunks as the GPT decoder runs, so the first audio arrives in ~200–400 ms rather than waiting for the full clip.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (CPU fallback supported, but slow)
- [ffmpeg](https://ffmpeg.org/) on PATH (required for MP3 output only)
- A local XTTS-v2 model directory containing:
  ```
  config.json
  model.pth
  vocab.json
  speakers_xtts.pth   (optional)
  ```

### 2. Install dependencies

> **macOS note:** `llvmlite` (a transitive dep via numba → librosa → TTS) does not build from source on macOS without LLVM. Use conda instead of a plain venv:
> ```bash
> brew install miniconda
> conda create -n xtts python=3.11
> conda activate xtts
> conda install -c conda-forge llvmlite numba librosa
> pip install -r requirements.txt
> ```

On Linux / Docker, a plain pip install works:

```bash
pip install pip-tools
pip install -r requirements.txt
```

To regenerate the lockfile after editing `requirements.in`:

```bash
pip-compile requirements.in -o requirements.txt
```

### 3. Run the server

```bash
export MODEL_PATH=/path/to/your/xtts-v2
python main.py
```

The server starts at `http://localhost:8000`.  
OpenAPI docs: `http://localhost:8000/docs`

---

## Configuration

All settings are read from environment variables (or a `.env` file in the working directory).

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | **required** | Path to local XTTS-v2 model directory |
| `NUM_GPUS` | auto-detect | Number of GPUs to use |
| `WORKERS_PER_GPU` | `"1"` | Workers per GPU. Single int or comma-separated list e.g. `"2,3,2"` |
| `DEFAULT_LANGUAGE` | `"tr"` | Language used when request omits `language` field |
| `MAX_QUEUE_SIZE` | `100` | Max requests waiting before 503 is returned |
| `JOB_TTL_SECONDS` | `300` | How long completed jobs stay in the store before cleanup |
| `SPEAKERS_DIR` | `./speakers` | Directory for speaker audio and latent files |
| `OUTPUTS_DIR` | `./outputs` | Directory for rendered audio files |
| `MAX_TEXT_LENGTH` | `5000` | Maximum input text length (characters) |
| `SAMPLE_RATE` | `24000` | XTTS-v2 output sample rate (do not change) |
| `LOG_LEVEL` | `INFO` | Python logging level: DEBUG, INFO, WARNING, ERROR |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

### Multi-GPU example

```bash
# 3 GPUs: 2 workers on GPU 0, 3 on GPU 1, 2 on GPU 2 (7 concurrent synthesis processes)
export MODEL_PATH=/models/xtts-v2
export WORKERS_PER_GPU=2,3,2
python main.py
```

---

## API Reference

### Register a speaker voice

```bash
curl -X POST http://localhost:8000/v1/clone \
  -F "speaker_name=alice" \
  -F "audio=@/path/to/alice_reference.wav"
```

Response:
```json
{
  "speaker_name": "alice",
  "created_at": "2026-05-13T10:00:00+00:00",
  "message": "Speaker 'alice' registered. Use it with POST /v1/tts."
}
```

### Synthesise speech (async job)

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba, nasılsın?", "speaker_name": "alice", "format": "wav"}'
```

Response `202 Accepted`:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "poll_url": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
}
```

### Poll job status

```bash
curl http://localhost:8000/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```

Response when done:
```json
{
  "job_id": "550e8400-...",
  "status": "done",
  "queue_wait_ms": 12.3,
  "synthesis_ms": 1842.1,
  "audio_url": "/v1/tts/550e8400-.../audio"
}
```

### Download audio

```bash
curl http://localhost:8000/v1/tts/550e8400-.../audio -o output.wav
```

### Batch synthesis

```bash
curl -X POST http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Birinci cümle.", "speaker_name": "alice"},
      {"text": "İkinci cümle.", "speaker_name": "alice"},
      {"text": "Third sentence.", "speaker_name": "alice", "language": "en"}
    ]
  }'
```

### WebSocket streaming

```python
import asyncio, json, numpy as np, websockets

async def stream():
    async with websockets.connect("ws://localhost:8000/v1/stream") as ws:
        await ws.send(json.dumps({
            "text": "Bu ses gerçek zamanlı akıyor.",
            "speaker_name": "alice"
        }))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # Raw float32 PCM at 24 000 Hz — pipe to your audio player
                pcm = np.frombuffer(msg, dtype=np.float32)
                print(f"Received {len(pcm)} samples")
            else:
                data = json.loads(msg)
                print("Stream ended:", data["status"])
                break

asyncio.run(stream())
```

### System info

```bash
curl http://localhost:8000/v1/system/info
```

---

## Docker

```bash
# Build
docker build -t xtts-server .

# Run with GPU support
docker run --gpus all \
  -e MODEL_PATH=/models/xtts-v2 \
  -e WORKERS_PER_GPU=2 \
  -v /host/models/xtts-v2:/models/xtts-v2:ro \
  -v /host/speakers:/app/speakers \
  -v /host/outputs:/app/outputs \
  -p 8000:8000 \
  xtts-server
```

> **Note:** Mount your model directory read-only (`:ro`) — the server never writes to MODEL_PATH.

---

## Project Structure

```
xtts-v2-api-server/
├── requirements.in     # Direct dependencies (pip-tools source)
├── requirements.txt    # Pinned lockfile (generated — do not edit by hand)
├── pyproject.toml      # ruff config
├── Dockerfile
├── .env.example
└── xtts_server/
    ├── main.py             # FastAPI app factory, startup/shutdown lifecycle
    ├── config.py           # Pydantic settings, MODEL_PATH validation
    ├── dispatcher.py       # Worker process pool, least-loaded routing
    ├── queue_manager.py    # asyncio waiting queue, backpressure
    ├── job_store.py        # In-memory job state + TTL cleanup
    ├── worker.py           # XTTS-v2 model process (inference, stream, latents)
    ├── speakers.py         # Speaker registration, disk persistence, RAM cache
    ├── audio.py            # PCM → WAV / MP3 / OGG / FLAC encoding
    ├── logging_config.py   # Structured logging, rotating file handler
    ├── routers/
    │   ├── system.py       # GET /health, GET /v1/system/info
    │   ├── tts.py          # POST /v1/tts, GET /v1/tts/{id}/audio
    │   ├── clone.py        # POST /v1/clone
    │   ├── batch.py        # POST /v1/batch
    │   ├── jobs.py         # GET /v1/jobs/{id}, GET /v1/jobs
    │   └── speakers.py     # GET/DELETE /v1/speakers
    └── ws/
        └── stream.py       # WS /v1/stream
```

---

## Supported Languages

`en` `es` `fr` `de` `it` `pt` `pl` `tr` `ru` `nl` `cs` `ar` `zh-cn` `ja` `hu` `ko` `hi`

The server defaults to Turkish (`tr`). Requests in other languages are accepted without restriction; a `WARNING` is logged when the language differs from `DEFAULT_LANGUAGE`.

---

## Logging

Logs are written to both stdout and `xtts_server/logs/xtts_server.log` (rotating, 10 MB × 3 backups).

Key log events:
- Startup: config dump, MODEL_PATH file check with sizes, GPU inventory, worker spawn
- Per request: text preview, language, speaker, queue position, worker assigned, synthesis duration, realtime factor (RTF)
- Periodic (every 60 s): per-worker stats, GPU VRAM usage, queue depth, job store counts
- Errors: synthesis failures with full tracebacks, missing files, queue overflow

---

## License

This server is released under the MIT License. The underlying XTTS-v2 model is subject to the [Coqui Public Model License](https://coqui.ai/cpml) — review it before commercial use.
