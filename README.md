# QUAKER-XTTS Inference Server

A production-grade, multi-GPU text-to-speech API server built on [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) and FastAPI.

## Features

- **Multi-GPU worker pool** — spawn multiple XTTS-v2 processes across any number of GPUs, each configured independently
- **Async fire-and-poll jobs** — `POST /v1/tts` returns a `job_id` instantly; clients poll until done
- **Synchronous endpoint** — `POST /v1/tts/sync` blocks and returns the audio file directly in the response
- **WebSocket streaming** — receive raw PCM chunks as they are generated (~200–400 ms to first audio)
- **58 built-in studio speakers** — seed from the model's `speakers_xtts.pth` in seconds, no audio upload needed
- **Voice cloning** — upload a reference audio clip, register a speaker name, reuse it forever
- **Batch synthesis** — submit up to 50 TTS items in a single request
- **Request queue** — clients wait instead of receiving 503 (only returned when the queue itself is full)
- **Live worker observability** — periodic log shows each in-flight job, elapsed time, queue depth, and total throughput per worker; also exposed via `GET /v1/system/info`
- **One-shot audio download** — `GET /v1/tts/{id}/audio` deletes the file immediately after serving; second download returns `410 Gone`
- **Verbose structured logging** — every request lifecycle event, GPU VRAM, worker stats, RTF, first-chunk latency
- **HTTP access log middleware** — one line per request with method, path, status, duration, client IP, and an `X-Request-ID` correlation header
- **GPU memory cap** — set `GPU_MEMORY_FRACTION` (e.g. `0.8`) to limit per-worker VRAM via `torch.cuda.set_per_process_memory_fraction`
- **GPU selection** — `CUDA_VISIBLE_DEVICES` controls which physical GPUs are exposed to the server
- **QUAKER-XTTS startup banner** — Spring Boot / vLLM-style ASCII art banner on every startup

---

## Architecture

```
Client
  │
  ├── POST /v1/tts          ──►  QueueManager (asyncio.Queue)
  ├── POST /v1/tts/sync     ──►        │  (blocks, returns audio directly)
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

**Linux (GPU) — recommended path:**

```bash
chmod +x install.sh
./install.sh
```

`install.sh` runs the full setup in one shot:
1. Checks OS, CUDA (≥ 12.1), Python (≥ 3.11), `nvidia-smi`
2. Creates `.venv/` and upgrades `pip`
3. Installs `pip-tools` into the venv and runs `pip-compile requirements.in -o requirements.txt` to generate a fresh lockfile
4. Installs `coqui-tts==0.27.1 --no-deps` (encodec license exclusion)
5. Installs all remaining deps from the generated `requirements.txt`
6. Verifies `torch.cuda.is_available()` and GPU count
7. Copies `.env.example → xtts_server/.env` if absent, creates `speakers/` and `outputs/`

Only `requirements.in` needs to be present — `requirements.txt` is generated during install.

> **macOS note:** `llvmlite` (a transitive dep via numba → librosa → TTS) does not build from source on macOS without LLVM. Use conda instead of a plain venv:
> ```bash
> brew install miniconda
> conda create -n xtts python=3.11
> conda activate xtts
> conda install -c conda-forge llvmlite numba librosa
> pip install -r requirements.txt
> ```

### 3. Seed the studio speakers (optional but recommended)

XTTS-v2 ships with 58 built-in voices stored in `speakers_xtts.pth`. Register them all in seconds — no audio upload, no model load required:

```bash
# run from project root
python seed_studio_speakers.py --model-path ./model --speakers-dir ./xtts_server/speakers
```

| Flag | Description |
|---|---|
| `--model-path` | Directory containing `speakers_xtts.pth` (default: `$MODEL_PATH` or `./model`) |
| `--speakers-dir` | SpeakerStore root (default: `$SPEAKERS_DIR` or `./xtts_server/speakers`) |
| `--force` | Overwrite already-registered speakers |
| `--dry-run` | Preview what would be written without touching disk |

Speaker names are derived from the display names by replacing spaces with underscores and stripping accented characters (e.g. `"Alma María"` → `"Alma_Maria"`). The original name is preserved in `meta.json` as `"original_name"`.

After seeding, use any studio speaker directly:

```bash
curl -X POST http://localhost:8000/v1/tts/sync \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "speaker_name": "Craig_Gutsy", "language": "en"}' \
  -o output.wav
```

<details>
<summary>All 58 studio speaker slugs</summary>

`Aaron_Dreschner` `Abrahan_Mack` `Adde_Michal` `Alexandra_Hisakawa` `Alison_Dietlinde` `Alma_Maria` `Ana_Florence` `Andrew_Chipper` `Annmarie_Nele` `Asya_Anara` `Badr_Odhiambo` `Baldur_Sanjin` `Barbora_MacLean` `Brenda_Stern` `Camilla_Holmstrom` `Chandra_MacFarland` `Claribel_Dervla` `Craig_Gutsy` `Daisy_Studious` `Damien_Black` `Damjan_Chapman` `Dionisio_Schuyler` `Eugenio_Matarac` `Ferran_Simen` `Filip_Traverse` `Gilberto_Mathias` `Gitta_Nikolina` `Gracie_Wise` `Henriette_Usha` `Ige_Behringer` `Ilkin_Urbano` `Kazuhiko_Atallah` `Kumar_Dahl` `Lidiya_Szekeres` `Lilya_Stainthorpe` `Ludvig_Milivoj` `Luis_Moray` `Maja_Ruoho` `Marcos_Rudaski` `Narelle_Moon` `Nova_Hogarth` `Rosemary_Okafor` `Royston_Min` `Sofia_Hellen` `Suad_Qasim` `Szofi_Granger` `Tammie_Ema` `Tammy_Grit` `Tanja_Adelina` `Torcull_Diarmuid` `Uta_Obando` `Viktor_Eka` `Viktor_Menelaos` `Vjollca_Johnnie` `Wulf_Carlevaro` `Xavier_Hayasaka` `Zacharie_Aimilios` `Zofija_Kendrick`

</details>

### 4. Run the server

**Linux (GPU) — recommended path:**

```bash
# Edit the CONFIGURATION block at the top of start-server.sh, then:
chmod +x start-server.sh
./start-server.sh
```

`start-server.sh` validates the environment (GPU inventory, CUDA version, torch install), optionally seeds studio speakers, then launches the server.

**Manual / macOS dev:**

```bash
export MODEL_PATH=/path/to/your/xtts-v2
cd xtts_server && python main.py
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
| `CUDA_VISIBLE_DEVICES` | all GPUs | Comma-separated GPU indices to expose (e.g. `"0,1"`) |
| `GPU_MEMORY_FRACTION` | `1.0` | Per-worker VRAM cap as a fraction of total (e.g. `0.8` = 80%) |

### Multi-GPU example

```bash
# 3 GPUs: 2 workers on GPU 0, 3 on GPU 1, 2 on GPU 2 (7 concurrent synthesis processes)
export MODEL_PATH=/models/xtts-v2
export WORKERS_PER_GPU=2,3,2
export CUDA_VISIBLE_DEVICES=0,1,2
export GPU_MEMORY_FRACTION=0.8
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

### Synthesise speech (sync — waits and returns audio directly)

```bash
curl -X POST http://localhost:8000/v1/tts/sync \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "speaker_name": "Claribel_Dervla", "language": "en"}' \
  -o speech.wav
```

The response body **is** the audio file. An `X-Job-Id` header is included for log correlation.

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

The `workers` array in the response now includes `active_jobs` — a list of `{job_id, elapsed_s}` objects for every synthesis currently in progress on that worker:

```json
{
  "workers": [
    {
      "worker_id": "gpu0-w0",
      "active_requests": 1,
      "total_requests": 47,
      "avg_synthesis_ms": 1842.0,
      "active_jobs": [
        { "job_id": "550e8400-...", "elapsed_s": 1.4 }
      ]
    }
  ]
}
```

The same information is logged every 60 seconds:

```
--- Worker status | queue=0 waiting ---
  worker=gpu0-w0  gpu=0  alive=True  active=1  total=47  avg_ms=1842.0
    job=550e8400  elapsed=1.4s
  worker=gpu0-w1  gpu=0  alive=True  active=0  total=31  avg_ms=1650.0
    (idle)
--- End worker status ---
```

### Download audio (one-shot)

```bash
curl http://localhost:8000/v1/tts/550e8400-.../audio -o output.wav
```

The audio file is deleted from disk immediately after the response is sent. A second request for the same job returns `410 Gone`. The job record itself stays in memory until the TTL expires.

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
├── install.sh               # Linux-only setup: checks CUDA/Python/torch, creates .venv, installs deps
├── start-server.sh          # GPU-aware launch: validates env, optional speaker seed, then exec python main.py
├── requirements.in          # Direct dependencies — edit this, not requirements.txt
├── requirements.txt         # Pinned lockfile (generated by install.sh via pip-compile)
├── pyproject.toml           # ruff config
├── Dockerfile
├── .env.example
├── seed_studio_speakers.py  # One-time script: register all 58 studio voices
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
    │   ├── tts.py          # POST /v1/tts, POST /v1/tts/sync, GET /v1/tts/{id}/audio
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
- **Startup:** QUAKER-XTTS ASCII art banner, config dump, MODEL_PATH file check with sizes, GPU inventory, worker spawn
- **Access log:** one line per HTTP request — `METHOD /path → STATUS | elapsed_ms | client=IP | req=ID`. `/health` is logged at DEBUG to reduce noise from probes. The `X-Request-ID` header is echoed back to the client (or generated if absent) for client-side correlation.
- **Per request:** text preview, language, speaker, synthesis duration, audio duration, realtime factor (RTF)
- **WebSocket stream:** dispatched worker, first-chunk latency, per-stream RTF, total bytes
- **Job lifecycle:** total time from queue entry to `on_complete` completion (queue wait + synthesis + audio save)
- **Voice clone:** latent computation time, gpt/embedding shapes
- **Periodic (every 60 s):** per-worker active jobs with elapsed time, GPU VRAM, queue depth, job store counts
- **Errors:** synthesis failures with full tracebacks, missing files, queue overflow

---

## License

This server is released under the MIT License. The underlying XTTS-v2 model is subject to the [Coqui Public Model License](https://coqui.ai/cpml) — review it before commercial use.
