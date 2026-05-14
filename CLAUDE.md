# CLAUDE.md — Developer Cheatsheet

This file is a checkpoint for Claude Code (and human developers). It captures architecture decisions, non-obvious invariants, known gotchas, and the reasoning behind key design choices so you can continue development without re-deriving context.

---

## Project Layout

```
xtts-v2-api-server/
├── requirements.in       # hand-edited direct deps
├── requirements.txt      # generated lockfile (pip-compile)
├── pyproject.toml        # ruff config
├── Dockerfile
├── .env.example
└── xtts_server/          # ALL Python source lives here
    ├── main.py
    ├── config.py
    ├── dispatcher.py
    ├── queue_manager.py
    ├── job_store.py
    ├── worker.py
    ├── speakers.py
    ├── audio.py
    ├── logging_config.py
    ├── routers/
    │   ├── system.py
    │   ├── tts.py
    │   ├── clone.py
    │   ├── batch.py
    │   ├── jobs.py
    │   └── speakers.py
    └── ws/
        └── stream.py
```

Run the server from `xtts_server/`:
```bash
cd xtts_server && python main.py
```

---

## Core Data Flow

```
POST /v1/tts
  → TtsRequest validated (Pydantic)
  → _resolve_speaker() → (gpt_cond_latent: np.ndarray, speaker_embedding: np.ndarray, speaker_id: str)
  → job_store.create() → Job (PENDING)
  → dispatcher.make_queue() → multiprocessing.Queue (spawn context)
  → SynthesisRequest(job_id, text, lang, latents, result_queue)
  → queue_manager.submit_job(request, on_complete)
      → asyncio.Queue.put_nowait (raises QueueFullError if full)
  → return 202 {job_id, poll_url}

[drain loop]
  → asyncio.Queue.get()
  → dispatcher.wait_for_free_worker()   # asyncio.Condition wait
  → dispatcher.dispatch(request)         # put into worker's mp.Queue in executor
  → asyncio.create_task(_collect_result)

[worker process]
  → receives SynthesisRequest from mp.Queue
  → model.inference(text, lang, gpt_latent_tensor, spk_emb_tensor)
  → puts SynthesisResult(job_id, audio_np, sample_rate) on result_queue

[_collect_result task]
  → run_in_executor(result_queue.get)   # blocking get, off event loop
  → dispatcher.release(worker_id)       # notifies Condition → drain loop unblocks
  → on_complete(worker, result)
      → job_store.mark_running → mark_done
      → run_in_executor(save_audio, ...) # blocking I/O off event loop
```

---

## Critical Design Decisions

### 1. `inference()` not `synthesize()`
`model.synthesize()` re-encodes the reference audio WAV on every call — adds 100-500 ms overhead.
This server calls `model.inference(text, lang, gpt_cond_latent, speaker_embedding)` directly with **pre-computed** latents. Latents are computed once at clone time via `model.get_conditioning_latents()` and stored as `.npz` on disk.

For streaming: `model.inference_stream()` — yields chunks so the first audio arrives ~200-400 ms into synthesis.

### 2. `multiprocessing.get_context("spawn")`
CUDA does not survive `fork()`. All worker processes and their `multiprocessing.Queue` instances **must** be created with the `spawn` context. The module-level `_MP_CTX = multiprocessing.get_context("spawn")` in `dispatcher.py` is the single source of truth. Always use `dispatcher.make_queue()` — never `multiprocessing.Queue()` directly.

### 3. Lazy `asyncio.Condition` in Dispatcher
`asyncio.Condition` (and `asyncio.Lock`) must be created **inside a running event loop** — creating them in `__init__` before the loop exists causes "attached to a different loop" errors. `dispatcher.py` stores `self._cond: asyncio.Condition | None = None` and creates it lazily on first use via `_get_cond()`.

### 4. No Redis
Job state is a plain `dict[str, Job]` guarded by `asyncio.Lock`. TTL cleanup runs every 60 s in a background task. This is sufficient for single-process deployments. If you ever need multi-process uvicorn workers or cross-node sharing, add Redis then.

### 5. Blocking I/O always in `run_in_executor`
The event loop must never block. Three places where this matters:
- `result_queue.get()` — in `queue_manager._collect_result` and `ws/stream.py`
- `save_audio()` (soundfile write) — in the `on_complete` callback in `tts.py` and `batch.py`
- `process.join()` — in `dispatcher.shutdown`

### 6. Workers are released before `on_complete` is awaited
In `_collect_result`: `dispatcher.release()` is called **before** `on_complete()`. This is intentional — the worker slot is freed as soon as synthesis is done so the drain loop can dispatch the next request. Audio encoding (`save_audio`) happens after release and does not hold up the next job.

### 7. `asyncio.Future[WorkerHandle]` for WebSocket streams
`submit_stream()` returns a `Future` that is resolved by the drain loop once the request is dispatched. The WebSocket handler `await`s this future to learn which `WorkerHandle` was assigned, so it can call `dispatcher.release(worker.worker_id)` in the `finally` block. Without this, the WS handler would not know which worker to release.

### 8. Worker slot release on WebSocket disconnect
`ws/stream.py` wraps the chunk-reading loop in `try/finally`:
```python
try:
    while True:
        item = await loop.run_in_executor(None, result_queue.get)
        ...
finally:
    await state.dispatcher.release(worker.worker_id, elapsed_ms=0.0)
```
This guarantees the slot is freed even if the client disconnects mid-stream or an unexpected error occurs.

---

## Speaker Store

On-disk layout for each speaker (`SPEAKERS_DIR/{name}/`):
```
ref.wav        # copy of the original reference audio
latents.npz    # gpt_cond_latent (shape 1×T×1024) + speaker_embedding (shape 1×512)
meta.json      # {"name": "...", "created_at": "ISO-8601"}
```

`SpeakerStore` keeps a `dict[str, SpeakerRecord]` RAM cache. `preload_all()` is called at startup. `get()` falls back to disk on cache miss. `delete()` evicts cache **before** `shutil.rmtree` (under `threading.Lock`) — this order is important; reversing it creates a window where the cache holds a record pointing to deleted files.

Worker processes receive latents as **CPU numpy arrays** (picklable). Each worker converts them to device tensors inside `_handle_synthesis()`. Never pass GPU tensors across process boundaries.

---

## Job Lifecycle

```
PENDING  → created, waiting in asyncio.Queue
RUNNING  → dispatched to a worker (mark_running called by on_complete)
DONE     → audio written to disk (audio_path set)
FAILED   → synthesis or I/O error (error message stored)
```

Only DONE and FAILED jobs are evicted by the TTL cleanup. PENDING/RUNNING jobs are never touched.

TTL default: 300 s. Audio files are deleted from disk at eviction time.

---

## Error Handling Patterns

- **HTTP exceptions:** always `raise HTTPException(...) from exc` (ruff B904)
- **Silent swallowing:** always `contextlib.suppress(SomeException)` not `try/except: pass` (ruff SIM105)
- **Shutdown cancellation:** `contextlib.suppress(asyncio.CancelledError)` after `task.cancel()`
- **Fire-and-forget tasks:** `_task = asyncio.create_task(...); del _task` — the `del` is intentional, keeps a reference long enough to avoid GC but signals it's fire-and-forget (ruff RUF006)

---

## Config Reference

Key settings in `xtts_server/config.py` → `Settings(BaseSettings)`:

| Field | Type | Notes |
|---|---|---|
| `MODEL_PATH` | `str` | Required. Validated at startup — missing files → `sys.exit(1)` |
| `WORKERS_PER_GPU` | `str` | `"1"` or `"2,3,2"`. Parsed into `workers_per_gpu_list: list[int]` |
| `NUM_GPUS` | `int \| None` | Auto-detected via `torch.cuda.device_count()` if None |
| `DEFAULT_LANGUAGE` | `str` | `"tr"`. Used as fallback when request omits language |
| `MAX_QUEUE_SIZE` | `int` | asyncio.Queue maxsize. 503 only when this is full |
| `JOB_TTL_SECONDS` | `int` | Eviction age for DONE/FAILED jobs |

Settings are loaded via `load_settings()` in `main.py` lifespan. It logs all values and validates `MODEL_PATH` before proceeding.

---

## Startup / Shutdown Order

**Startup** (in `main.py` lifespan, order is load-bearing):
1. `load_settings()` — validates config and model files
2. `Dispatcher.start()` — spawns worker processes synchronously (before event loop enters async code — `multiprocessing.Process.start()` is not async-safe)
3. `JobStore`, `SpeakerStore`, `QueueManager` — plain construction
4. `job_store.start()`, `queue_manager.start()`, `dispatcher.start_background_tasks()` — start async background tasks
5. `speaker_store.preload_all()` — warm the RAM cache

**Shutdown** (reverse dependency order):
1. `queue_manager.shutdown()` — stop accepting new work
2. `dispatcher.shutdown()` — send `None` sentinels to workers, join processes
3. `job_store.shutdown()` — cancel TTL cleanup task

---

## app.state Attributes

Everything shared across routers lives on `request.app.state`:

| Attribute | Type | Set in |
|---|---|---|
| `settings` | `Settings` | lifespan |
| `dispatcher` | `Dispatcher` | lifespan |
| `job_store` | `JobStore` | lifespan |
| `speaker_store` | `SpeakerStore` | lifespan |
| `queue_manager` | `QueueManager` | lifespan |
| `start_time` | `float` | lifespan (monotonic) |
| `start_timestamp` | `float` | lifespan (wall clock) |

In WebSocket handlers use `websocket.app.state` (not `request.app.state`).

---

## Ruff

Config in `pyproject.toml`. Target: Python 3.11, line length 100.

```bash
ruff format xtts_server/   # format
ruff check xtts_server/    # lint
ruff check --fix xtts_server/  # lint + auto-fix
```

Active rule sets: `E`, `W`, `F`, `I`, `B`, `UP`, `C4`, `SIM`, `RUF`.
Ignored: `E501` (line length, handled by formatter), `B008` (FastAPI Depends pattern), `SIM108` (ternary), `UP007` (X | Y union syntax in older annotations).

---

## Audio Formats

`audio.py` supports `wav`, `mp3`, `ogg`, `flac`.

- `wav` / `ogg` / `flac` — via `soundfile` (libsndfile)
- `mp3` — via `pydub` + `ffmpeg` (ffmpeg must be on PATH)
- WebSocket streaming always sends raw `float32` PCM at 24 000 Hz mono (no container format)

---

## Known Limitations / Future Work

- **Authentication:** no API key or auth middleware yet. Add it as a FastAPI dependency on the router level.
- **Persistence:** job store is in-memory. Server restart loses all pending/running jobs. Add SQLite or Redis for durability.
- **Speaker update:** no `PUT /v1/speakers/{name}` endpoint. Delete + re-register to update a voice.
- **Long texts:** XTTS-v2 degrades on very long inputs. Consider splitting at sentence boundaries before `model.inference()` and concatenating audio chunks.
- **CPU fallback:** config supports 0 GPUs (single CPU worker), but latency will be high (~10-30× real time). Only suitable for dev/testing.
- **Dockerfile requirements.in path:** the Dockerfile still copies `xtts_server/requirements.in`. Now that `requirements.in` and `requirements.txt` live at the project root, update the `COPY` instruction when rebuilding the image.

---

## macOS Dev Setup

`llvmlite` (via numba → librosa → TTS) does not build from source on macOS without LLVM. Use conda:

```bash
brew install miniconda
conda create -n xtts python=3.11
conda activate xtts
conda install -c conda-forge llvmlite numba librosa
pip install -r requirements.txt
```

For GPU inference you need a Linux machine or Docker with `--gpus all`.

---

## Git / GitHub

- Repo: https://github.com/EngincanVaran/xtts-v2-api-server
- Branch: `main`
- Remote name: `origin`
- `.gitignore` excludes: `__pycache__/`, `.venv/`, `outputs/`, `speakers/`, `logs/`, `*.pth`, `.env`, `.idea/`, `model/`
