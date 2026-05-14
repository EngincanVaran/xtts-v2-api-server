# QUAKER-XTTS API Server — Full Context Dump

> Generated 2026-05-14. Use this file to restore full context in a new session.
> Contains: developer memory, architecture decisions, and session history.

---

## SECTION 1 — DEVELOPER MEMORY

### User preferences
- Write good inline comments explaining non-obvious logic.
- Include README.md and CLAUDE.md updates at the end of any build phase.
- Keep responses short and concise — no trailing summaries.
- User is the project owner / solo developer building this for production use.

### Project background
- Goal: serve 20+ concurrent TTS clients with multi-GPU support.
- Redis removed by user request — job store is in-memory only.
- GitHub repo: https://github.com/EngincanVaran/xtts-v2-api-server
- Branches: `main` (stable), `development` (active dev/test — current)
- macOS dev machine (Intel, no CUDA). GPU inference requires Linux / Docker.
- Active venv: `.venv-coqui` (Python 3.11, coqui-tts==0.27.1, no encodec)

### How to run the server (macOS dev)
```bash
cd xtts_server
MODEL_PATH=../model DEFAULT_LANGUAGE=tr WORKERS_PER_GPU=1 ../.venv-coqui/bin/python main.py
```
Logs go to both stdout and `xtts_server/logs/xtts_server.log` (rotating).
Server ready in ~2–3 s (API layer); worker model load takes ~70 s on CPU.
Watch for `Worker cpu-w0 — ready` in the log before sending requests.

### How to run the server (Linux / GPU production)
```bash
chmod +x install.sh start-server.sh
./install.sh           # one-time: creates .venv, runs pip-compile, installs deps
./start-server.sh      # every time: pre-flight checks + exec python main.py
```

### Key architectural decisions
| Decision | Why |
|---|---|
| `model.inference()` not `model.synthesize()` | synthesize() re-encodes reference WAV every call (100–500 ms overhead) |
| `multiprocessing.get_context("spawn")` | CUDA cannot survive `fork()` |
| Manager proxy queues for result queues | spawn-ctx Queues can't be pickled after process start (Python 3.11) |
| No Redis | User request; in-memory dict + asyncio.Lock sufficient for single-process |
| `asyncio.Condition` created lazily | must be created inside a running event loop |
| Speaker latents as CPU numpy `.npz` | GPU tensors are not picklable across process boundaries |
| `speaker_embedding` shape MUST be `(1, 512, 1)` | model's conv1d expects `(N, C_in=512, L=1)`; `(1, 512)` crashes inference |
| Audio deleted after first download | one-shot via `BackgroundTask`; 410 Gone on second request |
| `audio_to_bytes` in `run_in_executor` | sync endpoint must not block event loop during audio encoding |
| `coqui-tts==0.27.1 --no-deps` | encodec dep has license restrictions at user's workplace; all other deps listed explicitly in requirements.in |
| `pip-compile` runs inside `install.sh` | only `requirements.in` needs to exist on a fresh machine; lockfile generated at install time |

### Work completed in this session (2026-05-14)
1. **QUAKER-XTTS ASCII art banner** — `_log_banner()` in `main.py`, 6-row block letter art
2. **HTTP access log middleware** — one line per request, `X-Request-ID` correlation header
3. **Observability improvements** — RTF, synthesis_ms, audio_s, lifecycle timing, first-chunk latency, WebSocket stream metrics across all routers
4. **GPU_MEMORY_FRACTION** — `torch.cuda.set_per_process_memory_fraction()` in worker before model load; validated + exported by `start-server.sh`
5. **CUDA_VISIBLE_DEVICES** — GPU selection via standard env var, validated in `start-server.sh`
6. **`install.sh`** — Linux-only setup: OS/CUDA/Python checks, pip-tools, pip-compile, two-step coqui-tts install, torch/CUDA verification
7. **`start-server.sh`** — GPU-aware launch: pre-flight checks, optional speaker seeding, `exec python main.py`
8. **coqui-tts==0.27.1 migration** — replaced `TTS==0.22.0` with `coqui-tts==0.27.1 --no-deps`; all deps listed explicitly in `requirements.in` except encodec
9. **`.venv-coqui`** — fresh venv created and verified: server starts cleanly with coqui-tts==0.27.1
10. **`development` branch** — created from `main` for ongoing test/dev work
11. **`.gitignore`** — added `.venv-*/` pattern to cover named venvs

### Known limitations (open)
- No authentication / API key middleware
- Job store is in-memory — server restart loses all jobs
- No `PUT /v1/speakers/{name}` — must delete + re-register to update a voice
- Long texts degrade XTTS-v2 — consider sentence-splitting before inference
- CPU-only latency: ~10–30× real time (fine for dev/testing only)
- Dockerfile `COPY` path for requirements needs updating (deps now at project root)

---

## SECTION 2 — ARCHITECTURE REFERENCE

### Project layout
```
xtts-v2-api-server/
├── install.sh            # Linux-only setup: pip-tools + pip-compile + two-step coqui-tts install
├── start-server.sh       # GPU-aware launch script (CUDA_VISIBLE_DEVICES, GPU_MEMORY_FRACTION)
├── requirements.in       # hand-edited direct deps — only file needed before install
├── requirements.txt      # generated lockfile (produced by install.sh via pip-compile)
├── pyproject.toml        # ruff config
├── Dockerfile
├── .env.example
├── seed_studio_speakers.py
└── xtts_server/
    ├── main.py           # FastAPI app, lifespan, banner, access log middleware
    ├── config.py         # Pydantic settings, MODEL_PATH validation
    ├── dispatcher.py     # Worker pool, least-loaded routing, active job tracking
    ├── queue_manager.py  # asyncio queue, drain loop, lifecycle timing
    ├── job_store.py      # In-memory job state, TTL cleanup, clear_audio_path()
    ├── worker.py         # XTTS-v2 model process; GPU_MEMORY_FRACTION; inference/stream/latents
    ├── speakers.py       # SpeakerStore; disk + RAM cache; preload_all()
    ├── audio.py          # PCM → WAV/MP3/OGG/FLAC
    ├── logging_config.py # Structured logging, rotating file handler
    ├── routers/
    │   ├── system.py     # GET /health, GET /v1/system/info
    │   ├── tts.py        # POST /v1/tts, POST /v1/tts/sync (audio_to_bytes in executor), GET /v1/tts/{id}/audio
    │   ├── clone.py      # POST /v1/clone; latents_ms, total_ms logging
    │   ├── batch.py      # POST /v1/batch; formats log, accepted preview log
    │   ├── jobs.py       # GET /v1/jobs/{id} (hides audio_url after download), GET /v1/jobs
    │   └── speakers.py   # GET/DELETE /v1/speakers
    └── ws/
        └── stream.py     # WS /v1/stream; first-chunk latency, RTF, total_bytes
```

### API surface
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","uptime_s":…}` — logged at DEBUG |
| `GET` | `/v1/system/info` | Worker stats, active jobs, queue depth, GPU VRAM |
| `POST` | `/v1/tts` | Async job → `202 {job_id, poll_url}` |
| `POST` | `/v1/tts/sync` | Blocks, returns audio file directly |
| `GET` | `/v1/tts/{job_id}/audio` | One-shot download; `410` on second request |
| `GET` | `/v1/jobs/{job_id}` | Poll status + timing; hides `audio_url` after download |
| `GET` | `/v1/jobs` | List all jobs (debug) |
| `POST` | `/v1/batch` | Up to 50 TTS items; all-or-nothing validation |
| `POST` | `/v1/clone` | Upload ref audio → register speaker (latents_ms logged) |
| `GET` | `/v1/speakers` | List speakers |
| `GET` | `/v1/speakers/{name}` | Speaker metadata |
| `DELETE` | `/v1/speakers/{name}` | Evict cache + rmtree |
| `WS` | `/v1/stream` | Raw float32 PCM chunks, 24 000 Hz mono |

### Dependency install sequence (enforced by install.sh)
```
1. pip install pip-tools
2. pip-compile requirements.in -o requirements.txt
3. pip install coqui-tts==0.27.1 --no-deps   ← encodec excluded
4. pip install -r requirements.txt             ← everything else
```

### Key env vars
| Variable | Default | Where set | Where read |
|---|---|---|---|
| `MODEL_PATH` | required | `.env` / shell | `config.py` |
| `WORKERS_PER_GPU` | `"1"` | `.env` / shell | `config.py` |
| `CUDA_VISIBLE_DEVICES` | all GPUs | `start-server.sh` | PyTorch (automatic) |
| `GPU_MEMORY_FRACTION` | `"1.0"` | `start-server.sh` (exported) | `worker.py _apply_memory_fraction()` |
| `DEFAULT_LANGUAGE` | `"tr"` | `.env` / shell | `config.py` |
| `MAX_QUEUE_SIZE` | `100` | `.env` / shell | `queue_manager.py` |
| `JOB_TTL_SECONDS` | `300` | `.env` / shell | `job_store.py` |

### Logging structure
- Access log: `[method] [path] → [status] | [ms] | client=[ip] | req=[X-Request-ID]`
- `/health` logged at DEBUG; all other routes at INFO
- Per-synthesis: `inference done | job=[id] | ms=[ms] | audio_s=[s] | RTF=[rtf]`
- WebSocket: `first chunk | latency_ms=[ms]`, `stream done | rtf=[rtf] | first_chunk_ms=[ms]`
- Job lifecycle: `lifecycle complete | synthesis_ms=[ms] | total_ms=[ms]` (queue wait + synth + save)
- Worker status (every 60 s): active jobs per worker + elapsed time, queue depth

### Critical invariants
- **Never** use `multiprocessing.Queue()` for result queues — always `dispatcher.make_queue()` (Manager proxy)
- **Never** pass GPU tensors between processes — CPU numpy only
- **Never** block the event loop — all I/O in `run_in_executor`
- `speaker_embedding` shape must be `(1, 512, 1)` — not `(1, 512)` (conv1d channel mismatch)
- `asyncio.Condition` created lazily — not in `__init__`
- `dispatcher.release()` called **before** `on_complete()` — frees slot immediately
- `encodec` must NOT be installed — excluded intentionally from requirements.in

---

## SECTION 3 — SESSION COMMIT HISTORY (2026-05-14)

### main branch
| Commit | Description |
|---|---|
| `3366cb9` | Update README: worker observability and one-shot audio download |
| `0ab0b4f` | Improve worker observability, fix total_requests, delete audio after download |
| `a4bd537` | Fix dispatcher shutdown: await cancelled _status_task |
| `54fd708` | Update README and CLAUDE.md: sync endpoint + studio speakers |
| `3ef4257` | Fix studio speaker embedding shape: (1,512,1) not (1,512) |
| `3fa61ea` | Add observability, access logging, GPU controls, QUAKER-XTTS banner |
| `2c0c6d3` | Update README and CLAUDE.md: scripts, GPU controls, access log, banner |

### development branch (branched from main at 2c0c6d3)
| Commit | Description |
|---|---|
| `3f67b35` | Switch to coqui-tts==0.27.1 --no-deps; list all deps explicitly (no encodec) |
| `5518b99` | Regenerate requirements.txt for coqui-tts==0.27.1 |
| `38fd9ae` | install.sh: install pip-tools and run pip-compile before pip install |
| `0966194` | Update README and CLAUDE.md: document pip-tools + pip-compile in install.sh |
| `6c2cdd8` | Regenerate requirements.txt with coqui-tts venv; ignore .venv-* dirs |
