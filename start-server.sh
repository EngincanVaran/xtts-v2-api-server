#!/usr/bin/env bash
# start-server.sh — configure and launch the XTTS-v2 API Server on Linux + GPU.
#
# Edit the CONFIGURATION section below, then run:
#   chmod +x start-server.sh
#   ./start-server.sh
#
# Any variable can be overridden from the shell without editing this file:
#   CUDA_VISIBLE_DEVICES=0,1 GPU_MEMORY_FRACTION=0.8 ./start-server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# CONFIGURATION — edit these values to match your deployment
# ===========================================================================

# ---------------------------------------------------------------------------
# Model (required)
# ---------------------------------------------------------------------------
# Absolute path to the local XTTS-v2 model directory.
# Must contain: config.json  model.pth  vocab.json
export MODEL_PATH="${MODEL_PATH:-/path/to/xtts-v2}"

# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------
# Comma-separated list of GPU indices to expose to the server.
# CUDA remaps these to 0, 1, 2, … inside the process, so PyTorch always
# sees a contiguous range starting at 0.
#
#   "0"       → use only GPU 0               (1 physical GPU)
#   "0,1"     → use GPUs 0 and 1             (2 physical GPUs)
#   "0,1,2"   → use GPUs 0, 1, and 2         (3 physical GPUs)
#   ""        → use all available GPUs        (default)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# ---------------------------------------------------------------------------
# GPU memory limit
# ---------------------------------------------------------------------------
# Maximum fraction of each GPU's total VRAM that any single worker process
# may allocate.  Applied via torch.cuda.set_per_process_memory_fraction().
#
#   1.0  → no limit (full VRAM available)
#   0.8  → each worker capped at 80 % of the GPU's total VRAM
#   0.5  → each worker capped at 50 % — useful when sharing a GPU across
#           multiple services or when running 2 workers on one GPU
#
# Must be in the range (0.0, 1.0].  Values outside this range are rejected.
export GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-1.0}"

# ---------------------------------------------------------------------------
# Worker topology
# ---------------------------------------------------------------------------
# Workers per GPU.  Single integer replicates across all visible GPUs.
# Comma-separated list sets per-GPU counts individually.
#
#   "1"     → 1 worker per visible GPU       (recommended default)
#   "2"     → 2 workers per visible GPU      (only if VRAM allows it)
#   "2,1"   → 2 workers on GPU 0, 1 on GPU 1
#
# Each worker loads a full copy of the model.  A single XTTS-v2 model uses
# ~3–4 GB of VRAM; plan accordingly when combining with GPU_MEMORY_FRACTION.
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

# Number of GPUs to register with the dispatcher.  Leave unset to
# auto-detect from CUDA_VISIBLE_DEVICES (strongly recommended).
# Only set manually if auto-detection gives the wrong result.
# export NUM_GPUS=2

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
# BCP-47 code used when a request omits the 'language' field.
# Supported: en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko hi
export DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-tr}"

# ---------------------------------------------------------------------------
# Queue and job store
# ---------------------------------------------------------------------------
export MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-100}"
export JOB_TTL_SECONDS="${JOB_TTL_SECONDS:-300}"

# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------
export SPEAKERS_DIR="${SPEAKERS_DIR:-$SCRIPT_DIR/xtts_server/speakers}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$SCRIPT_DIR/xtts_server/outputs}"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
export MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-5000}"
export SAMPLE_RATE="${SAMPLE_RATE:-24000}"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# ---------------------------------------------------------------------------
# Optional: seed studio speakers on startup
# ---------------------------------------------------------------------------
# Set to "true" to run seed_studio_speakers.py before the server starts.
# Idempotent — existing speakers are skipped unless --force is appended.
SEED_SPEAKERS="${SEED_SPEAKERS:-false}"

# ===========================================================================
# RUNTIME — do not edit below this line
# ===========================================================================

# Minimum version requirements (must match install.sh)
MIN_PYTHON_MAJOR=3;   MIN_PYTHON_MINOR=11
MIN_CUDA_MAJOR=12;    MIN_CUDA_MINOR=1
MIN_TORCH_MAJOR=2;    MIN_TORCHAUDIO_MAJOR=2

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_info()  { printf '\033[0;34m[INFO]\033[0m  %s\n' "$*"; }
_ok()    { printf '\033[0;32m[ OK ]\033[0m  %s\n' "$*"; }
_warn()  { printf '\033[0;33m[WARN]\033[0m  %s\n' "$*"; }
_err()   { printf '\033[0;31m[ERR ]\033[0m  %s\n' "$*" >&2; }
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_sep()   { printf '%s\n' "------------------------------------------------------------"; }

_bold "=== XTTS-v2 API Server — Pre-flight checks ==="
echo ""

# ===========================================================================
# CHECK 1: OS — Linux only
# ===========================================================================
_sep; _info "Operating system …"
OS="$(uname -s)"
if [[ "$OS" != "Linux" ]]; then
    _err "Linux required. Detected: $OS"
    exit 1
fi
_ok "Linux"

# ===========================================================================
# CHECK 2: MODEL_PATH guard
# ===========================================================================
_sep; _info "MODEL_PATH …"

if [[ "$MODEL_PATH" == "/path/to/xtts-v2" ]]; then
    _err "MODEL_PATH is still the placeholder value."
    _err "Edit the MODEL_PATH variable in this script or export it before running:"
    _err "  MODEL_PATH=/your/model ./start-server.sh"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    _err "MODEL_PATH directory does not exist: $MODEL_PATH"
    exit 1
fi

for _f in config.json model.pth vocab.json; do
    if [[ ! -f "$MODEL_PATH/$_f" ]]; then
        _err "Required model file missing: $MODEL_PATH/$_f"
        exit 1
    fi
done
_ok "$MODEL_PATH  (config.json / model.pth / vocab.json present)"

# ===========================================================================
# CHECK 3: NVIDIA driver — nvidia-smi
# ===========================================================================
_sep; _info "NVIDIA GPU driver (nvidia-smi) …"

if ! command -v nvidia-smi &>/dev/null; then
    _err "nvidia-smi not found. Install the NVIDIA GPU driver:"
    _err "  Ubuntu: apt install nvidia-driver-<version>"
    exit 1
fi

# ===========================================================================
# CHECK 4: CUDA version
# ===========================================================================
_sep; _info "CUDA version …"

CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
if [[ -z "$CUDA_VER" ]]; then
    _err "Could not parse CUDA version from nvidia-smi. Check driver installation."
    exit 1
fi
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if [[ "$CUDA_MAJOR" -lt "$MIN_CUDA_MAJOR" ]] || \
   { [[ "$CUDA_MAJOR" -eq "$MIN_CUDA_MAJOR" ]] && [[ "$CUDA_MINOR" -lt "$MIN_CUDA_MINOR" ]]; }; then
    _err "CUDA ${MIN_CUDA_MAJOR}.${MIN_CUDA_MINOR}+ required, detected ${CUDA_VER}."
    exit 1
fi
_ok "CUDA runtime: $CUDA_VER"

# ===========================================================================
# CHECK 5: GPU inventory and CUDA_VISIBLE_DEVICES
# ===========================================================================
_sep; _info "GPU inventory …"

TOTAL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "$TOTAL_GPUS" -eq 0 ]]; then
    _err "nvidia-smi reports 0 GPUs. Check driver installation."
    exit 1
fi

echo ""
echo "  Physical GPUs available:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name drv mem util temp; do
    printf "    GPU %-2s  %-32s  vram=%-8s  util=%-6s  temp=%s\n" \
        "$idx" "$(echo "$name" | xargs)" \
        "$(echo "$mem" | xargs)" \
        "$(echo "$util" | xargs)" \
        "$(echo "$temp" | xargs)"
done
echo ""

# Determine which GPUs will be visible to this server
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE_GPU_COUNT="$TOTAL_GPUS"
    _ok "CUDA_VISIBLE_DEVICES: (unset) → all $TOTAL_GPUS GPU(s) visible"
else
    # Validate each index is within range
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    for _idx in "${GPU_IDS[@]}"; do
        _idx=$(echo "$_idx" | xargs)  # trim whitespace
        if ! [[ "$_idx" =~ ^[0-9]+$ ]]; then
            _err "Invalid GPU index in CUDA_VISIBLE_DEVICES: '$_idx' (must be a non-negative integer)"
            exit 1
        fi
        if [[ "$_idx" -ge "$TOTAL_GPUS" ]]; then
            _err "GPU index $_idx is out of range (system has $TOTAL_GPUS GPU(s): 0–$((TOTAL_GPUS - 1)))."
            exit 1
        fi
    done
    VISIBLE_GPU_COUNT="${#GPU_IDS[@]}"
    _ok "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES → $VISIBLE_GPU_COUNT GPU(s) selected"
fi

# Auto-set NUM_GPUS to match the visible GPU count if not explicitly configured
if [[ -z "${NUM_GPUS:-}" ]]; then
    export NUM_GPUS="$VISIBLE_GPU_COUNT"
    _info "NUM_GPUS auto-set to $NUM_GPUS"
else
    if [[ "$NUM_GPUS" -ne "$VISIBLE_GPU_COUNT" ]]; then
        _warn "NUM_GPUS=$NUM_GPUS but $VISIBLE_GPU_COUNT GPU(s) are visible via CUDA_VISIBLE_DEVICES."
        _warn "Forcing NUM_GPUS=$VISIBLE_GPU_COUNT to avoid dispatcher misconfiguration."
        export NUM_GPUS="$VISIBLE_GPU_COUNT"
    fi
    _ok "NUM_GPUS: $NUM_GPUS"
fi

# ===========================================================================
# CHECK 6: GPU_MEMORY_FRACTION
# ===========================================================================
_sep; _info "GPU memory fraction …"

# Validate that it's a float in (0.0, 1.0]
if ! python3 -c "
import sys
try:
    v = float('${GPU_MEMORY_FRACTION}')
except ValueError:
    print(f'Not a valid number: ${GPU_MEMORY_FRACTION}', file=sys.stderr)
    sys.exit(1)
if not (0.0 < v <= 1.0):
    print(f'Must be in range (0.0, 1.0], got {v}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
    _err "GPU_MEMORY_FRACTION='${GPU_MEMORY_FRACTION}' is invalid."
    _err "Must be a decimal number in the range (0.0, 1.0]."
    _err "Examples: 1.0 (no limit)  0.8 (80%)  0.5 (50%)"
    exit 1
fi

if [[ "$GPU_MEMORY_FRACTION" == "1.0" ]] || [[ "$GPU_MEMORY_FRACTION" == "1" ]]; then
    _ok "GPU_MEMORY_FRACTION: 1.0 (no limit)"
else
    _ok "GPU_MEMORY_FRACTION: $GPU_MEMORY_FRACTION  (each worker capped at $(python3 -c "print(f'{float(\"${GPU_MEMORY_FRACTION}\")*100:.0f}%')") of GPU VRAM)"
fi

# ===========================================================================
# CHECK 7: Python interpreter
# ===========================================================================
_sep; _info "Python interpreter …"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    _err "Venv not found at $SCRIPT_DIR/.venv/"
    _err "Run ./install.sh first."
    exit 1
fi

PYTHON_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ]] || \
   { [[ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ]] && [[ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]]; }; then
    _err "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required inside venv, found $PYTHON_VER."
    _err "Delete .venv/ and re-run ./install.sh with Python 3.11."
    exit 1
fi
_ok "Python: $PYTHON_VER  ($PYTHON)"

# ===========================================================================
# CHECK 8: PyTorch + CUDA availability
# ===========================================================================
_sep; _info "PyTorch …"

TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCH_VER" ]]; then
    _err "torch not importable. Run ./install.sh to install dependencies."
    exit 1
fi
TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1)
if [[ "$TORCH_MAJOR" -lt "$MIN_TORCH_MAJOR" ]]; then
    _err "PyTorch ${MIN_TORCH_MAJOR}.0+ required, found $TORCH_VER."
    exit 1
fi
_ok "torch: $TORCH_VER"

TORCHAUDIO_VER=$("$PYTHON" -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCHAUDIO_VER" ]]; then
    _warn "torchaudio not importable — audio features may be limited."
else
    _ok "torchaudio: $TORCHAUDIO_VER"
fi

_sep; _info "PyTorch CUDA …"

CUDA_AVAILABLE=$("$PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$CUDA_AVAILABLE" != "True" ]]; then
    _err "torch.cuda.is_available() returned False."
    _err "Ensure the CUDA-enabled build of PyTorch is installed."
    _err "  Check: https://pytorch.org/get-started/locally/"
    exit 1
fi

TORCH_CUDA_VER=$("$PYTHON" -c "import torch; print(torch.version.cuda or 'n/a')" 2>/dev/null || echo "n/a")
TORCH_GPU_COUNT=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

# The count PyTorch sees respects CUDA_VISIBLE_DEVICES (already exported above).
if [[ "$TORCH_GPU_COUNT" -eq 0 ]]; then
    _err "torch sees 0 GPUs. Verify CUDA_VISIBLE_DEVICES and driver compatibility."
    exit 1
fi

if [[ "$TORCH_GPU_COUNT" -ne "$NUM_GPUS" ]]; then
    _warn "torch.cuda.device_count()=$TORCH_GPU_COUNT but NUM_GPUS=$NUM_GPUS."
    _warn "Adjusting NUM_GPUS to $TORCH_GPU_COUNT."
    export NUM_GPUS="$TORCH_GPU_COUNT"
fi

_ok "torch CUDA build: $TORCH_CUDA_VER  |  GPU(s) visible to torch: $TORCH_GPU_COUNT"

# VRAM summary for visible GPUs
echo ""
echo "  VRAM per visible GPU:"
"$PYTHON" -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    vram_gb = p.total_memory / 1024**3
    print(f'    GPU {i}  {p.name}  {vram_gb:.1f} GB total')
"
echo ""

# ===========================================================================
# Optional: seed studio speakers
# ===========================================================================
if [[ "$SEED_SPEAKERS" == "true" ]]; then
    _sep; _info "Seeding studio speakers …"
    "$PYTHON" "$SCRIPT_DIR/seed_studio_speakers.py" \
        --model-path "$MODEL_PATH" \
        --speakers-dir "$SPEAKERS_DIR"
    _ok "Speaker seeding complete."
fi

# ===========================================================================
# Launch summary
# ===========================================================================
_sep
_bold "=== All checks passed — starting server ==="
echo ""
printf "  %-20s %s\n" "Python:"            "$PYTHON_VER  ($PYTHON)"
printf "  %-20s %s\n" "MODEL_PATH:"        "$MODEL_PATH"
printf "  %-20s %s\n" "CUDA:"              "$CUDA_VER  (torch build: $TORCH_CUDA_VER)"
printf "  %-20s %s\n" "Visible GPUs:"      "$NUM_GPUS  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"
printf "  %-20s %s\n" "Workers/GPU:"       "$WORKERS_PER_GPU"
printf "  %-20s %s\n" "GPU mem fraction:"  "$GPU_MEMORY_FRACTION"
printf "  %-20s %s\n" "Default language:"  "$DEFAULT_LANGUAGE"
printf "  %-20s %s\n" "Queue size:"        "$MAX_QUEUE_SIZE"
printf "  %-20s %s\n" "Job TTL:"           "${JOB_TTL_SECONDS}s"
printf "  %-20s %s\n" "Speakers dir:"      "$SPEAKERS_DIR"
printf "  %-20s %s\n" "Outputs dir:"       "$OUTPUTS_DIR"
printf "  %-20s %s\n" "Bind:"              "$HOST:$PORT"
printf "  %-20s %s\n" "Log level:"         "$LOG_LEVEL"
echo ""

# The server reads .env from its CWD (xtts_server/), but env vars take
# priority over .env — so everything exported above overrides the file.
cd "$SCRIPT_DIR/xtts_server"
exec "$PYTHON" main.py