#!/usr/bin/env bash
# install.sh — set up the XTTS-v2 API Server on a Linux machine with NVIDIA GPUs.
#
# What this script does:
#   1. Verifies OS, GPU driver, CUDA, and Python requirements.
#   2. Creates a Python 3.11 venv at .venv/ and installs all dependencies.
#   3. Verifies that PyTorch can see CUDA after install.
#   4. Creates xtts_server/.env from .env.example if it does not already exist.
#   5. Creates output and speaker directories.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After install, edit xtts_server/.env (set at least MODEL_PATH) then run:
#   ./start-server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Minimum version requirements
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11
MIN_CUDA_MAJOR=12
MIN_CUDA_MINOR=1
MIN_TORCH_MAJOR=2
MIN_TORCHAUDIO_MAJOR=2

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_info()  { printf '\033[0;34m[INFO]\033[0m  %s\n' "$*"; }
_ok()    { printf '\033[0;32m[ OK ]\033[0m  %s\n' "$*"; }
_warn()  { printf '\033[0;33m[WARN]\033[0m  %s\n' "$*"; }
_err()   { printf '\033[0;31m[ERR ]\033[0m  %s\n' "$*" >&2; }
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_sep()   { printf '%s\n' "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Helper: compare two dot-separated version strings  (0 = first >= second)
# ---------------------------------------------------------------------------
_version_ge() {
    # Returns 0 (success) if $1 >= $2
    python3 -c "
import sys
a = tuple(int(x) for x in '$1'.split('.') if x.isdigit())
b = tuple(int(x) for x in '$2'.split('.') if x.isdigit())
sys.exit(0 if a >= b else 1)
"
}

_bold "=== XTTS-v2 API Server — Install ==="
echo "Project root : $SCRIPT_DIR"
echo ""

# ===========================================================================
# 1. OS check — Linux only
# ===========================================================================
_sep; _info "Checking operating system …"

OS="$(uname -s)"
if [[ "$OS" != "Linux" ]]; then
    _err "This script requires Linux. Detected: $OS"
    _err "For macOS development see the README (conda + CPU-only)."
    exit 1
fi
_ok "OS: Linux"

# ===========================================================================
# 2. GPU driver — nvidia-smi
# ===========================================================================
_sep; _info "Checking NVIDIA GPU driver (nvidia-smi) …"

if ! command -v nvidia-smi &>/dev/null; then
    _err "nvidia-smi not found. Install the NVIDIA GPU driver and re-run."
    _err "  Ubuntu: apt install nvidia-driver-<version>"
    exit 1
fi

# Show full GPU inventory
echo ""
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name drv mem cap; do
    printf "  GPU %-2s  %-32s  driver=%-12s  vram=%-8s  compute=%s\n" \
        "$idx" "$(echo "$name" | xargs)" "$(echo "$drv" | xargs)" \
        "$(echo "$mem" | xargs)" "$(echo "$cap" | xargs)"
done
echo ""

TOTAL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "$TOTAL_GPUS" -eq 0 ]]; then
    _err "nvidia-smi found but reported 0 GPUs. Check driver installation."
    exit 1
fi
_ok "GPU count: $TOTAL_GPUS"

# ===========================================================================
# 3. CUDA version
# ===========================================================================
_sep; _info "Checking CUDA version …"

# CUDA runtime version (from nvidia-smi header)
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
if [[ -z "$CUDA_VER" ]]; then
    _err "Could not parse CUDA version from nvidia-smi output."
    exit 1
fi
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
_ok "CUDA runtime: $CUDA_VER (via nvidia-smi)"

if [[ "$CUDA_MAJOR" -lt "$MIN_CUDA_MAJOR" ]] || \
   { [[ "$CUDA_MAJOR" -eq "$MIN_CUDA_MAJOR" ]] && [[ "$CUDA_MINOR" -lt "$MIN_CUDA_MINOR" ]]; }; then
    _err "CUDA ${MIN_CUDA_MAJOR}.${MIN_CUDA_MINOR}+ required, detected ${CUDA_VER}."
    _err "Update your NVIDIA driver to get a newer CUDA runtime."
    exit 1
fi

# CUDA toolkit (nvcc) — optional; only the runtime is strictly required.
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    _ok "CUDA toolkit (nvcc): $NVCC_VER"
else
    _warn "nvcc not found — CUDA toolkit not installed, only the runtime is present."
    _warn "  This is fine for inference. Install cuda-toolkit if you need to compile CUDA extensions."
fi

# ===========================================================================
# 4. Python version
# ===========================================================================
_sep; _info "Checking Python version …"

if ! command -v python3 &>/dev/null; then
    _err "python3 not found. Install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+."
    _err "  Ubuntu: apt install python3.11 python3.11-venv"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ]] || \
   { [[ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ]] && [[ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]]; }; then
    _err "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, detected ${PYTHON_VER}."
    _err "  Ubuntu: apt install python3.11 python3.11-venv"
    exit 1
fi
_ok "Python: $PYTHON_VER  ($(which python3))"

# Check python3-venv module
if ! python3 -c "import venv" &>/dev/null; then
    _err "python3-venv module not found."
    _err "  Ubuntu: apt install python3.11-venv"
    exit 1
fi

# ===========================================================================
# 5. pip
# ===========================================================================
_sep; _info "Checking pip …"

PIP_VER=$(python3 -m pip --version 2>/dev/null | awk '{print $2}' || echo "")
if [[ -z "$PIP_VER" ]]; then
    _err "pip not found. Install it with: python3 -m ensurepip --upgrade"
    exit 1
fi
_ok "pip: $PIP_VER"

# ===========================================================================
# 6. Create venv
# ===========================================================================
_sep; _info "Setting up Python virtual environment …"

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    _warn "Venv already exists at $VENV_DIR — skipping creation."
    _warn "Delete .venv/ and re-run to start fresh."
else
    _info "Creating venv at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
    _ok "Venv created."
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

"$PIP" install --upgrade pip --quiet
_ok "pip upgraded inside venv: $("$PIP" --version | awk '{print $2}')"

# ===========================================================================
# 7. Install dependencies
# ===========================================================================
_sep; _info "Installing dependencies from requirements.txt …"
_info "This may take several minutes (PyTorch, TTS, etc.)."
echo ""

REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
if [[ ! -f "$REQUIREMENTS" ]]; then
    _err "requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

"$PIP" install -r "$REQUIREMENTS"
_ok "Dependencies installed."

# ===========================================================================
# 8. Post-install verification — PyTorch + CUDA
# ===========================================================================
_sep; _info "Verifying PyTorch and CUDA availability …"

# torch
TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCH_VER" ]]; then
    _err "torch could not be imported after install. Check requirements.txt."
    exit 1
fi
TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1)
if [[ "$TORCH_MAJOR" -lt "$MIN_TORCH_MAJOR" ]]; then
    _err "PyTorch ${MIN_TORCH_MAJOR}.0+ required, found $TORCH_VER."
    exit 1
fi
_ok "torch: $TORCH_VER"

# torchaudio
TORCHAUDIO_VER=$("$PYTHON" -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCHAUDIO_VER" ]]; then
    _warn "torchaudio could not be imported — some audio features may be unavailable."
else
    TORCHAUDIO_MAJOR=$(echo "$TORCHAUDIO_VER" | cut -d. -f1)
    if [[ "$TORCHAUDIO_MAJOR" -lt "$MIN_TORCHAUDIO_MAJOR" ]]; then
        _warn "torchaudio ${MIN_TORCHAUDIO_MAJOR}.0+ recommended, found $TORCHAUDIO_VER."
    else
        _ok "torchaudio: $TORCHAUDIO_VER"
    fi
fi

# CUDA in torch
CUDA_AVAILABLE=$("$PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$CUDA_AVAILABLE" != "True" ]]; then
    _err "torch.cuda.is_available() returned False."
    _err "Ensure you installed the CUDA-enabled build of PyTorch."
    _err "  Check: https://pytorch.org/get-started/locally/"
    exit 1
fi

TORCH_CUDA_VER=$("$PYTHON" -c "import torch; print(torch.version.cuda or 'n/a')" 2>/dev/null || echo "n/a")
TORCH_GPU_COUNT=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
_ok "torch CUDA build: $TORCH_CUDA_VER  |  visible GPUs: $TORCH_GPU_COUNT"

if [[ "$TORCH_GPU_COUNT" -eq 0 ]]; then
    _err "torch sees 0 GPUs. Check driver compatibility with this PyTorch CUDA build."
    exit 1
fi

# ===========================================================================
# 9. Create xtts_server/.env from .env.example if absent
# ===========================================================================
_sep; _info "Checking environment configuration …"

ENV_FILE="$SCRIPT_DIR/xtts_server/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        _warn ".env created at $ENV_FILE from .env.example"
        _warn "Edit it and set MODEL_PATH before starting the server."
    else
        _warn "No .env.example found. Create $ENV_FILE manually with at least MODEL_PATH."
    fi
else
    _ok ".env already exists at $ENV_FILE"
fi

# ===========================================================================
# 10. Create runtime directories
# ===========================================================================
mkdir -p "$SCRIPT_DIR/xtts_server/speakers"
mkdir -p "$SCRIPT_DIR/xtts_server/outputs"
_ok "Runtime directories ready (speakers/, outputs/)"

# ===========================================================================
# Done
# ===========================================================================
_sep
_bold "=== Install complete ==="
echo ""
echo "  Python venv : $VENV_DIR"
echo "  GPUs found  : $TOTAL_GPUS"
echo "  CUDA        : $CUDA_VER"
echo "  torch       : $TORCH_VER (CUDA $TORCH_CUDA_VER)"
echo ""
_bold "Next steps:"
echo ""
echo "  1. Edit $ENV_FILE"
echo "     Set MODEL_PATH to your local XTTS-v2 model directory."
echo ""
echo "  2. (Optional) Seed the 58 built-in studio speakers:"
echo "     .venv/bin/python seed_studio_speakers.py \\"
echo "       --model-path \$MODEL_PATH --speakers-dir ./xtts_server/speakers"
echo ""
echo "  3. Start the server:"
echo "     ./start-server.sh"
echo ""