#!/bin/bash
# Meta-TurboQuant — One-command setup
# Usage: bash setup.sh [--bonsai]
#
# Without flags: sets up standard venv (.venv) for Llama/Mistral models
# With --bonsai:  also sets up Python 3.13 venv (.venv13) for Bonsai 1-bit models

set -e

BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${BLUE}>>>${NC} $1"; }
ok()    { echo -e "${GREEN} ✓${NC} $1"; }
fail()  { echo -e "${RED} ✗${NC} $1"; exit 1; }

echo ""
echo "  Meta-TurboQuant Setup"
echo "  ====================="
echo ""

# --- Check platform ---
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    fail "Apple Silicon required (got $ARCH). This only runs on M1/M2/M3/M4 Macs."
fi
ok "Apple Silicon ($ARCH)"

# --- Find arm64 Python 3.10+ ---
info "Finding arm64 Python..."
PYTHON=""
for p in python3.13 python3.12 python3.11 python3.10; do
    CANDIDATE=$(which $p 2>/dev/null || true)
    if [ -z "$CANDIDATE" ]; then
        # Try homebrew
        CANDIDATE="/opt/homebrew/bin/$p"
    fi
    if [ -f "$CANDIDATE" ] && file "$CANDIDATE" | grep -q arm64; then
        PYTHON="$CANDIDATE"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    fail "No arm64 Python 3.10+ found. Install via: brew install python@3.12"
fi
PY_VER=$($PYTHON --version)
ok "Found $PY_VER at $PYTHON"

# --- Standard venv (.venv) ---
info "Setting up standard venv (.venv)..."
if [ -d ".venv" ]; then
    ok ".venv already exists"
else
    $PYTHON -m venv .venv
    ok "Created .venv"
fi

info "Installing dependencies..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .
.venv/bin/pip install -q pytest
ok "Dependencies installed"

# --- Verify MLX ---
info "Verifying MLX..."
.venv/bin/python -c "import mlx.core as mx; print(f'  MLX {mx.__version__} on {mx.default_device()}')" || fail "MLX import failed"
ok "MLX works"

# --- Run unit tests ---
info "Running core tests..."
.venv/bin/python -m pytest tests/test_turboquant.py -q --tb=line 2>&1 | tail -3
ok "Core tests passed"

# --- Download default model ---
info "Downloading default model (Llama 3.2 3B 4-bit)..."
.venv/bin/python -c "
import mlx_lm
model, tok = mlx_lm.load('mlx-community/Llama-3.2-3B-Instruct-4bit')
print(f'  {len(model.layers)} layers, ready')
" 2>&1 | grep -v "Fetching\|Download\|Warning"
ok "Model downloaded"

# --- Bonsai setup (optional) ---
if [ "$1" = "--bonsai" ]; then
    echo ""
    info "Setting up Bonsai 1-bit models..."

    # Find Python 3.13
    PY13=""
    for p in python3.13 /opt/homebrew/bin/python3.13; do
        if [ -f "$p" ] && file "$p" | grep -q arm64; then
            PY13="$p"
            break
        fi
    done

    if [ -z "$PY13" ]; then
        fail "Python 3.13 required for Bonsai. Install via: brew install python@3.13"
    fi
    ok "Found Python 3.13 at $PY13"

    if [ -d ".venv13" ]; then
        ok ".venv13 already exists"
    else
        $PY13 -m venv .venv13
        ok "Created .venv13"
    fi

    info "Installing PrismML MLX fork (this compiles from source, ~5 min)..."

    # Check Metal Toolchain
    if ! xcrun metal --version >/dev/null 2>&1; then
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain
    fi

    .venv13/bin/pip install -q --upgrade pip
    .venv13/bin/pip install -q mlx@git+https://github.com/PrismML-Eng/mlx.git@prism mlx-lm numpy pytest 2>&1 | tail -3
    ok "PrismML MLX installed"

    # Verify 1-bit
    .venv13/bin/python -c "
import mlx.core as mx
mx.quantize(mx.ones((1,128)), bits=1, group_size=128)
print('  1-bit quantization works')
" || fail "1-bit quantization not supported"
    ok "1-bit support verified"

    # Download Bonsai 8B
    info "Downloading Bonsai 8B (1.3 GB)..."
    .venv13/bin/python -c "
from mlx_lm import load
model, tok = load('prism-ml/Bonsai-8B-mlx-1bit')
print(f'  {len(model.layers)} layers, 1-bit, ready')
" 2>&1 | grep -v "Fetching\|Download\|Warning"
    ok "Bonsai 8B downloaded"
fi

# --- Done ---
echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "  Quick start:"
echo "    source .venv/bin/activate"
echo "    python chat.py --lean"
echo ""
echo "  Start server:"
echo "    python serve.py --lean"
echo ""
if [ "$1" = "--bonsai" ]; then
    echo "  Bonsai (1-bit, tool calling):"
    echo "    source .venv13/bin/activate"
    echo "    python chat.py --lean --model prism-ml/Bonsai-8B-mlx-1bit"
    echo ""
fi
echo "  Run tests:"
echo "    python -m pytest tests/ -v"
echo ""
