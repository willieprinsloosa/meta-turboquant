#!/bin/bash
# ============================================================
# Meta-TurboQuant Installer
# One command to install and run a local AI with tool calling
#
# Usage:
#   bash install.sh            # auto-detect best option
#   bash install.sh --qwen     # force Qwen3 (no Xcode needed)
#   bash install.sh --bonsai   # force Bonsai 1-bit (needs Xcode)
# ============================================================

set -e

BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}>>>${NC} $1"; }
ok()    { echo -e "${GREEN} ✓${NC} $1"; }
warn()  { echo -e "${YELLOW} ⚠${NC} $1"; }
fail()  { echo -e "${RED} ✗${NC} $1"; exit 1; }

echo ""
echo -e "${BOLD}  Meta-TurboQuant Installer${NC}"
echo -e "  Local AI with tool calling on Apple Silicon"
echo -e "  ─────────────────────────────────────────"
echo ""

# ── Parse args ──────────────────────────────────────────────
FORCE_QWEN=false
FORCE_BONSAI=false
if [ "$1" = "--qwen" ]; then FORCE_QWEN=true; fi
if [ "$1" = "--bonsai" ]; then FORCE_BONSAI=true; fi

# ── Step 1: Check platform ─────────────────────────────────
info "Checking system..."

ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" != "Darwin" ]; then
    fail "macOS required (got $OS). Meta-TurboQuant only runs on Mac."
fi

if [ "$ARCH" != "arm64" ]; then
    fail "Apple Silicon required (got $ARCH). M1/M2/M3/M4 Mac needed."
fi

RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
if [ "$RAM_GB" -lt 8 ]; then
    fail "At least 8GB RAM required (got ${RAM_GB}GB)"
fi

ok "macOS on Apple Silicon, ${RAM_GB}GB RAM"

# ── Step 2: Detect Xcode / Metal ────────────────────────────
HAS_METAL=false
if xcrun metal --version &>/dev/null 2>&1; then
    HAS_METAL=true
fi

HAS_XCODE=false
if [ -d "/Applications/Xcode.app" ] || [ -d "/Applications/Xcode-beta.app" ]; then
    HAS_XCODE=true
fi

# ── Step 3: Choose model ────────────────────────────────────
MODEL_TYPE=""

if [ "$FORCE_QWEN" = true ]; then
    MODEL_TYPE="qwen"
elif [ "$FORCE_BONSAI" = true ]; then
    if [ "$HAS_METAL" = false ] && [ "$HAS_XCODE" = false ]; then
        fail "Bonsai requires Xcode (for Metal compiler). Install Xcode from App Store first, or use: bash install.sh --qwen"
    fi
    MODEL_TYPE="bonsai"
else
    # Auto-detect best option
    if [ "$HAS_METAL" = true ]; then
        MODEL_TYPE="bonsai"
        ok "Metal Toolchain found — will install Bonsai 8B (1-bit, 1.3GB)"
    else
        MODEL_TYPE="qwen"
        warn "No Metal Toolchain — will install Qwen3-8B (4-bit, 4.5GB)"
        echo ""
        echo -e "  ${BOLD}Two AI models available:${NC}"
        echo ""
        echo -e "  ${GREEN}Qwen3-8B (recommended, installing now)${NC}"
        echo -e "    Size: 4.5 GB  |  Speed: ~80 tok/s  |  No Xcode needed"
        echo -e "    Better quality, tool calling, 100+ languages"
        echo ""
        echo -e "  ${YELLOW}Bonsai 8B (optional, needs Xcode)${NC}"
        echo -e "    Size: 1.3 GB  |  Speed: ~90 tok/s  |  Needs Xcode from App Store"
        echo -e "    Smallest memory, fastest inference"
        echo ""
    fi
fi

# ── Step 4: Install Homebrew if missing ─────────────────────
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    ok "Homebrew installed"
else
    ok "Homebrew found"
fi

# ── Step 5: Install Python ──────────────────────────────────
if [ "$MODEL_TYPE" = "bonsai" ]; then
    # Bonsai needs Python 3.13 (PrismML builds for 3.13)
    PY=""
    for p in python3.13 /opt/homebrew/bin/python3.13; do
        if [ -f "$p" ] 2>/dev/null && file "$p" 2>/dev/null | grep -q arm64; then
            PY="$p"
            break
        fi
    done
    if [ -z "$PY" ]; then
        info "Installing Python 3.13..."
        brew install python@3.13
        PY="/opt/homebrew/bin/python3.13"
    fi
    ok "Python 3.13 found"
    VENV_DIR=".venv13"
else
    # Qwen works with Python 3.10+
    PY=""
    for p in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10; do
        if [ -f "$p" ] 2>/dev/null && file "$p" 2>/dev/null | grep -q arm64; then
            PY="$p"
            break
        fi
    done
    if [ -z "$PY" ]; then
        info "Installing Python 3.12..."
        brew install python@3.12
        PY="/opt/homebrew/bin/python3.12"
    fi
    ok "Python found: $($PY --version)"
    VENV_DIR=".venv"
fi

# ── Step 6: Clone repo if needed ────────────────────────────
if [ ! -f "serve.py" ]; then
    if [ ! -d "meta-turboquant" ]; then
        info "Downloading Meta-TurboQuant..."
        git clone https://github.com/willieprinsloosa/meta-turboquant.git
        cd meta-turboquant
    else
        cd meta-turboquant
    fi
    ok "Project ready"
else
    ok "Already in project directory"
fi

# ── Step 7: Create venv ─────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python environment..."
    $PY -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q 2>/dev/null
ok "Virtual environment ready ($VENV_DIR)"

# ── Step 8: Install dependencies ────────────────────────────
if [ "$MODEL_TYPE" = "bonsai" ]; then
    # Bonsai: needs PrismML MLX fork
    info "Installing PrismML MLX (1-bit support)..."

    # Ensure Metal Toolchain
    if [ "$HAS_METAL" = false ]; then
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain 2>&1 | tail -1
    fi

    pip install mlx@git+https://github.com/PrismML-Eng/mlx.git@prism mlx-lm numpy -q 2>&1 | tail -3
    ok "PrismML MLX installed (1-bit support)"

    MODEL_NAME="prism-ml/Bonsai-8B-mlx-1bit"
    MODEL_SHORT="Bonsai-8B-mlx-1bit"
    MODEL_SIZE="1.3 GB"
    MODEL_SPEED="~90 tok/s"
else
    # Qwen: standard MLX
    info "Installing MLX..."
    pip install mlx mlx-lm numpy -q 2>&1 | tail -3
    ok "MLX installed"

    MODEL_NAME="Qwen/Qwen3-8B-MLX-4bit"
    MODEL_SHORT="Qwen3-8B-MLX-4bit"
    MODEL_SIZE="4.5 GB"
    MODEL_SPEED="~80 tok/s"
fi

# ── Step 9: Download model ──────────────────────────────────
info "Downloading $MODEL_SHORT ($MODEL_SIZE, first time only)..."
python -c "
from mlx_lm import load
model, tok = load('$MODEL_NAME')
print(f'  {len(model.layers)} layers, ready')
" 2>&1 | grep -v "Fetching\|Warning\|Download"
ok "$MODEL_SHORT downloaded"

# ── Step 10: Quick test ─────────────────────────────────────
info "Testing the AI model..."
python -c "
from mlx_lm import load, generate
model, tokenizer = load('$MODEL_NAME')
r = generate(model, tokenizer, prompt='Say hello in one sentence.', max_tokens=20, verbose=False)
print(f'  AI says: {r.strip()[:80]}')
" 2>&1 | grep -v "Fetching\|Warning"
ok "AI model working"

# ── Done! ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}Model:${NC}  $MODEL_SHORT"
echo -e "  ${BOLD}Size:${NC}   $MODEL_SIZE"
echo -e "  ${BOLD}Speed:${NC}  $MODEL_SPEED"
echo -e "  ${BOLD}Tools:${NC}  Function calling supported"
echo ""
echo -e "  ${BOLD}Start the AI server:${NC}"
echo -e "    cd $(pwd)"
echo -e "    source $VENV_DIR/bin/activate"
echo -e "    python serve.py --lean --model $MODEL_NAME"
echo ""
echo -e "  ${BOLD}Open chat in browser:${NC}"
echo -e "    http://localhost:11434/chat"
echo ""
echo -e "  ${BOLD}Terminal chat:${NC}"
echo -e "    python chat.py --lean --model $MODEL_NAME"
echo ""
echo -e "  ${BOLD}API endpoint:${NC}"
echo -e "    http://localhost:11434/v1/chat/completions"
echo ""

# Offer to start
read -p "  Start the AI server now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    python serve.py --lean --model "$MODEL_NAME"
fi
