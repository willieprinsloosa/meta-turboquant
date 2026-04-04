#!/bin/bash
# ============================================================
# Meta-TurboQuant Installer
# One command to install and run a local AI with tool calling
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/willieprinsloosa/meta-turboquant/main/install.sh | bash
#
#   Or if already cloned:
#   bash install.sh
# ============================================================

set -e

BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

info()  { echo -e "${BLUE}>>>${NC} $1"; }
ok()    { echo -e "${GREEN} ✓${NC} $1"; }
warn()  { echo -e "${YELLOW} ⚠${NC} $1"; }
fail()  { echo -e "${RED} ✗${NC} $1"; exit 1; }

echo ""
echo -e "${BOLD}  Meta-TurboQuant Installer${NC}"
echo -e "  ${DIM}Local AI with 1-bit models + tool calling${NC}"
echo -e "  ─────────────────────────────────────────"
echo ""

# ── Step 1: Check platform ──────────────────────────────────
info "Checking system..."

ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" != "Darwin" ]; then
    fail "macOS required (got $OS). Meta-TurboQuant only runs on Mac."
fi

if [ "$ARCH" != "arm64" ]; then
    fail "Apple Silicon required (got $ARCH). M1/M2/M3/M4 Mac needed."
fi

ok "macOS on Apple Silicon ($ARCH)"

# Check RAM
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
if [ "$RAM_GB" -lt 8 ]; then
    fail "At least 8GB RAM required (got ${RAM_GB}GB)"
fi
ok "${RAM_GB}GB RAM"

# ── Step 2: Install Homebrew if missing ─────────────────────
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    ok "Homebrew installed"
else
    ok "Homebrew found"
fi

# ── Step 3: Install Python 3.13 if missing ──────────────────
PY13=""
for p in python3.13 /opt/homebrew/bin/python3.13; do
    if [ -f "$p" ] 2>/dev/null && file "$p" 2>/dev/null | grep -q arm64; then
        PY13="$p"
        break
    fi
done

if [ -z "$PY13" ]; then
    info "Installing Python 3.13..."
    brew install python@3.13
    PY13="/opt/homebrew/bin/python3.13"
    ok "Python 3.13 installed"
else
    ok "Python 3.13 found at $PY13"
fi

# ── Step 4: Install Xcode Command Line Tools ────────────────
if ! xcode-select -p &>/dev/null; then
    info "Installing Xcode Command Line Tools..."
    xcode-select --install 2>/dev/null || true
    echo "  Waiting for Xcode tools installation..."
    until xcode-select -p &>/dev/null; do sleep 5; done
    ok "Xcode Command Line Tools installed"
else
    ok "Xcode Command Line Tools found"
fi

# ── Step 5: Install Metal Toolchain ─────────────────────────
if ! xcrun metal --version &>/dev/null 2>&1; then
    info "Installing Metal Toolchain (needed for 1-bit AI models)..."
    xcodebuild -downloadComponent MetalToolchain 2>&1 | tail -1
    ok "Metal Toolchain installed"
else
    ok "Metal Toolchain found"
fi

# ── Step 6: Clone repo (if not already in it) ───────────────
if [ ! -f "serve.py" ]; then
    if [ ! -d "meta-turboquant" ]; then
        info "Downloading Meta-TurboQuant..."
        git clone https://github.com/willieprinsloosa/meta-turboquant.git
        cd meta-turboquant
        ok "Downloaded"
    else
        cd meta-turboquant
        ok "Using existing meta-turboquant directory"
    fi
else
    ok "Already in project directory"
fi

# ── Step 7: Create Python environment ───────────────────────
if [ ! -d ".venv13" ]; then
    info "Creating Python environment..."
    $PY13 -m venv .venv13
    ok "Virtual environment created"
else
    ok "Virtual environment exists"
fi

source .venv13/bin/activate

# ── Step 8: Install dependencies ────────────────────────────
info "Installing AI framework..."
pip install --upgrade pip -q 2>/dev/null

# Try pre-built wheel first (no Xcode needed, ~10 seconds)
WHEEL_DIR="$(dirname "$0")/wheels"
if ls "$WHEEL_DIR"/mlx-*.whl 1>/dev/null 2>&1; then
    info "Using pre-built wheel (no Xcode required)..."
    pip install "$WHEEL_DIR"/mlx-*.whl mlx-lm numpy pytest -q 2>&1 | tail -3
    ok "AI framework installed (from wheel — no compilation needed)"
else
    # Fall back to building from source (~5 minutes, needs Metal Toolchain)
    warn "No pre-built wheel found. Building from source (~5 minutes)..."

    # Ensure Metal Toolchain is available
    if ! xcrun metal --version &>/dev/null 2>&1; then
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain 2>&1 | tail -1
    fi

    pip install mlx@git+https://github.com/PrismML-Eng/mlx.git@prism mlx-lm numpy pytest -q 2>&1 | tail -3
    ok "AI framework installed (compiled from source)"
fi

# ── Step 9: Verify 1-bit support ────────────────────────────
info "Verifying 1-bit model support..."
python -c "
import mlx.core as mx
mx.quantize(mx.ones((1,128)), bits=1, group_size=128)
print('  1-bit quantization: working')
" || fail "1-bit support verification failed"
ok "1-bit AI models supported"

# ── Step 10: Download the AI model ──────────────────────────
info "Downloading Bonsai 8B model (1.3 GB, first time only)..."
python -c "
from mlx_lm import load
print('  Downloading...')
model, tok = load('prism-ml/Bonsai-8B-mlx-1bit')
print(f'  Model loaded: {len(model.layers)} layers')
" 2>&1 | grep -v "Fetching\|Warning\|Download"
ok "Bonsai 8B model ready"

# ── Step 11: Quick test ─────────────────────────────────────
info "Running quick test..."
python -c "
from mlx_lm import load, generate
model, tokenizer = load('prism-ml/Bonsai-8B-mlx-1bit')
response = generate(model, tokenizer, prompt='Say hello in one sentence.', max_tokens=20, verbose=False)
print(f'  AI says: {response.strip()}')
" 2>&1 | grep -v "Fetching\|Warning"
ok "AI model working"

# ── Done! ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}Start the AI server:${NC}"
echo -e "    cd $(pwd)"
echo -e "    source .venv13/bin/activate"
echo -e "    python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit"
echo ""
echo -e "  ${BOLD}Open the chat:${NC}"
echo -e "    http://localhost:11434/chat"
echo ""
echo -e "  ${BOLD}Interactive terminal chat:${NC}"
echo -e "    python chat.py --lean --model prism-ml/Bonsai-8B-mlx-1bit"
echo ""
echo -e "  ${BOLD}API endpoint:${NC}"
echo -e "    http://localhost:11434/v1/chat/completions"
echo ""
echo -e "  ${BOLD}What's installed:${NC}"
echo -e "    Model:  Bonsai 8B (1-bit, 1.3 GB, ~90 tokens/sec)"
echo -e "    Memory: ${RAM_GB}GB available"
echo -e "    Tools:  Function calling supported"
echo -e "    Speed:  ~90 tokens/second"
echo ""
echo -e "  For more info: ${BLUE}https://github.com/willieprinsloosa/meta-turboquant${NC}"
echo ""

# Offer to start the server
read -p "  Start the AI server now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit
fi
