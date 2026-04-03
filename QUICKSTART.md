# Meta-TurboQuant Quick Start Guide
## Running 1-Bit Bonsai Models on Mac

This guide gets you from zero to running an 8B parameter AI model with tool calling on your Mac in under 10 minutes.

---

## What You Need

- **Mac with Apple Silicon** (M1, M2, M3, or M4)
- **16GB RAM** (minimum)
- **macOS 13.5 or later**
- **Internet connection** (for first-time model download only — works offline after)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/willieprinsloosa/meta-turboquant.git
cd meta-turboquant
```

---

## Step 2: Install Metal Toolchain

The 1-bit models require a custom MLX build which needs Apple's Metal compiler:

```bash
xcodebuild -downloadComponent MetalToolchain
```

This downloads ~700MB. Only needed once.

---

## Step 3: Create Python Environment

Bonsai 1-bit models require **Python 3.13** (arm64):

```bash
# Check if Python 3.13 is available
python3.13 --version

# If not installed:
brew install python@3.13

# Create virtual environment
python3.13 -m venv .venv13
source .venv13/bin/activate
```

---

## Step 4: Install Dependencies

```bash
# Install PrismML MLX fork with 1-bit support (compiles from source, ~5 min)
pip install mlx@git+https://github.com/PrismML-Eng/mlx.git@prism mlx-lm numpy pytest

# Verify 1-bit support works
python -c "import mlx.core as mx; mx.quantize(mx.ones((1,128)), bits=1, group_size=128); print('1-bit support: OK')"
```

You should see: `1-bit support: OK`

---

## Step 5: Start the Server

```bash
# Make sure the venv is active
source .venv13/bin/activate

# Start the server (first run downloads the model — 1.3GB)
python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit
```

You should see:
```
Loading model: prism-ml/Bonsai-8B-mlx-1bit
  36 layers, head_dim=128
  Strategy: V2 4-bit LEAN (group_size=64)
Warming up...
Ready.

Serving on http://localhost:11434
```

---

## Step 6: Test It

Open a **new terminal** and run:

```bash
# Health check
curl http://localhost:11434/health

# Chat
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'

# Open web chat in browser
open http://localhost:11434/chat
```

---

## Available Models

| Model | Size | Speed | Tool Calling | Command |
|-------|------|-------|:------------:|---------|
| **Bonsai 8B** | 1.3 GB | ~90 tok/s | Yes | `python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit` |
| **Bonsai 4B** | 0.7 GB | ~120 tok/s | Yes | `python serve.py --lean --model prism-ml/Bonsai-4B-mlx-1bit` |
| **Bonsai 1.7B** | 80 MB | ~200 tok/s | Yes | `python serve.py --lean --model prism-ml/Bonsai-1.7B-mlx-1bit` |

All models download automatically on first use.

---

## Connect Your App

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://localhost:11434/v1/chat/completions` | POST | Chat (OpenAI-compatible) |
| `http://localhost:11434/v1/responses` | POST | Responses API |
| `http://localhost:11434/v1/models` | GET | List models |
| `http://localhost:11434/health` | GET | Health check |
| `http://localhost:11434/chat` | GET | Web chat UI |

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

response = client.chat.completions.create(
    model="prism-ml/Bonsai-8B-mlx-1bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### JavaScript

```javascript
const response = await fetch("http://localhost:11434/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello!" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

---

## Tool Calling

Bonsai models support function calling. Send tools in your request:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

The model responds with a tool call:
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Tokyo\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

---

## Interactive Chat (Terminal)

```bash
source .venv13/bin/activate
python chat.py --lean --model prism-ml/Bonsai-8B-mlx-1bit
```

Commands: `/clear` (reset), `/stats` (session info), `/quit` (exit)

---

## Run as Background Service

To keep the server running after closing the terminal:

```bash
# Start in background
nohup python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit > server.log 2>&1 &

# Check it's running
curl http://localhost:11434/health

# View logs
tail -f server.log

# Stop the server
kill $(lsof -ti:11434)
```

---

## Troubleshooting

**"No matching distribution found for mlx"**
Your Python is running under Rosetta (x86_64). Check with:
```bash
python3 -c "import platform; print(platform.machine())"
```
Must print `arm64`. Use `/opt/homebrew/bin/python3.13` instead.

**"Address already in use"**
Another instance is running. Kill it:
```bash
kill $(lsof -ti:11434)
```

**"cannot execute tool 'metal'"**
Metal Toolchain not installed. Run:
```bash
xcodebuild -downloadComponent MetalToolchain
```

**Model download is slow**
Set a HuggingFace token for faster downloads:
```bash
export HF_TOKEN=your_token_here
```

**Server seems slow on first request**
The first request after startup includes model warmup. Subsequent requests are faster (~90 tok/s for Bonsai 8B).

---

## What's Running Under the Hood

- **Model:** Bonsai 8B — 8 billion parameters compressed to 1.3GB using 1-bit quantization
- **KV-Cache:** TurboQuant V2 compression — 3.6x memory savings for longer conversations
- **Speed:** ~90 tokens/second on Apple Silicon
- **Privacy:** Everything runs locally — no data leaves your machine
- **Cost:** Zero inference costs — no cloud API needed

---

*For full documentation, see [README.md](README.md) and [INTEGRATION.md](INTEGRATION.md)*
