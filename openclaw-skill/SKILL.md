---
name: turboquant_local_llm
description: Run local LLMs on Apple Silicon with TurboQuant KV-cache compression for longer context and lower memory usage.
user-invocable: true
metadata:
  {
    "openclaw": {
      "os": ["darwin"],
      "requires": {
        "bins": ["python3"]
      },
      "install": [
        {
          "id": "pip-mlx",
          "kind": "node",
          "package": "mlx",
          "bins": [],
          "label": "Install MLX framework"
        }
      ]
    }
  }
---

# TurboQuant Local LLM

Run local language models on your Mac with **TurboQuant KV-cache compression**, achieving 3-5x memory savings. This allows longer context windows and larger models on limited hardware.

## When to use

- When the user wants **private, offline, local LLM inference**
- When the user wants to run models with **longer context** on limited memory (16GB)
- When the user asks about TurboQuant, KV-cache compression, or local MLX models

## Setup

The TurboQuant server must be running. Start it with the `exec` tool:

```bash
cd /Users/wlprinsloo/Documents/my\ projects/metaTurboQuant/turboquant-mlx && source .venv/bin/activate && python serve.py
```

### Available models for 16GB Mac

| Model | Memory | Best for |
|-------|--------|----------|
| `mlx-community/Llama-3.2-3B-Instruct-4bit` (default) | ~1.8 GB | Fast responses, general tasks |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | ~0.7 GB | Fastest, lightweight tasks |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4.5 GB | Best quality at 7B |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB | Best quality at 8B |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.5 GB | Strong multilingual |

### Server options

```bash
python serve.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --bits 3 --strategy v2
python serve.py --port 8800
python serve.py --strategy v3 --bits 3  # better quality, slower
```

## How to interact with the local LLM

Use the `fetch` tool to send requests to the local TurboQuant server:

### Chat completion

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 256
  }'
```

### Check server status

```bash
curl -s http://localhost:11434/health
```

### List available models

```bash
curl -s http://localhost:11434/v1/models
```

## OpenClaw provider configuration

To use TurboQuant as an LLM provider in OpenClaw, add to your `~/.openclaw/openclaw.json`:

```json
{
  "providers": {
    "turboquant": {
      "type": "openai",
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "local",
      "models": ["mlx-community/Llama-3.2-3B-Instruct-4bit"]
    }
  }
}
```

## Compression strategies

| Strategy | Speed | Quality | Compression |
|----------|-------|---------|-------------|
| V2 4-bit | Fast (85 tok/s) | Near-lossless | 3.1x |
| V2 3-bit | Good (58 tok/s) | +5% PPL | 3.8x |
| V3 3-bit | Slower | Best at 3-bit | 4.7x |
| V3 2.5-bit | Slower | +10-27% PPL | 5.5x |

Default is **V2 4-bit** — best balance of speed, quality, and compression.
