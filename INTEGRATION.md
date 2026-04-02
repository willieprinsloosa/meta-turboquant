# Meta-TurboQuant Integration Guide

Run local LLMs on your Mac with KV-cache compression. This guide covers how to integrate the Meta-TurboQuant server with various clients, frameworks, and your own applications.

## Table of Contents

- [Server Setup](#server-setup)
- [Python (OpenAI SDK)](#python-openai-sdk)
- [JavaScript / TypeScript](#javascript--typescript)
- [curl](#curl)
- [Tool Calling (Function Calling)](#tool-calling-function-calling)
- [OpenClaw](#openclaw)
- [LangChain](#langchain)
- [Continue.dev (VS Code)](#continuedev-vs-code)
- [Atomic Chat / Jan](#atomic-chat--jan)
- [Custom HTTP Client](#custom-http-client)
- [Choosing a Model](#choosing-a-model)
- [Troubleshooting](#troubleshooting)

---

## Server Setup

### Standard models

```bash
cd turboquant-mlx
source .venv/bin/activate
python serve.py --lean
```

### Bonsai 1-bit models (recommended)

```bash
cd turboquant-mlx
source .venv13/bin/activate
python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit
```

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Llama-3.2-3B-Instruct-4bit` | HuggingFace model ID |
| `--port` | `11434` | Server port |
| `--lean` | off | Disable rotation for max speed |
| `--bits` | `4` | KV-cache quantization (2, 3, or 4) |
| `--strategy` | `v2` | Compression strategy (`v2` or `v3`) |

### Verify the server is running

```bash
curl http://localhost:11434/health
# {"status": "ok", "model": "prism-ml/Bonsai-8B-mlx-1bit"}
```

---

## Python (OpenAI SDK)

Install the OpenAI Python SDK:

```bash
pip install openai
```

### Basic chat

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="local",  # any string works
)

response = client.chat.completions.create(
    model="prism-ml/Bonsai-8B-mlx-1bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is quantum computing?"},
    ],
    max_tokens=256,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="prism-ml/Bonsai-8B-mlx-1bit",
    messages=[{"role": "user", "content": "Write a poem about coding"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Tool calling

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
]

# Step 1: Send message with tools
response = client.chat.completions.create(
    model="prism-ml/Bonsai-8B-mlx-1bit",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

message = response.choices[0].message

if message.tool_calls:
    # Step 2: Execute the tool (your code)
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Model wants to call: {tool_call.function.name}({args})")

    # Simulate tool result
    tool_result = json.dumps({"temp": 22, "condition": "sunny", "humidity": 65})

    # Step 3: Send tool result back
    response2 = client.chat.completions.create(
        model="prism-ml/Bonsai-8B-mlx-1bit",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            message,  # assistant message with tool_calls
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            },
        ],
        tools=tools,
    )

    print(response2.choices[0].message.content)
```

### Multi-turn conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="prism-ml/Bonsai-8B-mlx-1bit",
        messages=messages,
        max_tokens=512,
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"Assistant: {reply}\n")
```

---

## JavaScript / TypeScript

```bash
npm install openai
```

### Basic chat

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "local",
});

const response = await client.chat.completions.create({
  model: "prism-ml/Bonsai-8B-mlx-1bit",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0].message.content);
```

### Streaming

```typescript
const stream = await client.chat.completions.create({
  model: "prism-ml/Bonsai-8B-mlx-1bit",
  messages: [{ role: "user", content: "Explain recursion" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

### Tool calling

```typescript
const response = await client.chat.completions.create({
  model: "prism-ml/Bonsai-8B-mlx-1bit",
  messages: [{ role: "user", content: "Search for AI news" }],
  tools: [
    {
      type: "function",
      function: {
        name: "search_web",
        description: "Search the web",
        parameters: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
        },
      },
    },
  ],
});

const toolCalls = response.choices[0].message.tool_calls;
if (toolCalls) {
  console.log("Tool call:", toolCalls[0].function.name);
  console.log("Arguments:", toolCalls[0].function.arguments);
}
```

---

## curl

### Chat

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prism-ml/Bonsai-8B-mlx-1bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### List models

```bash
curl http://localhost:11434/v1/models
```

---

## Tool Calling (Function Calling)

Tool calling lets the model decide when to use external functions. The flow is:

```
User message + tool definitions
        |
        v
   Model decides: respond directly OR call a tool
        |
        v
   If tool_call: your code executes the function
        |
        v
   Send tool result back to model
        |
        v
   Model generates final response using the result
```

### Supported models

| Model | Tool Calling |
|-------|:------------:|
| `prism-ml/Bonsai-8B-mlx-1bit` | Yes (Qwen3) |
| `prism-ml/Bonsai-4B-mlx-1bit` | Yes (Qwen3) |
| `prism-ml/Bonsai-1.7B-mlx-1bit` | Yes (Qwen3) |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | Yes |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | No |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | No |

### How it works

1. You send `tools` array in the request (same format as OpenAI)
2. The model generates `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
3. The server parses this into standard OpenAI `tool_calls` format
4. You execute the tool and send the result back as a `tool` role message
5. The model uses the result to generate a natural language response

### Multiple tools

You can provide multiple tools and the model will choose the right one:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What time is it in London?"}],
    "tools": [
      {"type": "function", "function": {"name": "get_time", "description": "Get current time in a timezone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}},
      {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
      {"type": "function", "function": {"name": "search_web", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
  }'
# Model will correctly choose get_time with timezone "Europe/London"
```

---

## OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) can use Meta-TurboQuant as a local LLM provider.

### Configure provider

Add to `~/.openclaw/openclaw.json`:

```json
{
  "providers": {
    "turboquant": {
      "type": "openai",
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "local",
      "models": ["prism-ml/Bonsai-8B-mlx-1bit"]
    }
  }
}
```

### Install the skill (optional)

```bash
cp -r openclaw-skill ~/.openclaw/skills/turboquant-local-llm
```

---

## LangChain

```bash
pip install langchain-openai
```

### Basic usage

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="local",
    model="prism-ml/Bonsai-8B-mlx-1bit",
    temperature=0.7,
)

response = llm.invoke("Explain machine learning in simple terms")
print(response.content)
```

### With tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 22C and sunny."

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Paris?")

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc['name']}, Args: {tc['args']}")
```

### Agent with tool execution

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [get_weather], prompt)
executor = AgentExecutor(agent=agent, tools=[get_weather])

result = executor.invoke({"input": "What's the weather in Tokyo and London?"})
print(result["output"])
```

---

## Continue.dev (VS Code)

[Continue](https://continue.dev) is an AI coding assistant for VS Code. Point it at your local server.

### Configure

Edit `~/.continue/config.yaml`:

```yaml
models:
  - title: "TurboQuant Bonsai 8B"
    provider: openai
    model: prism-ml/Bonsai-8B-mlx-1bit
    apiBase: http://localhost:11434/v1
    apiKey: local
```

Or in `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "TurboQuant Bonsai 8B",
      "provider": "openai",
      "model": "prism-ml/Bonsai-8B-mlx-1bit",
      "apiBase": "http://localhost:11434/v1",
      "apiKey": "local"
    }
  ]
}
```

---

## Atomic Chat / Jan

Atomic Chat (fork of Jan) uses the OpenAI chat completions format.

### Configure

1. Open **Settings > Model Providers**
2. Add a **Custom OpenAI-compatible** provider (not the "OpenAI" option)
3. Set:
   - **Base URL**: `http://localhost:11434/v1`
   - **API Key**: `local`
4. Select model: `prism-ml/Bonsai-8B-mlx-1bit`

> **Important**: Use "Custom/OpenAI-compatible", not "OpenAI". The "OpenAI" option uses the `/v1/responses` endpoint which may not stream correctly in all clients.

---

## Custom HTTP Client

The server implements the standard OpenAI API. Here is the full request/response format.

### POST /v1/chat/completions

**Request:**

```json
{
  "model": "prism-ml/Bonsai-8B-mlx-1bit",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false,
  "tools": []
}
```

**Response (normal):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "prism-ml/Bonsai-8B-mlx-1bit",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
}
```

**Response (tool call):**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
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

**Streaming response (SSE):**

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-3","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### POST /v1/responses

Same as `/v1/chat/completions` but accepts `input` instead of `messages`:

```json
{
  "model": "prism-ml/Bonsai-8B-mlx-1bit",
  "input": "What is AI?",
  "max_output_tokens": 256,
  "stream": true
}
```

`input` can be a string or an array of message objects (supports `developer` role mapped to `system`).

### GET /v1/models

```json
{
  "object": "list",
  "data": [{"id": "prism-ml/Bonsai-8B-mlx-1bit", "object": "model", "owned_by": "local"}]
}
```

### GET /health

```json
{"status": "ok", "model": "prism-ml/Bonsai-8B-mlx-1bit"}
```

---

## Choosing a Model

### For tool calling / agents

Use Bonsai (Qwen3 architecture). Requires `.venv13`:

```bash
source .venv13/bin/activate
python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit   # best quality
python serve.py --lean --model prism-ml/Bonsai-4B-mlx-1bit   # faster
python serve.py --lean --model prism-ml/Bonsai-1.7B-mlx-1bit # fastest, 80MB
```

### For general chat (no tools needed)

Standard models work with `.venv`:

```bash
source .venv/bin/activate
python serve.py --lean                                                        # Llama 3.2 3B (default)
python serve.py --lean --model mlx-community/Llama-3.1-8B-Instruct-4bit     # best quality
python serve.py --lean --model mlx-community/Mistral-7B-Instruct-v0.3-4bit  # strong 7B
```

### Memory budget (16GB Mac)

| Model | Memory | Remaining for KV + OS |
|-------|--------|----------------------|
| Bonsai 1.7B | 80 MB | ~15.9 GB |
| Bonsai 4B | 0.7 GB | ~15.3 GB |
| Bonsai 8B | 1.3 GB | ~14.7 GB |
| Llama 3.2 3B 4-bit | 1.8 GB | ~14.2 GB |
| Llama 3.1 8B 4-bit | 4.5 GB | ~11.5 GB |

TurboQuant V2 4-bit compresses KV-cache by 3.1-3.6x, enabling longer context windows.

---

## Troubleshooting

### Server won't start: "Address already in use"

```bash
lsof -ti:11434 | xargs kill -9
python serve.py --lean
```

### "No module named 'mlx.core'"

You're using the wrong venv. Standard models need `.venv`, Bonsai needs `.venv13`:

```bash
# Standard models
source .venv/bin/activate

# Bonsai 1-bit models
source .venv13/bin/activate
```

### "No matching distribution found for mlx"

Your Python is running under Rosetta (x86_64). Check:

```bash
python3 -c "import platform; print(platform.machine())"
# Must print: arm64
```

Use `/opt/homebrew/bin/python3` or install arm64 Python via Homebrew.

### Atomic Chat hangs

Use **Custom/OpenAI-compatible** provider, not the "OpenAI" option. The "OpenAI" option hits `/v1/responses` which may not work with all streaming clients.

### Tool calls not working

Make sure you're using a tool-capable model (Bonsai/Qwen). Llama models don't support tool calling.

### Slow responses

- Use `--lean` flag (disables rotation, ~2x faster)
- Use a smaller model (Bonsai 4B or 1.7B)
- Close other GPU-intensive apps

### Model download fails

Set a HuggingFace token for faster downloads:

```bash
export HF_TOKEN=your_token_here
python serve.py --lean --model prism-ml/Bonsai-8B-mlx-1bit
```
