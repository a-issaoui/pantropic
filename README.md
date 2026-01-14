<H1 align="center">
Antropic
</H1>
<div align="center">

**Intelligent Local LLM Server**

*GPU-optimized GGUF model serving with OpenAI-compatible API*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## 🚀 Features

### Core Capabilities
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **Intelligent Context Sizing** - Dynamic context window based on request needs
- **GPU-First Loading** - Maximizes VRAM utilization, offloads only when necessary
- **Request Queue** - Serializes concurrent requests to prevent CUDA crashes
- **Multi-Model Support** - Chat, embeddings, vision, audio, and tool-calling models

### Smart Hardware Management
- **Dynamic GPU Layer Calculation** - Optimal layer distribution based on model size and available VRAM
- **Automatic Model Eviction** - LRU-based cache management with intelligent scoring
- **VRAM Profiling** - Real-time GPU memory monitoring and allocation
- **Context Expansion** - Auto-reload with larger context when needed

### Model Features
- **GGUF Model Scanner** - Auto-detection of model capabilities from metadata
- **Chat Templates** - Automatic Jinja2 template extraction and rendering
- **Tool Calling** - Native function calling support for compatible models
- **Streaming Responses** - Server-Sent Events (SSE) for real-time output
- **Session Management** - Persistent conversation sessions with disk storage

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ VRAM for optimal performance

### Quick Install

```bash
# Clone the repository
git clone https://github.com/pantropic/pantropic.git
cd pantropic

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install with GPU support
CMAKE_ARGS="-DGGML_CUDA=on" pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Configuration

Create `config.yaml` in the project root:

```yaml
# Server Configuration
host: "0.0.0.0"
port: 8090

# Model paths
model_paths:
  - "/path/to/your/models"

# Hardware settings
flash_attention: true
use_mmap: true
default_context: 8192
max_context: 131072

# Agent Mode (recommended for AI agents)
agent_mode: true
gpu_priority: "max"  # max | balanced | efficient
context_expansion_threshold: 0.8
min_gpu_layers_percent: 70
```

---

## 🎯 Quick Start

### 1. Scan Models

```bash
# Scan models directory and generate models.json
python -m pantropic.main --scan
```

### 2. Start Server

```bash
# Normal startup
python -m pantropic.main

# With model rescan
python -m pantropic.main --scan
```

### 3. Use the API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="not-needed"  # No auth required for local
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen2.5-3b-instruct-q4_k_m.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Qwen2.5-3b-instruct-q4_k_m.gguf",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
embeddings = client.embeddings.create(
    model="nomic-embed-text-v1.5.Q8_0.gguf",
    input="Hello, world!"
)
print(f"Embedding dimension: {len(embeddings.data[0].embedding)}")
```

---

## 🔄 Service Management

Use the included `pantropic.sh` script to manage the server:

```bash
# Start server in background
./pantropic.sh start

# Check status
./pantropic.sh status

# View logs (live)
./pantropic.sh logs

# Restart
./pantropic.sh restart

# Stop
./pantropic.sh stop

# Rescan models
./pantropic.sh scan
```

---

## 📚 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/v1/sessions` | GET/POST | Session management |
| `/v1/sessions/{id}` | GET/DELETE | Get/delete session |
| `/v1/sessions/{id}/messages` | POST | Add message to session |
| `/health` | GET | Health check |
| `/metrics` | GET | Server metrics |

### Chat Completions

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Tool Calling

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }]
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text-v1.5.Q8_0.gguf",
    "input": ["Hello, world!", "How are you?"]
  }'
```

---

## ⚙️ Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | str | "0.0.0.0" | Server bind address |
| `port` | int | 8090 | Server port |
| `model_paths` | list | [] | Paths to scan for GGUF models |
| `flash_attention` | bool | true | Enable Flash Attention |
| `use_mmap` | bool | true | Memory-map model files |
| `default_context` | int | 8192 | Default context window size |
| `max_context` | int | 131072 | Maximum context window size |
| `agent_mode` | bool | true | Enable intelligent context sizing |
| `gpu_priority` | str | "max" | GPU usage priority (max/balanced/efficient) |
| `context_expansion_threshold` | float | 0.8 | Trigger reload at this context usage |
| `min_gpu_layers_percent` | int | 70 | Minimum GPU layers before CPU offload |

---

## 🏗️ Architecture

```
pantropic/
├── api/                    # FastAPI application
│   ├── app.py             # App factory with error handlers
│   ├── routes/            # API endpoints
│   └── schemas/           # Request/response models
├── core/                   # Core components
│   ├── config.py          # Configuration management
│   ├── container.py       # Dependency injection
│   ├── exceptions.py      # Custom exceptions
│   └── types.py           # Type definitions
├── hardware/              # Hardware management
│   ├── gpu.py             # GPU monitoring
│   ├── optimizer.py       # Layer calculation
│   ├── allocator.py       # VRAM allocation
│   └── vram.py            # VRAM profiling
├── inference/             # Inference engine
│   ├── engine.py          # Main inference logic
│   ├── queue.py           # Request queue
│   └── sessions.py        # Session management
├── model_manager/         # Model management
│   ├── loader.py          # Model loading
│   ├── registry.py        # Model registry
│   └── scanner.py         # GGUF scanner
├── tools/                 # Tool calling
│   ├── extractor.py       # Extract tool calls
│   └── injector.py        # Inject tool definitions
├── observability/         # Logging
│   └── logging.py         # Structured logging
└── main.py                # Entry point
```

---

## 🧪 Testing

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=pantropic --cov-report=html

# Run API tests (requires running server)
python examples/test_all_apis.py

# Run edge case tests
python examples/test_edge_cases.py

# Run error handling tests
python examples/test_error_handling.py

# Run request queue test
python examples/test_request_queue.py

# Run context size tests
python examples/test_context_sizes.py
```

---

## 📊 Performance

### Intelligent Context Management

Pantropic dynamically sizes context windows based on:
- Model's trained context length (from GGUF metadata)
- Current request token count
- Available GPU VRAM

Context tiers are automatically calculated as percentages of max context:
- 2k, 4k (fixed small tiers)
- 25%, 50%, 75%, 100% (dynamic tiers)

### GPU Optimization

| Priority Mode | Behavior |
|---------------|----------|
| `max` | Use 100% GPU layers if possible |
| `balanced` | 70-100% GPU layers |
| `efficient` | Minimum viable GPU layers |

---

## 🔧 Development

### Code Style

```bash
# Lint check
ruff check pantropic/

# Auto-fix
ruff check --fix pantropic/

# Type checking
mypy pantropic/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF inference engine
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

---

<div align="center">

**Made with ❤️ for local AI**

</div>
