# nodetool-whispercpp

High-performance Whisper.cpp speech-to-text nodes for [Nodetool](https://github.com/nodetool-ai/nodetool). This package wraps the community [pywhispercpp](https://github.com/absadiki/pywhispercpp) bindings so you can run low-latency transcription workflows entirely on the CPU with streaming support.

## Why nodetool-whispercpp?

- **Local-first ASR** – keep conversations on-device by decoding audio without cloud services
- **Streaming-friendly** – ingest real-time audio chunks and emit incremental transcript updates for responsive UIs
- **Whisper.cpp ecosystem** – reuse ggml checkpoints from `ggerganov/whisper.cpp` via the Hugging Face cache manager built into Nodetool

## Provided Nodes

All nodes live under `src/nodetool/nodes/whispercpp`:

- `whispercpp.WhisperCpp` – configurable Whisper.cpp transcription with streaming text (`chunk`) and final transcript (`text`) outputs

Typed DSL wrappers are available under `src/nodetool/dsl/whispercpp` for use in generated workflows.

## Requirements

- Python 3.11
- CPU with AVX instructions (recommended for Whisper.cpp performance)
- [nodetool-core](https://github.com/nodetool-ai/nodetool-core) v0.6.0+
- ggml Whisper model files cached locally from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)

## Installation

### From the Nodetool UI

1. Open Nodetool → **Tools ▸ Packages**
2. Install the `nodetool-whispercpp` pack from the package registry
3. Nodetool will handle dependencies and expose the Whisper nodes in the graph editor once installed

### From source (development)

```bash
git clone https://github.com/nodetool-ai/nodetool-whispercpp.git
cd nodetool-whispercpp
uv pip install -e .
uv pip install -r requirements-dev.txt
```

If you prefer Poetry or pip, install the project the same way—just ensure dependencies are resolved against Python 3.11.

## Managing Models

The Whisper.cpp node expects ggml model files (e.g., `ggml-base.en.bin`) in the local Hugging Face cache. Download and manage them from the **Models Manager** in Nodetool:

1. Open Nodetool → **Menu ▸ Models**
2. Select the `whispercpp` tab to view recommended checkpoints for each model size
3. Click **Download** for the models you plan to use; Nodetool stores them in the Hugging Face cache automatically
4. The UI tracks model availability and prompts you when updates are available

Advanced users can pre-populate the cache manually by running `huggingface-cli download ggerganov/whisper.cpp ggml-<model>.bin`, but using the UI integration keeps paths consistent and avoids missing-model errors at runtime.

## Usage

1. Install `nodetool-core` and this package in the same environment
2. Run `nodetool package scan` to generate metadata and DSL bindings
3. (Optional) `nodetool codegen` to refresh typed DSL wrappers
4. Build workflows in the Nodetool UI or through Python DSL scripts using the `whispercpp` namespace

Example (Python DSL):

```python
from nodetool.dsl.whispercpp import WhisperCpp

node = WhisperCpp(model="base.en", n_threads=8)
```

Connect an audio source (file, microphone stream, or `AudioRef`) to the node. The node emits incremental transcript chunks on `chunk` and the final transcript on `text`, plus timing and probability metadata for downstream nodes.

## Development

Run tests and lint checks before submitting PRs:

```bash
pytest -q
ruff check .
black --check .
```

Please open issues or pull requests for bug fixes, additional Whisper.cpp capabilities, or performance improvements. Contributions are welcome!
