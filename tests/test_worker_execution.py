import io
import wave
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

import nodetool.nodes.whispercpp.whispercpp as whispercpp_module
from nodetool.worker.executor import execute_node, execute_node_stream
from nodetool.worker.node_loader import node_to_metadata

NODE_TYPE = "whispercpp.whispercpp.WhisperCpp"


@dataclass
class FakeSegment:
    t0: int
    t1: int
    text: str
    probability: float = float("nan")


class FakeModel:
    instances: ClassVar[list["FakeModel"]] = []

    def __init__(self, model_path: str, **params):
        self.model_path = model_path
        self.params = params
        self.calls: list[int] = []
        FakeModel.instances.append(self)

    def transcribe(self, media, **kwargs):
        assert kwargs == {"extract_probability": True}
        self.calls.append(len(media))
        index = len(self.calls)
        return [FakeSegment(t0=0, t1=len(media) * 100 // 16000, text=f" part{index}")]


def _wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    samples = (np.sin(np.linspace(0, 440 * seconds, int(seconds * sample_rate))) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    FakeModel.instances = []
    monkeypatch.setattr(whispercpp_module, "PWModel", FakeModel)
    monkeypatch.setattr(whispercpp_module, "_resolve_model_path", lambda name: f"/cache/ggml-{name}.bin")


async def test_execute_transcribes_audio_input():
    result = await execute_node(
        node_type=NODE_TYPE,
        fields={"length_ms": 1000},
        secrets={},
        input_blobs={"audio": _wav_bytes(2.5)},
    )

    assert len(FakeModel.instances) == 1
    assert FakeModel.instances[0].calls == [16000, 16000, 8000]
    assert result["outputs"]["text"] == "part1 part2 part3"


async def test_execute_stream_emits_chunks_and_final_text():
    frames = [
        frame
        async for frame in execute_node_stream(
            node_type=NODE_TYPE,
            fields={"length_ms": 1000},
            secrets={},
            input_blobs={"audio": _wav_bytes(1.5)},
        )
    ]

    outputs = [frame["outputs"] for frame in frames]
    chunk_texts = [o["chunk"]["content"] for o in outputs if "chunk" in o]
    assert chunk_texts == [" part1", " part2", ""]
    assert [o["chunk"]["done"] for o in outputs if "chunk" in o] == [False, False, True]
    assert outputs[-1] == {"text": "part1 part2"}


async def test_execute_rejects_missing_audio():
    with pytest.raises(ValueError, match="Audio input is empty"):
        await execute_node(node_type=NODE_TYPE, fields={}, secrets={}, input_blobs={})
    assert FakeModel.instances == []


def test_metadata_declares_buffered_input():
    metadata = node_to_metadata(whispercpp_module.WhisperCpp)
    assert metadata["is_streaming_input"] is False
    assert metadata["is_streaming_output"] is True
