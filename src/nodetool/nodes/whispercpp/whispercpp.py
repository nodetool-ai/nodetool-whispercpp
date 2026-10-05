from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, AsyncGenerator

import numpy as np
from pydantic import Field
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, HuggingFaceModel
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

# Optional: used for streaming string chunks
from nodetool.providers import Chunk

# New library: pywhispercpp
from pywhispercpp.model import Model as PWModel, Segment

log = get_logger(__name__)

REPO_ID = "ggerganov/whisper.cpp"


def _resolve_model_path(model_name: str) -> str:
    """Map model short name (e.g., 'tiny.en') to ggml file in HF cache.

    Uses Hugging Face cache to avoid network downloads.
    """
    # pywhispercpp uses whisper.cpp ggml files named ggml-<name>.bin
    filename = f"ggml-{model_name}.bin"
    filepath = try_to_load_from_cache(REPO_ID, filename)
    if isinstance(filepath, str):
        return filepath
    if filepath is _CACHED_NO_EXIST:
        raise FileNotFoundError(
            f"Model file not found in HF cache: {REPO_ID}:{filename}"
        )
    raise FileNotFoundError(f"Model file not found in HF cache: {REPO_ID}:{filename}")


class WhisperCpp(BaseNode):
    """
    Transcribe an audio asset using whispercpp (whisper.cpp bindings) and stream strings.
    whisper, whispercpp, asr, speech-to-text, streaming, huggingface-cache

    - Model file is loaded from the local Hugging Face cache (repo + filename)
    - Transcribes the audio in windows of `length_ms`
    - Emits each window's text on `chunk` and the full transcript on `text`
    """

    class Model(str, Enum):
        # tiny
        TINY = "tiny"
        TINY_Q5_1 = "tiny-q5_1"
        TINY_Q8_0 = "tiny-q8_0"
        TINY_EN = "tiny.en"
        TINY_EN_Q5_1 = "tiny.en-q5_1"
        TINY_EN_Q8_0 = "tiny.en-q8_0"

        # base
        BASE = "base"
        BASE_Q5_1 = "base-q5_1"
        BASE_Q8_0 = "base-q8_0"
        BASE_EN = "base.en"
        BASE_EN_Q5_1 = "base.en-q5_1"
        BASE_EN_Q8_0 = "base.en-q8_0"

        # small
        SMALL = "small"
        SMALL_Q5_1 = "small-q5_1"
        SMALL_Q8_0 = "small-q8_0"
        SMALL_EN = "small.en"
        SMALL_EN_Q5_1 = "small.en-q5_1"
        SMALL_EN_Q8_0 = "small.en-q8_0"

        # medium
        MEDIUM = "medium"
        MEDIUM_Q5_0 = "medium-q5_0"
        MEDIUM_Q8_0 = "medium-q8_0"
        MEDIUM_EN = "medium.en"
        MEDIUM_EN_Q5_0 = "medium.en-q5_0"
        MEDIUM_EN_Q8_0 = "medium.en-q8_0"

        # large variants
        LARGE_V1 = "large-v1"
        LARGE_V2 = "large-v2"
        LARGE_V2_Q5_0 = "large-v2-q5_0"
        LARGE_V2_Q8_0 = "large-v2-q8_0"
        LARGE_V3 = "large-v3"
        LARGE_V3_Q5_0 = "large-v3-q5_0"
        LARGE_V3_TURBO = "large-v3-turbo"
        LARGE_V3_TURBO_Q5_0 = "large-v3-turbo-q5_0"
        LARGE_V3_TURBO_Q8_0 = "large-v3-turbo-q8_0"

    model: Model = Field(
        default=Model.TINY_EN,
        description="Model to use from ggerganov/whisper.cpp",
    )

    # Decoding parameters (subset, expanded as needed)
    n_threads: int = Field(default=4, description="Decoder CPU threads")
    length_ms: int = Field(
        default=5000, description="Window length in milliseconds for streaming output"
    )

    audio: AudioRef = Field(default=AudioRef(), description="Audio to transcribe")
    chunk: Chunk = Field(
        default=Chunk(),
        description="Live audio chunks are not supported yet; connect an audio asset instead",
    )

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        # The Python bridge delivers one snapshot of the inputs per execution,
        # so a live chunk stream cannot reach this node.
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
            "t0": float,
            "t1": float,
            "probability": float,
        }

    def _load_whisper(self) -> PWModel:
        # Resolve local path from HF cache and instantiate pywhispercpp model
        model_path = _resolve_model_path(self.model.value)
        return PWModel(
            model_path,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            single_segment=True,
            n_threads=self.n_threads,
        )

    async def _audio_to_float32(
        self, context: ProcessingContext, audio: AudioRef
    ) -> tuple[np.ndarray, int]:
        """Convert AudioRef to float32 mono array at 16kHz and return (array, sample_rate)."""
        if not audio or audio.is_empty():
            raise ValueError("Audio input is empty; please connect an audio source")

        audio_segment = await context.audio_to_audio_segment(audio)
        # Ensure mono, 16kHz, 16-bit
        audio_segment = (
            audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        )
        # Convert to float32 in range [-1.0, 1.0]
        raw_data = audio_segment.raw_data
        if not isinstance(raw_data, (bytes, bytearray, memoryview)):
            raise TypeError("AudioSegment.raw_data is not bytes-like")
        pcm = np.frombuffer(raw_data, dtype=np.int16)
        arr = (pcm.astype(np.float32) / 32768.0).flatten()
        return arr, 16000

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.chunk.content:
            raise ValueError(
                "WhisperCpp does not support live audio chunks yet; connect an audio asset to `audio`"
            )
        samples, sample_rate = await self._audio_to_float32(context, self.audio)

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(None, self._load_whisper)

        def transcribe(window: np.ndarray) -> list[Segment]:
            return model.transcribe(window, extract_probability=True)

        window_len = max(1, self.length_ms * sample_rate // 1000)
        texts: list[str] = []
        for start in range(0, samples.shape[0], window_len):
            window = samples[start : start + window_len]
            try:
                segments = await loop.run_in_executor(None, transcribe, window)
            except Exception as e:
                raise RuntimeError(
                    f"Whisper (pywhispercpp) transcription failed: {e}"
                ) from e

            text = "".join(segment.text for segment in segments)
            if text:
                texts.append(text)
                yield {"chunk": Chunk(content=text, done=False)}
            for segment in segments:
                yield {
                    "t0": segment.t0,
                    "t1": segment.t1,
                    "probability": segment.probability,
                }

        yield {"chunk": Chunk(content="", done=True)}
        yield {"text": "".join(texts).strip()}

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        """Recommend ggml Whisper models from ggerganov/whisper.cpp for local cache use.

        These correspond to files listed on the HF repo page and are suitable
        for whisper.cpp bindings.
        """
        paths = [f"ggml-{m.value}.bin" for m in cls.Model]
        return [HuggingFaceModel(repo_id=REPO_ID, path=p) for p in paths]
