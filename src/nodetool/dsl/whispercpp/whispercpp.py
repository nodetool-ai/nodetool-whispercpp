from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.whispercpp.whispercpp


class WhisperCpp(GraphNode):
    """
    Transcribe an audio asset using whispercpp (whisper.cpp bindings) and stream strings.
    whisper, whispercpp, asr, speech-to-text, streaming, huggingface-cache

    - Model file is loaded from the local Hugging Face cache (repo + filename)
    - Emits streaming text deltas on `chunk` and final transcript on `text`
    """

    Model: typing.ClassVar[type] = nodetool.nodes.whispercpp.whispercpp.WhisperCpp.Model
    model: nodetool.nodes.whispercpp.whispercpp.WhisperCpp.Model = Field(
        default=nodetool.nodes.whispercpp.whispercpp.WhisperCpp.Model.TINY_EN,
        description="Model to use from ggerganov/whisper.cpp",
    )
    n_threads: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4, description="Decoder CPU threads"
    )
    length_ms: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5000, description="Chunk length in milliseconds for pseudo-streaming"
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="Audio to transcribe",
    )
    chunk: types.Chunk | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Chunk(
            type="chunk",
            node_id=None,
            content_type="text",
            content="",
            content_metadata={},
            done=False,
        ),
        description="Chunk to transcribe",
    )

    @classmethod
    def get_node_type(cls):
        return "whispercpp.whispercpp.WhisperCpp"
