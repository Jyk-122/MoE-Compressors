"""
Binary I/O for per-forward MoE expert routing traces (see MoEStatsCollector).

File layout (little-endian):
  - Header: magic b\"MOETRC\", uint16 version, uint32 num_experts, uint32 num_layers
  - Stream of records:
      uint8 tag: 1 = SAMPLE_START, 2 = FORWARD_BLOCK
      SAMPLE_START: (no payload)
      FORWARD_BLOCK: uint32 sequence_length, uint32 n_layer_rows
        repeated n_layer_rows times:
          uint32 layer_idx, uint32 n_tokens, uint32 top_k
          int16[n_tokens * top_k] row-major (use -1 for inactive slots)
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

TRACE_MAGIC = b"MOETRC"
TRACE_VERSION = 1

TAG_SAMPLE_START = 1
TAG_FORWARD_BLOCK = 2

_HEADER_STRUCT = struct.Struct("<6sHII")  # magic 6, version u16, num_experts, num_layers
_RECORD_SAMPLE = struct.Struct("<B")
_RECORD_FWD_HEAD = struct.Struct("<BII")  # tag, seq_len, n_layers


@dataclass(frozen=True)
class TraceHeader:
    version: int
    num_experts: int
    num_layers: int


def write_trace_header(fp: BinaryIO, *, num_experts: int, num_layers: int) -> None:
    packed = _HEADER_STRUCT.pack(TRACE_MAGIC, TRACE_VERSION, int(num_experts), int(num_layers))
    fp.write(packed)
    fp.flush()


def read_trace_header(fp: BinaryIO) -> TraceHeader:
    data = fp.read(_HEADER_STRUCT.size)
    if len(data) < _HEADER_STRUCT.size:
        raise ValueError("Trace file too short for header")
    magic, version, num_experts, num_layers = _HEADER_STRUCT.unpack(data)
    if magic != TRACE_MAGIC:
        raise ValueError(f"Bad trace magic: {magic!r}")
    return TraceHeader(version=int(version), num_experts=int(num_experts), num_layers=int(num_layers))


def write_sample_start(fp: BinaryIO) -> None:
    fp.write(_RECORD_SAMPLE.pack(TAG_SAMPLE_START))


def write_forward_block(
    fp: BinaryIO,
    *,
    sequence_length: int,
    layers: list[tuple[int, np.ndarray]],
) -> None:
    """
    layers: list of (layer_idx, array int16 shape (n_tokens, top_k))
    """
    n_layers = len(layers)
    fp.write(_RECORD_FWD_HEAD.pack(TAG_FORWARD_BLOCK, int(sequence_length), int(n_layers)))
    layer_head = struct.Struct("<III")
    for layer_idx, arr in layers:
        arr = np.asarray(arr, dtype=np.int16)
        if arr.ndim != 2:
            raise ValueError(f"layer {layer_idx}: expected 2d array, got {arr.shape}")
        n_tokens, top_k = int(arr.shape[0]), int(arr.shape[1])
        fp.write(layer_head.pack(int(layer_idx), n_tokens, top_k))
        fp.write(arr.tobytes(order="C"))


def load_trace_header_only(path: str | Path) -> TraceHeader:
    with Path(path).open("rb") as fp:
        return read_trace_header(fp)


def iter_trace_records(
    path: str | Path,
) -> Iterator[tuple[str, dict[str, object]]]:
    """
    Yields tuples (kind, payload).
    kind == \"sample_start\": payload {}
    kind == \"forward\": payload {\"sequence_length\": int, \"layers\": {layer_idx: ndarray int16 (n,k)}}
    """
    p = Path(path)
    with p.open("rb") as fp:
        read_trace_header(fp)
        layer_head = struct.Struct("<III")
        while True:
            tag_b = fp.read(1)
            if not tag_b:
                break
            (tag,) = struct.unpack("<B", tag_b)
            if tag == TAG_SAMPLE_START:
                yield ("sample_start", {})
            elif tag == TAG_FORWARD_BLOCK:
                head = fp.read(8)
                if len(head) < 8:
                    break
                seq_len, n_layers = struct.unpack("<II", head)
                layers: dict[int, np.ndarray] = {}
                for _ in range(n_layers):
                    lh = fp.read(layer_head.size)
                    if len(lh) < layer_head.size:
                        raise ValueError("Truncated FORWARD_BLOCK layer header")
                    layer_idx, n_tokens, top_k = layer_head.unpack(lh)
                    nbytes = int(n_tokens) * int(top_k) * 2
                    raw = fp.read(nbytes)
                    if len(raw) < nbytes:
                        raise ValueError("Truncated FORWARD_BLOCK payload")
                    arr = np.frombuffer(raw, dtype=np.int16).reshape(int(n_tokens), int(top_k))
                    layers[int(layer_idx)] = arr.copy()
                yield ("forward", {"sequence_length": int(seq_len), "layers": layers})
            else:
                raise ValueError(f"Unknown trace record tag: {tag}")


def iter_samples(path: str | Path) -> Iterator[list[dict[str, object]]]:
    """
    Each sample is a list of forward steps (chronological).
    Each step: {\"sequence_length\": int, \"layers\": dict[int, ndarray]}.
    """
    current: list[dict[str, object]] = []
    for kind, payload in iter_trace_records(path):
        if kind == "sample_start":
            if current:
                yield current
            current = []
        elif kind == "forward":
            current.append(payload)
    if current:
        yield current


def count_trace_samples(path: str | Path) -> int:
    n = 0
    for _ in iter_samples(path):
        n += 1
    return n
