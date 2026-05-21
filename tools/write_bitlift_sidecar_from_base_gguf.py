#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from write_bitlift_sidecar_gguf import (
    DS4_N_EMBD,
    DS4_N_EXPERT,
    DS4_N_FF_EXP,
    GGUF,
    GGUF_BLOCK,
    GGUF_TENSOR_I32,
    GGUF_TENSOR_Q4_K,
    align_up,
    choose_layers,
    load_plan,
    parse_layers,
    tensor_nbytes,
    write_kv_string,
    write_kv_u32,
    write_str,
    GGUF_MAGIC,
    GGUF_VERSION,
)


GGUF_TENSOR_Q2_K = 10
GGUF_TENSOR_IQ2_XXS = 16


@dataclass
class OutputTensor:
    name: str
    dims: list[int]
    typ: int
    nbytes: int
    rel_offset: int = 0
    source_name: str | None = None
    source_type: int | None = None
    experts: list[int] | None = None
    ids: list[int] | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_quants_lib(out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib = out_dir / f"libds4quants{ext}"
    src = repo_root() / "gguf-tools" / "quants.c"
    header = repo_root() / "gguf-tools" / "quants.h"
    if not force and lib.exists() and lib.stat().st_mtime >= max(src.stat().st_mtime, header.stat().st_mtime):
        return lib
    cmd = [
        "cc",
        "-O3",
        "-ffast-math",
        "-std=c11",
        "-fPIC",
        "-dynamiclib" if platform.system() == "Darwin" else "-shared",
        "-o",
        str(lib),
        str(src),
        "-lm",
        "-pthread",
    ]
    subprocess.run(cmd, check=True)
    return lib


class QuantsLib:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        self.lib.ds4q_can_dequantize.argtypes = [ctypes.c_int]
        self.lib.ds4q_can_dequantize.restype = ctypes.c_bool
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_init.restype = None
        self.lib.ds4q_dequantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        self.lib.ds4q_dequantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(GGUF_TENSOR_Q4_K)
        self.lib.ds4q_quantize_init(GGUF_TENSOR_IQ2_XXS)

    def row_size(self, typ: int, cols: int) -> int:
        out = int(self.lib.ds4q_row_size(typ, cols))
        if out <= 0:
            raise SystemExit(f"unsupported row layout type={typ} cols={cols}")
        return out

    def can_dequantize(self, typ: int) -> bool:
        return bool(self.lib.ds4q_can_dequantize(typ))


def source_tensor_set(src: GGUF, layer: int):
    names = {
        "gate": f"blk.{layer}.ffn_gate_exps.weight",
        "up": f"blk.{layer}.ffn_up_exps.weight",
        "down": f"blk.{layer}.ffn_down_exps.weight",
    }
    tensors = {k: src.tensors.get(v) for k, v in names.items()}
    if any(v is None for v in tensors.values()):
        return None
    expect = {
        "gate": [DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT],
        "up": [DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT],
        "down": [DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT],
    }
    for kind, dims in expect.items():
        if tensors[kind].dims != dims:
            raise SystemExit(f"{tensors[kind].name} has dims {tensors[kind].dims}, expected {dims}")
    return tensors


def build_output_tensors(src: GGUF, qlib: QuantsLib, layers: list[dict]) -> list[OutputTensor]:
    out: list[OutputTensor] = []
    for layer in layers:
        il = int(layer["layer"])
        experts = [int(e) for e in layer["experts"]]
        tensors = source_tensor_set(src, il)
        if tensors is None:
            raise SystemExit(f"source GGUF does not have routed tensors for layer {il}")
        for kind, src_t in (("gate", tensors["gate"]), ("up", tensors["up"]), ("down", tensors["down"])):
            if not qlib.can_dequantize(src_t.typ):
                raise SystemExit(f"{src_t.name} type {src_t.typ} cannot be dequantized")
            out_name = {
                "gate": f"blk.{il}.ffn_gate_exps.bitlift_q4.weight",
                "up": f"blk.{il}.ffn_up_exps.bitlift_q4.weight",
                "down": f"blk.{il}.ffn_down_exps.bitlift_q4.weight",
            }[kind]
            dims = [src_t.dims[0], src_t.dims[1], len(experts)]
            out.append(OutputTensor(
                name=out_name,
                dims=dims,
                typ=GGUF_TENSOR_Q4_K,
                nbytes=tensor_nbytes(GGUF_TENSOR_Q4_K, dims),
                source_name=src_t.name,
                source_type=src_t.typ,
                experts=experts,
            ))
        out.append(OutputTensor(
            name=f"blk.{il}.ffn_exps.bitlift_q4.ids",
            dims=[len(experts)],
            typ=GGUF_TENSOR_I32,
            nbytes=tensor_nbytes(GGUF_TENSOR_I32, [len(experts)]),
            ids=experts,
        ))
    return out


def assign_offsets(tensors: list[OutputTensor], alignment: int) -> int:
    off = 0
    for t in tensors:
        off = align_up(off, alignment)
        t.rel_offset = off
        off += t.nbytes
    return off


def expert_bytes_for(src_type: int, dims: list[int], qlib: QuantsLib) -> int:
    if len(dims) != 3:
        raise SystemExit(f"expected 3D expert tensor dims, got {dims}")
    return dims[1] * qlib.row_size(src_type, dims[0])


def write_requantized_experts(src: GGUF, qlib: QuantsLib, out_f, t: OutputTensor, row_chunk: int) -> dict:
    assert t.source_name is not None and t.source_type is not None and t.experts is not None
    src_t = src.tensors[t.source_name]
    cols = int(src_t.dims[0])
    rows = int(src_t.dims[1])
    src_row_bytes = qlib.row_size(src_t.typ, cols)
    out_row_bytes = qlib.row_size(GGUF_TENSOR_Q4_K, cols)
    src_expert_bytes = expert_bytes_for(src_t.typ, src_t.dims, qlib)
    out_start = out_f.tell()
    total_in = 0
    total_out = 0
    for expert in t.experts:
        expert_base = src_t.abs_offset + expert * src_expert_bytes
        for row0 in range(0, rows, row_chunk):
            nr = min(row_chunk, rows - row0)
            src.f.seek(expert_base + row0 * src_row_bytes)
            src_bytes = src.f.read(nr * src_row_bytes)
            if len(src_bytes) != nr * src_row_bytes:
                raise SystemExit(f"unexpected EOF while reading {src_t.name} expert {expert}")
            in_buf = ctypes.create_string_buffer(src_bytes, len(src_bytes))
            f32_buf = (ctypes.c_float * (nr * cols))()
            q4_buf = ctypes.create_string_buffer(nr * out_row_bytes)
            got = qlib.lib.ds4q_dequantize_chunk(
                src_t.typ,
                ctypes.cast(in_buf, ctypes.c_void_p),
                ctypes.cast(f32_buf, ctypes.c_void_p),
                nr,
                cols,
            )
            want_floats = nr * cols * ctypes.sizeof(ctypes.c_float)
            if int(got) != want_floats:
                raise SystemExit(f"dequantized unexpected byte count for {src_t.name}: {got} != {want_floats}")
            wrote = qlib.lib.ds4q_quantize_chunk(
                GGUF_TENSOR_Q4_K,
                ctypes.cast(f32_buf, ctypes.c_void_p),
                ctypes.cast(q4_buf, ctypes.c_void_p),
                0,
                nr,
                cols,
                None,
            )
            if int(wrote) != nr * out_row_bytes:
                raise SystemExit(f"quantized unexpected byte count for {src_t.name}: {wrote} != {nr * out_row_bytes}")
            out_f.write(q4_buf.raw)
            total_in += len(src_bytes)
            total_out += len(q4_buf.raw)
    if out_f.tell() != out_start + t.nbytes:
        raise SystemExit(f"wrote wrong byte count for {t.name}")
    return {"source_bytes": total_in, "q4_bytes": total_out}


def write_sidecar(src: GGUF, qlib: QuantsLib, out_path: Path, tensors: list[OutputTensor],
                  alignment: int, name: str, row_chunk: int) -> tuple[int, dict]:
    total_tensor_bytes = assign_offsets(tensors, alignment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"source_bytes": 0, "q4_bytes": 0}
    with out_path.open("wb") as out:
        out.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), 3))
        write_kv_string(out, "general.architecture", "deepseek4-bitlift-sidecar")
        write_kv_string(out, "general.name", name)
        write_kv_u32(out, "general.alignment", alignment)
        for t in tensors:
            write_str(out, t.name)
            out.write(struct.pack("<I", len(t.dims)))
            for d in t.dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<IQ", t.typ, t.rel_offset))
        pad = align_up(out.tell(), alignment) - out.tell()
        if pad:
            out.write(b"\0" * pad)
        data_start = out.tell()
        for i, t in enumerate(tensors, start=1):
            want = data_start + t.rel_offset
            if out.tell() < want:
                out.write(b"\0" * (want - out.tell()))
            elif out.tell() != want:
                raise SystemExit("internal offset accounting error")
            if t.ids is not None:
                out.write(struct.pack("<" + "i" * len(t.ids), *t.ids))
            else:
                layer = t.name.split(".")[1]
                print(f"[{i}/{len(tensors)}] requant {t.name} layer={layer} experts={len(t.experts or [])}", flush=True)
                part = write_requantized_experts(src, qlib, out, t, row_chunk)
                stats["source_bytes"] += part["source_bytes"]
                stats["q4_bytes"] += part["q4_bytes"]
            if out.tell() != want + t.nbytes:
                raise SystemExit(f"wrote wrong byte count for {t.name}")
    return total_tensor_bytes, stats


def source_type_summary(tensors: list[OutputTensor]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tensors:
        if t.source_type is None:
            continue
        out[str(t.source_type)] = out.get(str(t.source_type), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="DS4 Korean bitlift Q4 sidecar from base")
    ap.add_argument("--layers", help="Comma/range filter, e.g. 37,38,40-42")
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument("--build-dir", type=Path, default=Path("runs/bitlift_quants_lib"))
    ap.add_argument("--rebuild-quants-lib", action="store_true")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()

    if args.row_chunk <= 0:
        raise SystemExit("--row-chunk must be positive")

    lib_path = build_quants_lib(args.build_dir, force=args.rebuild_quants_lib)
    qlib = QuantsLib(lib_path)
    plan = load_plan(args.plan)
    layers = choose_layers(plan, parse_layers(args.layers))
    src = GGUF(args.base)
    try:
        tensors = build_output_tensors(src, qlib, layers)
        if not tensors:
            raise SystemExit("no sidecar tensors to write")
        total_tensor_bytes, stats = write_sidecar(src, qlib, args.out, tensors, src.alignment, args.name, args.row_chunk)
    finally:
        src.close()

    layer_ids = sorted({int(t.name.split(".")[1]) for t in tensors if t.name.startswith("blk.")})
    expert_slots = sum(t.dims[0] for t in tensors if t.name.endswith(".ids"))
    summary = {
        "schema": "ds4-bitlift-sidecar-from-base-build-summary-v1",
        "out": str(args.out),
        "base": str(args.base),
        "plan": str(args.plan),
        "quants_lib": str(lib_path),
        "row_chunk": args.row_chunk,
        "layers": layer_ids,
        "layer_count": len(layer_ids),
        "expert_slot_count": expert_slots,
        "tensor_count": len(tensors),
        "source_type_tensor_counts": source_type_summary(tensors),
        "tensor_payload_bytes": total_tensor_bytes,
        "file_bytes": args.out.stat().st_size,
        **stats,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
