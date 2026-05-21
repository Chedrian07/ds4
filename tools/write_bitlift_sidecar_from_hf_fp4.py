#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from write_bitlift_sidecar_gguf import (
    DS4_N_EMBD,
    DS4_N_FF_EXP,
    GGUF_MAGIC,
    GGUF_VERSION,
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
)
from write_bitlift_sidecar_from_base_gguf import build_quants_lib


HF_PARTS = {
    "gate": "w1",
    "up": "w3",
    "down": "w2",
}


@dataclass
class SafetensorEntry:
    dtype: str
    shape: list[int]
    begin: int
    end: int


@dataclass
class SafetensorShard:
    path: Path
    data_start: int
    tensors: dict[str, SafetensorEntry]

    @classmethod
    def open(cls, path: Path) -> "SafetensorShard":
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        tensors: dict[str, SafetensorEntry] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            tensors[name] = SafetensorEntry(
                dtype=info["dtype"],
                shape=[int(x) for x in info["shape"]],
                begin=int(begin),
                end=int(end),
            )
        return cls(path=path, data_start=8 + int(header_len), tensors=tensors)


@dataclass
class OutputTensor:
    name: str
    dims: list[int]
    typ: int
    nbytes: int
    rel_offset: int = 0
    layer: int | None = None
    kind: str | None = None
    experts: list[int] | None = None
    ids: list[int] | None = None


class HfIndex:
    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir
        data = json.loads((hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = data["weight_map"]
        self.shards: dict[str, SafetensorShard] = {}

    def shard_for(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise SystemExit(f"HF tensor not found in index: {name}") from exc

    def local_path_for(self, shard_name: str) -> Path:
        return self.hf_dir / shard_name

    def local_has(self, name: str) -> bool:
        shard_name = self.shard_for(name)
        return self.local_path_for(shard_name).exists()

    def tensor(self, name: str) -> tuple[SafetensorShard, SafetensorEntry]:
        shard_name = self.shard_for(name)
        path = self.local_path_for(shard_name)
        if not path.exists():
            raise SystemExit(f"required shard is missing locally: {path}")
        shard = self.shards.get(shard_name)
        if shard is None:
            shard = SafetensorShard.open(path)
            self.shards[shard_name] = shard
        try:
            return shard, shard.tensors[name]
        except KeyError as exc:
            raise SystemExit(f"HF tensor {name} missing in local shard {path}") from exc


class QuantsLib:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_init.restype = None
        self.lib.ds4q_quantize_fp4_e8m0_to_q4_k_chunk.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        self.lib.ds4q_quantize_fp4_e8m0_to_q4_k_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(GGUF_TENSOR_Q4_K)

    def row_size(self, typ: int, cols: int) -> int:
        out = int(self.lib.ds4q_row_size(typ, cols))
        if out <= 0:
            raise SystemExit(f"unsupported row layout type={typ} cols={cols}")
        return out


def hf_names(layer: int, expert: int, kind: str) -> tuple[str, str]:
    part = HF_PARTS[kind]
    prefix = f"layers.{layer}.ffn.experts.{expert}.{part}"
    return f"{prefix}.weight", f"{prefix}.scale"


def validate_source_tensor(idx: HfIndex, layer: int, expert: int, kind: str) -> tuple[int, int]:
    weight_name, scale_name = hf_names(layer, expert, kind)
    _, w = idx.tensor(weight_name)
    _, s = idx.tensor(scale_name)
    if w.dtype != "I8" or s.dtype != "F8_E8M0":
        raise SystemExit(f"{weight_name} / {scale_name} must be I8 + F8_E8M0, got {w.dtype} + {s.dtype}")
    if len(w.shape) != 2 or len(s.shape) != 2:
        raise SystemExit(f"{weight_name} must be 2D packed FP4")
    rows = w.shape[0]
    packed_cols = w.shape[1]
    cols = packed_cols * 2
    if cols % 32:
        raise SystemExit(f"{weight_name} logical cols not divisible by 32: {cols}")
    if s.shape != [rows, cols // 32]:
        raise SystemExit(f"{scale_name} shape {s.shape} does not match packed tensor {w.shape}")
    expected = {
        "gate": (DS4_N_FF_EXP, DS4_N_EMBD),
        "up": (DS4_N_FF_EXP, DS4_N_EMBD),
        "down": (DS4_N_EMBD, DS4_N_FF_EXP),
    }[kind]
    if (rows, cols) != expected:
        raise SystemExit(f"{weight_name} logical shape rows/cols {(rows, cols)} != expected {expected}")
    return rows, cols


def build_output_tensors(idx: HfIndex, layers: list[dict]) -> list[OutputTensor]:
    out: list[OutputTensor] = []
    for layer in layers:
        il = int(layer["layer"])
        experts = [int(e) for e in layer["experts"]]
        if not experts:
            continue
        # Validate one representative per kind and local shard availability for every expert.
        for kind in ("gate", "up", "down"):
            for expert in experts:
                weight_name, scale_name = hf_names(il, expert, kind)
                if not idx.local_has(weight_name) or not idx.local_has(scale_name):
                    raise SystemExit(f"layer {il} expert {expert} {kind} is not fully available locally")
                validate_source_tensor(idx, il, expert, kind)
            dims = {
                "gate": [DS4_N_EMBD, DS4_N_FF_EXP, len(experts)],
                "up": [DS4_N_EMBD, DS4_N_FF_EXP, len(experts)],
                "down": [DS4_N_FF_EXP, DS4_N_EMBD, len(experts)],
            }[kind]
            out.append(OutputTensor(
                name={
                    "gate": f"blk.{il}.ffn_gate_exps.bitlift_q4.weight",
                    "up": f"blk.{il}.ffn_up_exps.bitlift_q4.weight",
                    "down": f"blk.{il}.ffn_down_exps.bitlift_q4.weight",
                }[kind],
                dims=dims,
                typ=GGUF_TENSOR_Q4_K,
                nbytes=tensor_nbytes(GGUF_TENSOR_Q4_K, dims),
                layer=il,
                kind=kind,
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


def read_exact(path: Path, offset: int, nbytes: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise SystemExit(f"unexpected EOF while reading {path}")
    return data


def write_quantized_experts(idx: HfIndex, qlib: QuantsLib, out_f, t: OutputTensor, row_chunk: int) -> dict:
    assert t.layer is not None and t.kind is not None and t.experts is not None
    cols = int(t.dims[0])
    rows = int(t.dims[1])
    out_row_bytes = qlib.row_size(GGUF_TENSOR_Q4_K, cols)
    out_start = out_f.tell()
    source_bytes = 0
    q4_bytes = 0
    for expert in t.experts:
        weight_name, scale_name = hf_names(t.layer, expert, t.kind)
        w_shard, w = idx.tensor(weight_name)
        s_shard, s = idx.tensor(scale_name)
        packed_cols = w.shape[1]
        scale_cols = s.shape[1]
        if w.shape[0] != rows or packed_cols * 2 != cols:
            raise SystemExit(f"shape mismatch for {weight_name}: {w.shape}, expected rows={rows} cols={cols}")
        for row0 in range(0, rows, row_chunk):
            nr = min(row_chunk, rows - row0)
            w_off = w_shard.data_start + w.begin + row0 * packed_cols
            s_off = s_shard.data_start + s.begin + row0 * scale_cols
            w_bytes = read_exact(w_shard.path, w_off, nr * packed_cols)
            s_bytes = read_exact(s_shard.path, s_off, nr * scale_cols)
            w_buf = ctypes.create_string_buffer(w_bytes, len(w_bytes))
            s_buf = ctypes.create_string_buffer(s_bytes, len(s_bytes))
            q4_buf = ctypes.create_string_buffer(nr * out_row_bytes)
            wrote = qlib.lib.ds4q_quantize_fp4_e8m0_to_q4_k_chunk(
                ctypes.cast(w_buf, ctypes.c_void_p),
                ctypes.cast(s_buf, ctypes.c_void_p),
                ctypes.cast(q4_buf, ctypes.c_void_p),
                nr,
                packed_cols,
            )
            if int(wrote) != nr * out_row_bytes:
                raise SystemExit(f"FP4->Q4_K wrote unexpected byte count for {weight_name}: {wrote}")
            out_f.write(q4_buf.raw)
            source_bytes += len(w_bytes) + len(s_bytes)
            q4_bytes += len(q4_buf.raw)
    if out_f.tell() != out_start + t.nbytes:
        raise SystemExit(f"wrote wrong byte count for {t.name}")
    return {"source_bytes": source_bytes, "q4_bytes": q4_bytes}


def write_sidecar(idx: HfIndex, qlib: QuantsLib, out_path: Path, tensors: list[OutputTensor],
                  alignment: int, name: str, row_chunk: int) -> tuple[int, dict]:
    total_tensor_bytes = assign_offsets(tensors, alignment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"source_bytes": 0, "q4_bytes": 0}
    with out_path.open("wb") as out:
        out.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), 5))
        write_kv_string(out, "general.architecture", "deepseek4-bitlift-sidecar")
        write_kv_string(out, "general.name", name)
        write_kv_u32(out, "general.alignment", alignment)
        write_kv_string(out, "bitlift.source", "hf-fp4-e8m0")
        write_kv_string(out, "bitlift.source_hf_dir", str(idx.hf_dir))
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
                print(f"[{i}/{len(tensors)}] fp4->q4 {t.name} layer={t.layer} experts={len(t.experts or [])}", flush=True)
                part = write_quantized_experts(idx, qlib, out, t, row_chunk)
                stats["source_bytes"] += part["source_bytes"]
                stats["q4_bytes"] += part["q4_bytes"]
            if out.tell() != want + t.nbytes:
                raise SystemExit(f"wrote wrong byte count for {t.name}")
    return total_tensor_bytes, stats


def filter_locally_available_layers(idx: HfIndex, layers: list[dict]) -> tuple[list[dict], list[int]]:
    available = []
    skipped = []
    for layer in layers:
        il = int(layer["layer"])
        ok = True
        for expert in layer["experts"]:
            for kind in ("gate", "up", "down"):
                weight_name, scale_name = hf_names(il, int(expert), kind)
                if not idx.local_has(weight_name) or not idx.local_has(scale_name):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            available.append(layer)
        else:
            skipped.append(il)
    return available, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="DS4 Korean bitlift Q4 sidecar from HF FP4 source")
    ap.add_argument("--layers", help="Comma/range filter, e.g. 37,38,40-42")
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument("--build-dir", type=Path, default=Path("runs/bitlift_quants_lib"))
    ap.add_argument("--rebuild-quants-lib", action="store_true")
    ap.add_argument("--skip-missing-layers", action="store_true")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()

    if args.row_chunk <= 0:
        raise SystemExit("--row-chunk must be positive")
    if not (args.hf_dir / "model.safetensors.index.json").exists():
        raise SystemExit(f"missing safetensors index in {args.hf_dir}")

    idx = HfIndex(args.hf_dir)
    plan = load_plan(args.plan)
    layers = choose_layers(plan, parse_layers(args.layers))
    if args.skip_missing_layers:
        layers, skipped = filter_locally_available_layers(idx, layers)
    else:
        skipped = []
    if not layers:
        raise SystemExit("no locally available sidecar layers to write")

    lib_path = build_quants_lib(args.build_dir, force=args.rebuild_quants_lib)
    qlib = QuantsLib(lib_path)
    tensors = build_output_tensors(idx, layers)
    total_tensor_bytes, stats = write_sidecar(idx, qlib, args.out, tensors, 32, args.name, args.row_chunk)

    layer_ids = sorted({int(t.name.split(".")[1]) for t in tensors if t.name.startswith("blk.")})
    expert_slots = sum(t.dims[0] for t in tensors if t.name.endswith(".ids"))
    shards = sorted({idx.shard_for(hf_names(il, e, kind)[0])
                     for ilayer in layers
                     for il in [int(ilayer["layer"])]
                     for e in [int(x) for x in ilayer["experts"]]
                     for kind in ("gate", "up", "down")})
    summary = {
        "schema": "ds4-bitlift-sidecar-from-hf-fp4-build-summary-v1",
        "out": str(args.out),
        "hf_dir": str(args.hf_dir),
        "plan": str(args.plan),
        "quants_lib": str(lib_path),
        "row_chunk": args.row_chunk,
        "layers": layer_ids,
        "skipped_missing_layers": sorted(set(skipped)),
        "layer_count": len(layer_ids),
        "expert_slot_count": expert_slots,
        "tensor_count": len(tensors),
        "source_format": "packed_fp4_i8_plus_f8_e8m0_scales",
        "source_shards": shards,
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
