#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_STRING = 8
GGUF_TENSOR_Q4_K = 12
GGUF_TENSOR_I32 = 26
GGUF_BLOCK = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 40),
    10: (256, 84),
    11: (256, 110),
    GGUF_TENSOR_Q4_K: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 110),
    20: (256, 50),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    24: (1, 1),
    25: (1, 2),
    GGUF_TENSOR_I32: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, 56),
    30: (1, 2),
}
DS4_N_EMBD = 4096
DS4_N_FF_EXP = 2048
DS4_N_EXPERT = 256


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    typ: int
    rel_offset: int
    abs_offset: int
    nbytes: int


@dataclass
class OutputTensor:
    name: str
    dims: list[int]
    typ: int
    nbytes: int
    rel_offset: int = 0
    source: tuple[str, list[int]] | None = None
    ids: list[int] | None = None


class GGUF:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("rb")
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.alignment = 32
        self.tensor_data_pos = 0
        self.tensors: dict[str, TensorInfo] = {}
        self._parse()

    def close(self):
        self.f.close()

    def read_u32(self) -> int:
        return struct.unpack("<I", self.f.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.f.read(8))[0]

    def read_str(self) -> str:
        n = self.read_u64()
        return self.f.read(n).decode("utf-8")

    def skip_value(self, typ: int):
        sizes = {
            0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
            10: 8, 11: 8, 12: 8,
        }
        if typ in sizes:
            self.f.seek(sizes[typ], os.SEEK_CUR)
            return
        if typ == GGUF_VALUE_STRING:
            n = self.read_u64()
            self.f.seek(n, os.SEEK_CUR)
            return
        if typ == 9:
            item_type = self.read_u32()
            n = self.read_u64()
            if item_type in sizes:
                self.f.seek(sizes[item_type] * n, os.SEEK_CUR)
            else:
                for _ in range(n):
                    self.skip_value(item_type)
            return
        raise SystemExit(f"unsupported GGUF metadata value type {typ}")

    def _parse(self):
        magic = self.read_u32()
        if magic != GGUF_MAGIC:
            raise SystemExit(f"{self.path} is not GGUF")
        self.version = self.read_u32()
        if self.version != GGUF_VERSION:
            raise SystemExit(f"{self.path} is GGUF v{self.version}, expected v3")
        self.n_tensors = self.read_u64()
        self.n_kv = self.read_u64()
        for _ in range(self.n_kv):
            key = self.read_str()
            typ = self.read_u32()
            if key == "general.alignment" and typ == GGUF_VALUE_UINT32:
                pos = self.f.tell()
                self.alignment = self.read_u32()
                self.f.seek(pos)
            self.skip_value(typ)
        for _ in range(self.n_tensors):
            name = self.read_str()
            nd = self.read_u32()
            dims = [self.read_u64() for _ in range(nd)]
            typ = self.read_u32()
            rel_offset = self.read_u64()
            nbytes = tensor_nbytes(typ, dims)
            self.tensors[name] = TensorInfo(
                name=name,
                dims=dims,
                typ=typ,
                rel_offset=rel_offset,
                abs_offset=0,
                nbytes=nbytes,
            )
        self.tensor_data_pos = align_up(self.f.tell(), self.alignment)
        for t in self.tensors.values():
            t.abs_offset = self.tensor_data_pos + t.rel_offset


def align_up(v: int, a: int) -> int:
    rem = v % a
    return v if rem == 0 else v + a - rem


def tensor_nbytes(typ: int, dims: list[int]) -> int:
    elems = 1
    for d in dims:
        elems *= d
    try:
        block_elems, block_bytes = GGUF_BLOCK[typ]
    except KeyError as exc:
        raise SystemExit(f"unsupported tensor type {typ}") from exc
    return ((elems + block_elems - 1) // block_elems) * block_bytes


def row_bytes_q4(cols: int) -> int:
    if cols % 256 != 0:
        raise SystemExit(f"Q4_K row width must be divisible by 256, got {cols}")
    return (cols // 256) * 144


def expert_bytes_for(t: TensorInfo) -> int:
    if t.typ != GGUF_TENSOR_Q4_K or len(t.dims) != 3:
        raise SystemExit(f"{t.name} is not a 3D Q4_K expert tensor")
    return t.dims[1] * row_bytes_q4(t.dims[0])


def write_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key: str, value: str):
    write_str(f, key)
    f.write(struct.pack("<I", GGUF_VALUE_STRING))
    write_str(f, value)


def write_kv_u32(f, key: str, value: int):
    write_str(f, key)
    f.write(struct.pack("<I", GGUF_VALUE_UINT32))
    f.write(struct.pack("<I", value))


def copy_exact(src, dst, offset: int, nbytes: int, chunk_size: int = 1024 * 1024):
    src.seek(offset)
    left = nbytes
    while left:
        chunk = src.read(min(chunk_size, left))
        if not chunk:
            raise SystemExit("unexpected EOF while copying tensor bytes")
        dst.write(chunk)
        left -= len(chunk)


def load_plan(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "ds4-bitlift-sidecar-plan-v1":
        raise SystemExit(f"unsupported sidecar plan schema: {data.get('schema')!r}")
    return data


def choose_layers(plan: dict, only_layers: set[int] | None) -> list[dict]:
    layers = plan.get("layers", [])
    out = []
    for layer in layers:
        il = int(layer["layer"])
        if only_layers is not None and il not in only_layers:
            continue
        experts = [int(e) for e in layer.get("experts", [])]
        if not experts:
            continue
        if len(set(experts)) != len(experts):
            raise SystemExit(f"layer {il} repeats expert ids")
        bad = [e for e in experts if e < 0 or e >= DS4_N_EXPERT]
        if bad:
            raise SystemExit(f"layer {il} has invalid expert ids: {bad[:8]}")
        out.append({"layer": il, "experts": experts})
    return out


def source_tensor_set(src: GGUF, layer: int):
    names = {
        "gate": f"blk.{layer}.ffn_gate_exps.weight",
        "up": f"blk.{layer}.ffn_up_exps.weight",
        "down": f"blk.{layer}.ffn_down_exps.weight",
    }
    tensors = {k: src.tensors.get(v) for k, v in names.items()}
    if any(v is None for v in tensors.values()):
        return None
    if tensors["gate"].typ != GGUF_TENSOR_Q4_K or tensors["up"].typ != GGUF_TENSOR_Q4_K or tensors["down"].typ != GGUF_TENSOR_Q4_K:
        return None
    expect = {
        "gate": [DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT],
        "up": [DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT],
        "down": [DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT],
    }
    for k, dims in expect.items():
        if tensors[k].dims != dims:
            raise SystemExit(f"{tensors[k].name} has dims {tensors[k].dims}, expected {dims}")
    return tensors


def build_output_tensors(src: GGUF, layers: list[dict], allow_missing: bool):
    out: list[OutputTensor] = []
    skipped = []
    for layer in layers:
        il = layer["layer"]
        experts = layer["experts"]
        tensors = source_tensor_set(src, il)
        if tensors is None:
            if allow_missing:
                skipped.append(il)
                continue
            raise SystemExit(f"source GGUF does not have Q4 routed tensors for layer {il}")
        n = len(experts)
        for kind, src_t in (("gate", tensors["gate"]), ("up", tensors["up"]), ("down", tensors["down"])):
            out_name = {
                "gate": f"blk.{il}.ffn_gate_exps.bitlift_q4.weight",
                "up": f"blk.{il}.ffn_up_exps.bitlift_q4.weight",
                "down": f"blk.{il}.ffn_down_exps.bitlift_q4.weight",
            }[kind]
            dims = [src_t.dims[0], src_t.dims[1], n]
            out.append(OutputTensor(
                name=out_name,
                dims=dims,
                typ=GGUF_TENSOR_Q4_K,
                nbytes=tensor_nbytes(GGUF_TENSOR_Q4_K, dims),
                source=(src_t.name, experts),
            ))
        ids_dims = [n]
        out.append(OutputTensor(
            name=f"blk.{il}.ffn_exps.bitlift_q4.ids",
            dims=ids_dims,
            typ=GGUF_TENSOR_I32,
            nbytes=tensor_nbytes(GGUF_TENSOR_I32, ids_dims),
            ids=experts,
        ))
    return out, skipped


def assign_offsets(tensors: list[OutputTensor], alignment: int):
    off = 0
    for t in tensors:
        off = align_up(off, alignment)
        t.rel_offset = off
        off += t.nbytes
    return off


def write_sidecar(src: GGUF, out_path: Path, tensors: list[OutputTensor], alignment: int, name: str):
    total_tensor_bytes = assign_offsets(tensors, alignment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        for t in tensors:
            want = data_start + t.rel_offset
            if out.tell() < want:
                out.write(b"\0" * (want - out.tell()))
            elif out.tell() != want:
                raise SystemExit("internal offset accounting error")
            if t.ids is not None:
                out.write(struct.pack("<" + "i" * len(t.ids), *t.ids))
            else:
                src_name, experts = t.source
                src_t = src.tensors[src_name]
                expert_bytes = expert_bytes_for(src_t)
                for expert in experts:
                    copy_exact(src.f, out, src_t.abs_offset + expert * expert_bytes, expert_bytes)
            if out.tell() != want + t.nbytes:
                raise SystemExit(f"wrote wrong byte count for {t.name}")
    return total_tensor_bytes


def parse_layers(s: str | None) -> set[int] | None:
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, help="Reserved for provenance; not copied into sidecar.")
    ap.add_argument("--source-q4", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="DS4 Korean bitlift Q4 sidecar")
    ap.add_argument("--layers", help="Comma/range filter, e.g. 37,38,40-42")
    ap.add_argument("--allow-missing-source-q4", action="store_true")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()

    plan = load_plan(args.plan)
    layers = choose_layers(plan, parse_layers(args.layers))
    src = GGUF(args.source_q4)
    try:
        tensors, skipped = build_output_tensors(src, layers, args.allow_missing_source_q4)
        if not tensors:
            raise SystemExit("no sidecar tensors to write")
        total_tensor_bytes = write_sidecar(src, args.out, tensors, src.alignment, args.name)
    finally:
        src.close()

    layer_ids = sorted({int(t.name.split(".")[1]) for t in tensors if t.name.startswith("blk.")})
    expert_slots = sum(t.dims[0] for t in tensors if t.name.endswith(".ids"))
    summary = {
        "schema": "ds4-bitlift-sidecar-build-summary-v1",
        "out": str(args.out),
        "source_q4": str(args.source_q4),
        "base": str(args.base) if args.base else None,
        "plan": str(args.plan),
        "layers": layer_ids,
        "layer_count": len(layer_ids),
        "expert_slot_count": expert_slots,
        "tensor_count": len(tensors),
        "tensor_payload_bytes": total_tensor_bytes,
        "file_bytes": args.out.stat().st_size,
        "skipped_layers_missing_source_q4": skipped,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
