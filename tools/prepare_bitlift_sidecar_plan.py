#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


DS4_N_LAYER = 43
DS4_N_EXPERT = 256


def load_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") == "ds4-ko-bitlift-manifest-v1":
        return data
    if "layers" in data and isinstance(data["layers"], dict):
        by_layer = [
            {"layer": int(layer), "experts": experts}
            for layer, experts in data["layers"].items()
        ]
        data = {
            **data,
            "schema": "ds4-ko-bitlift-manifest-v1",
            "by_layer": by_layer,
            "pair_count": sum(len(x["experts"]) for x in by_layer),
        }
        return data
    if data.get("schema") != "ds4-ko-bitlift-manifest-v1":
        raise SystemExit(f"unsupported manifest schema: {data.get('schema')!r}")
    layers = data.get("by_layer")
    if not isinstance(layers, list):
        raise SystemExit("manifest is missing by_layer list")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_layers(manifest):
    out = []
    seen_layers = set()
    for row in manifest["by_layer"]:
        layer = int(row["layer"])
        if layer < 0 or layer >= DS4_N_LAYER:
            raise SystemExit(f"invalid layer index: {layer}")
        if layer in seen_layers:
            raise SystemExit(f"duplicate layer entry: {layer}")
        seen_layers.add(layer)

        experts = [int(e) for e in row.get("experts", [])]
        if not experts:
            continue
        if len(experts) > DS4_N_EXPERT:
            raise SystemExit(f"layer {layer} has too many experts: {len(experts)}")
        if len(set(experts)) != len(experts):
            raise SystemExit(f"layer {layer} repeats an expert id")
        bad = [e for e in experts if e < 0 or e >= DS4_N_EXPERT]
        if bad:
            raise SystemExit(f"layer {layer} has invalid expert ids: {bad[:8]}")

        out.append({
            "layer": layer,
            "expert_count": len(experts),
            "experts": experts,
            "tensors": {
                "gate": f"blk.{layer}.ffn_gate_exps.bitlift_q4.weight",
                "up": f"blk.{layer}.ffn_up_exps.bitlift_q4.weight",
                "down": f"blk.{layer}.ffn_down_exps.bitlift_q4.weight",
                "ids": f"blk.{layer}.ffn_exps.bitlift_q4.ids",
            },
        })
    return sorted(out, key=lambda x: x["layer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar-qtype", default="q4_k")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    layers = normalize_layers(manifest)
    pair_count = sum(layer["expert_count"] for layer in layers)
    if int(manifest.get("pair_count", pair_count)) != pair_count:
        raise SystemExit(f"pair_count mismatch: manifest={manifest.get('pair_count')} computed={pair_count}")

    plan = {
        "schema": "ds4-bitlift-sidecar-plan-v1",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "manifest_name": manifest.get("name"),
        "sidecar_qtype": args.sidecar_qtype,
        "runtime_tensor_contract": {
            "gate": "blk.N.ffn_gate_exps.bitlift_q4.weight",
            "up": "blk.N.ffn_up_exps.bitlift_q4.weight",
            "down": "blk.N.ffn_down_exps.bitlift_q4.weight",
            "ids": "blk.N.ffn_exps.bitlift_q4.ids",
            "ids_dtype": "i32",
            "lookup": "runtime builds expert_id -> sidecar_slot per layer",
        },
        "layer_count": len(layers),
        "expert_slot_count": pair_count,
        "tensor_count_to_add": len(layers) * 4,
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "plan": str(args.out),
        "layer_count": plan["layer_count"],
        "expert_slot_count": plan["expert_slot_count"],
        "tensor_count_to_add": plan["tensor_count_to_add"],
        "manifest_sha256": plan["source_manifest_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
