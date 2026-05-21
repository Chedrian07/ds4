#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path("/Users/kch3dri4n/llm_provide/ds4")
DEFAULT_OUT = Path("/tmp/ds4-ko-cal/kmmlu_sample100_20260519")

MODELS = {
    "base": "ds4flash.gguf",
    "worst5q4": "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf",
    "latestable5q4": "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf",
    "layer10q4": "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf",
    "thinktop32_sidecar": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf",
    },
    "mixed32_sidecar": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-Mixed32-from-base.sidecar.gguf",
    },
    "thinktop32_late5_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf",
    },
    "thinktop32_l10_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop32-L10-HF-FP4.sidecar.gguf",
    },
    "thinktop64_l10_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop64-L10-HF-FP4.sidecar.gguf",
    },
    "thinktop128_l10_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop128-L10-HF-FP4.sidecar.gguf",
    },
    "full256_l10_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-Full256-L10-HF-FP4.sidecar.gguf",
    },
    "thinktop64_l10_base_fp8_q4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop64-L10-BaseFP8-Q4.sidecar.gguf",
    },
    "thinktop128_l10_base_fp8_q4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop128-L10-BaseFP8-Q4.sidecar.gguf",
    },
    "full256_l10_base_fp8_q4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-Full256-L10-BaseFP8-Q4.sidecar.gguf",
    },
    "thinktop32_l8_l12_hf_fp4": {
        "base": "ds4flash.gguf",
        "sidecar": "gguf/DeepSeek-V4-Flash-KR-ThinkTop32-L8-L12-HF-FP4.sidecar.gguf",
    },
}


def model_cli_args(model_spec):
    if isinstance(model_spec, dict):
        return ["-m", model_spec["base"], "--bitlift-sidecar", model_spec["sidecar"]]
    return ["-m", model_spec]


def escape_tsv(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def unescape_model_text(text: str) -> str:
    return text.strip()


def answer_to_digit(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if s in {"1", "2", "3", "4"}:
        return s
    if s.upper() in {"A", "B", "C", "D"}:
        return str("ABCD".index(s.upper()) + 1)
    return None


def extract_prediction(output: str):
    text = unescape_model_text(output)
    m = re.search(r"(?<!\d)([1-4])(?!\d)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return str("ABCD".index(m.group(1).upper()) + 1)
    return None


def format_prompt(row):
    return (
        "다음 KMMLU 객관식 문제를 풀고 정답 번호만 출력하세요.\n"
        "설명, 문장, 마침표 없이 1, 2, 3, 4 중 하나만 출력하세요.\n\n"
        f"분야: {row['category']}\n"
        f"문제: {row['question']}\n"
        f"1. {row['A']}\n"
        f"2. {row['B']}\n"
        f"3. {row['C']}\n"
        f"4. {row['D']}\n\n"
        "정답 번호:"
    )


def list_test_files():
    api = HfApi()
    files = api.list_repo_files("HAERAE-HUB/KMMLU", repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith("data/") and f.endswith("-test.csv")
    )


def load_rows(cache_dir: Path):
    rows = []
    for filename in list_test_files():
        path = hf_hub_download(
            repo_id="HAERAE-HUB/KMMLU",
            repo_type="dataset",
            filename=filename,
            local_dir=str(cache_dir),
        )
        subject = filename.removeprefix("data/").removesuffix("-test.csv")
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                answer = answer_to_digit(r.get("answer"))
                if not answer:
                    continue
                rows.append({
                    "source_file": filename,
                    "subject": subject,
                    "row_index": i,
                    "id": f"kmmlu-{subject}-{i:05d}",
                    "question": r.get("question", ""),
                    "A": r.get("A", ""),
                    "B": r.get("B", ""),
                    "C": r.get("C", ""),
                    "D": r.get("D", ""),
                    "category": r.get("Category") or subject.replace("-", " "),
                    "human_accuracy": r.get("Human Accuracy", ""),
                    "answer": answer,
                })
    return rows


def sample_rows(rows, n, seed):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:n]


def write_batch_tsv(path: Path, items):
    system = (
        "You are a Korean multiple-choice exam solver. "
        "Follow the user instruction exactly and answer with only one digit."
    )
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                "\t".join([
                    escape_tsv(item["id"]),
                    "8",
                    escape_tsv(system),
                    escape_tsv(format_prompt(item)),
                ])
                + "\n"
            )


def run_model(model_name, model_path, items, out_dir: Path, ctx: int):
    batch_stem = f"batch_kmmlu{len(items)}_{model_name}"
    batch_in = out_dir / f"{batch_stem}.tsv"
    batch_out = out_dir / f"{batch_stem}.jsonl"
    batch_err = out_dir / f"{batch_stem}.stderr.log"
    batch_out.unlink(missing_ok=True)
    batch_err.unlink(missing_ok=True)
    write_batch_tsv(batch_in, items)

    cmd = [
        "./ds4",
        "--metal",
        *model_cli_args(model_path),
        "-c",
        str(ctx),
        "--nothink",
        "--temp",
        "0",
        "--seed",
        "1",
        "--batch-prompts-tsv",
        str(batch_in),
        "--batch-output-jsonl",
        str(batch_out),
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=36000,
    )
    wall = time.time() - t0
    batch_err.write_text(proc.stderr, encoding="utf-8")

    rows = []
    if batch_out.exists():
        with batch_out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if proc.returncode != 0 and not rows:
        raise RuntimeError(
            f"KMMLU batch failed for {model_name}: {proc.returncode}\n{proc.stderr[-3000:]}"
        )

    item_by_id = {x["id"]: x for x in items}
    results = []
    for row in rows:
        item = item_by_id.get(row.get("id"))
        if not item:
            continue
        output = row.get("output") or ""
        pred = extract_prediction(output)
        results.append({
            **item,
            "model": model_name,
            "prediction": pred,
            "correct": pred == item["answer"],
            "output": output.strip(),
            "returncode": row.get("returncode"),
            "prompt_tokens": row.get("prompt_tokens"),
            "generated_tokens": row.get("generated_tokens"),
            "prefill_tps": row.get("prefill_tps"),
            "generation_tps": row.get("generation_tps"),
            "prefill_seconds": row.get("prefill_seconds"),
            "decode_seconds": row.get("decode_seconds"),
            "batch_wall_seconds": wall,
        })
    return results


def summarize(results, selected_models):
    by_model = {}
    for model in selected_models:
        xs = [r for r in results if r["model"] == model]
        by_model[model] = {
            "n": len(xs),
            "correct": sum(1 for r in xs if r["correct"]),
            "accuracy": (sum(1 for r in xs if r["correct"]) / len(xs)) if xs else None,
            "invalid_predictions": sum(1 for r in xs if r["prediction"] is None),
            "avg_prefill_tps": avg(r.get("prefill_tps") for r in xs),
            "avg_generation_tps": avg(r.get("generation_tps") for r in xs),
            "avg_prompt_tokens": avg(r.get("prompt_tokens") for r in xs),
            "avg_generated_tokens": avg(r.get("generated_tokens") for r in xs),
            "subject_accuracy": subject_summary(xs),
            "answer_distribution": dict(Counter(r["prediction"] or "INVALID" for r in xs)),
        }
    return by_model


def avg(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def subject_summary(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["subject"]].append(r)
    out = {}
    for subject, xs in sorted(buckets.items()):
        out[subject] = {
            "n": len(xs),
            "correct": sum(1 for r in xs if r["correct"]),
            "accuracy": sum(1 for r in xs if r["correct"]) / len(xs),
        }
    return out


def write_scores_csv(path: Path, rows):
    fields = [
        "id", "model", "subject", "category", "row_index", "answer", "prediction",
        "correct", "human_accuracy", "prompt_tokens", "generated_tokens",
        "prefill_tps", "generation_tps", "prefill_seconds", "decode_seconds", "output",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_report(path: Path, summary, selected_models, sample_path: Path, n: int):
    lines = [
        f"# KMMLU Sample {n} Evaluation",
        "",
        f"- Dataset: `HAERAE-HUB/KMMLU` test CSV files",
        f"- Sample file: `{sample_path}`",
        "- Prompt mode: `--nothink`, greedy `--temp 0`, max tokens per item `8`",
        "- Scoring: first generated `1~4` or `A~D` is normalized to answer digit",
        "",
        "## Summary",
        "",
        "| model | correct / n | accuracy | invalid | avg prefill t/s | avg decode t/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in selected_models:
        s = summary[model]
        lines.append(
            f"| {model} | {s['correct']} / {s['n']} | {s['accuracy']:.3f} | "
            f"{s['invalid_predictions']} | {s['avg_prefill_tps']:.2f} | {s['avg_generation_tps']:.2f} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(f"- This is a deterministic {n}-question local sample, not the full KMMLU benchmark.")
    lines.append("- Accuracy can move noticeably with a different random seed or subject stratification.")
    lines.append("- Use this as a local regression signal between GGUF variants, not as a public leaderboard number.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--models", default="base,worst5q4,latestable5q4")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_path = out_dir / f"kmmlu_sample{args.n}.jsonl"
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    if sample_path.exists() and not args.rerun:
        sample = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = load_rows(cache_dir)
        sample = sample_rows(rows, args.n, args.seed)
        sample_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in sample) + "\n",
            encoding="utf-8",
        )

    all_results = []
    raw_path = out_dir / "raw_results.jsonl"
    if args.rerun:
        raw_path.unlink(missing_ok=True)

    done = set()
    if raw_path.exists() and not args.rerun:
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["model"], r["id"]))
                all_results.append(r)

    with raw_path.open("a", encoding="utf-8") as f:
        for model_name in selected_models:
            todo = [x for x in sample if (model_name, x["id"]) not in done]
            if not todo:
                continue
            print(f"RUN KMMLU model={model_name} n={len(todo)}", flush=True)
            results = run_model(model_name, MODELS[model_name], todo, out_dir, args.ctx)
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            all_results.extend(results)

    filtered = [r for r in all_results if r["model"] in selected_models]
    write_scores_csv(out_dir / "scores.csv", filtered)
    summary = summarize(filtered, selected_models)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", summary, selected_models, sample_path, len(sample))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
