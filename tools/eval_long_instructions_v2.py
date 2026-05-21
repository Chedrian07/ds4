#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/kch3dri4n/llm_provide/ds4")
DEFAULT_OUT = Path("/tmp/ds4-ko-cal/long_instructions_v2_20260519")

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


def hangul_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if re.match(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", c)) / len(chars)


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*[-*]\s+\S", line))


def count_steps(text: str) -> int:
    return len(re.findall(r"단계\s*\d+\s*:", text))


def has_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()]
    return any(line.startswith("|") and line.endswith("|") for line in lines)


def has_polite_style(text: str) -> bool:
    return bool(re.search(r"(습니다|합니다|드립니다|주세요|십시오|입니다)", text))


def build_prompts():
    topics = [
        ("수업 보고서", "복잡도와 순환 알고리즘 실습 결과를 보고서로 정리해야 합니다."),
        ("포트폴리오", "AI컴퓨터공학부 1학년 학생이 보안 전문가 진로를 설명해야 합니다."),
        ("회의록 정리", "팀 회의 의견을 실행 항목과 보류 항목으로 나누어야 합니다."),
        ("학습 계획", "자료구조, C언어, 리눅스 기초를 4주 동안 병행해서 공부해야 합니다."),
        ("기술 블로그", "한국어 LLM 양자화 실험을 독자가 이해하기 쉽게 정리해야 합니다."),
    ]
    prompts = []

    for i in range(15):
        topic, bg = topics[i % len(topics)]
        prompts.append({
            "id": f"longv2-basic-{i:03d}",
            "kind": "basic_plan",
            "n": 360,
            "title_max": 20,
            "exact_bullets": 3,
            "min_steps": 3,
            "requires_risk": False,
            "requires_validation": False,
            "requires_terms": [],
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 답변을 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}\n\n"
                "작성 조건:\n1. 첫 줄에 20자 이내 제목을 씁니다.\n2. 첫 문단은 문제 상황을 2문장으로 요약합니다.\n"
                "3. 핵심 목표를 정확히 3개 bullet로 정리합니다.\n4. 실행 계획은 '단계 N:' 형식으로 3단계 이상 작성합니다.\n"
                "5. 표는 사용하지 마세요.\n6. 마지막 문단은 앞으로 바로 할 수 있는 첫 행동 1개로 끝내세요.\n"
                "7. 전체 답변은 한국어 공손체로 작성하세요."
            ),
        })

    for i in range(15):
        topic, bg = topics[i % len(topics)]
        prompts.append({
            "id": f"longv2-risk-{i:03d}",
            "kind": "risk_plan",
            "n": 420,
            "title_max": 20,
            "exact_bullets": 3,
            "min_steps": 4,
            "requires_risk": True,
            "requires_validation": False,
            "requires_terms": [],
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 실행 계획을 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}\n\n"
                "작성 조건:\n1. 첫 줄 제목은 20자 이내로 작성합니다.\n2. 핵심 목표는 정확히 3개 bullet로 씁니다.\n"
                "3. 실행 계획은 '단계 N:' 형식으로 4단계 이상 작성합니다.\n4. 위험 요소 2개와 완화 방법 2개를 반드시 포함합니다.\n"
                "5. 확인되지 않은 성과를 단정하지 않습니다.\n6. 표는 사용하지 않습니다.\n7. 마지막 문장은 오늘 바로 할 수 있는 구체적 행동으로 끝냅니다."
            ),
        })

    term_sets = [
        ("양자화", "모델 가중치 표현 정밀도를 낮춰 용량을 줄이는 방법"),
        ("라우터", "MoE 모델에서 사용할 expert를 고르는 구성요소"),
        ("prefill", "입력 프롬프트를 한 번에 처리하는 단계"),
        ("decode", "토큰을 하나씩 생성하는 단계"),
        ("sidecar", "기존 모델 옆에 추가로 붙이는 보조 텐서 묶음"),
    ]
    for i in range(15):
        topic, bg = topics[i % len(topics)]
        term, meaning = term_sets[i % len(term_sets)]
        prompts.append({
            "id": f"longv2-terms-{i:03d}",
            "kind": "term_explain",
            "n": 420,
            "title_max": 20,
            "exact_bullets": 3,
            "min_steps": 3,
            "requires_risk": False,
            "requires_validation": False,
            "requires_terms": [term],
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 설명문을 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}\n"
                f"반드시 설명할 용어: {term} - {meaning}\n\n"
                "작성 조건:\n1. 첫 줄 제목은 20자 이내입니다.\n2. 첫 문단은 상황을 2문장으로 요약합니다.\n"
                "3. 핵심 목표를 정확히 3개 bullet로 정리합니다.\n4. 전문용어에는 괄호 안에 짧은 설명을 붙입니다.\n"
                "5. 실행 계획은 '단계 N:' 형식으로 3단계 이상 작성합니다.\n6. 표는 쓰지 않습니다.\n7. 전체 답변은 한국어 공손체로 작성합니다."
            ),
        })

    for i in range(15):
        topic, bg = topics[i % len(topics)]
        prompts.append({
            "id": f"longv2-verify-{i:03d}",
            "kind": "validation_plan",
            "n": 440,
            "title_max": 20,
            "exact_bullets": 3,
            "min_steps": 3,
            "requires_risk": False,
            "requires_validation": True,
            "requires_terms": [],
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 계획서를 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}\n\n"
                "작성 조건:\n1. 첫 줄 제목은 20자 이내입니다.\n2. 문제 상황을 2문장으로 요약합니다.\n"
                "3. 핵심 목표는 정확히 3개 bullet입니다.\n4. 실행 계획은 '단계 N:' 형식으로 3단계 이상입니다.\n"
                "5. 불확실한 내용은 단정하지 말고 '검증 필요 항목'으로 분리합니다.\n"
                "6. 표는 사용하지 않습니다.\n7. 마지막 문단은 바로 할 수 있는 첫 행동 1개로 끝냅니다."
            ),
        })

    return prompts


def write_batch_tsv(path: Path, items):
    system = "You are a helpful Korean writing assistant. Follow every formatting constraint exactly."
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                "\t".join([
                    escape_tsv(item["id"]),
                    str(item["n"]),
                    escape_tsv(system),
                    escape_tsv(item["prompt"]),
                ])
                + "\n"
            )


def score_output(item, row):
    text = (row.get("output") or "").strip()
    title = first_nonempty_line(text)
    checks = {
        "has_output": bool(text),
        "title_len_ok": bool(title) and len(title) <= item["title_max"],
        "korean_ratio_ok": hangul_ratio(text) >= 0.35,
        "polite_ok": has_polite_style(text),
        "bullet_count_ok": count_bullets(text) == item["exact_bullets"],
        "step_count_ok": count_steps(text) >= item["min_steps"],
        "no_table_ok": not has_markdown_table(text),
        "not_obviously_truncated": (row.get("generated_tokens") or 0) < item["n"],
    }
    if item["requires_risk"]:
        checks["risk_count_ok"] = len(re.findall(r"위험\s*요소|위험", text)) >= 2
        checks["mitigation_count_ok"] = len(re.findall(r"완화\s*방법|완화", text)) >= 2
    if item["requires_validation"]:
        checks["validation_section_ok"] = bool(re.search(r"검증 필요|확인 필요|검토 필요", text))
    for term in item["requires_terms"]:
        checks[f"term_{term}_ok"] = term in text and bool(re.search(re.escape(term) + r"[^\n]{0,20}\(", text))

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    return {
        "criteria_passed": passed,
        "criteria_total": total,
        "score": passed / total if total else 0.0,
        "pass": passed / total >= 0.80 if total else False,
        "checks": checks,
        "title": title,
        "bullet_count": count_bullets(text),
        "step_count": count_steps(text),
        "hangul_ratio": hangul_ratio(text),
    }


def run_model(model_name, model_path, items, out_dir: Path, ctx: int):
    batch_in = out_dir / f"batch_longv2_{model_name}.tsv"
    batch_out = out_dir / f"batch_longv2_{model_name}.jsonl"
    batch_err = out_dir / f"batch_longv2_{model_name}.stderr.log"
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
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=36000)
    wall = time.time() - t0
    batch_err.write_text(proc.stderr, encoding="utf-8")

    rows = []
    if batch_out.exists():
        with batch_out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if proc.returncode != 0 and not rows:
        raise RuntimeError(f"long v2 batch failed for {model_name}: {proc.returncode}\n{proc.stderr[-3000:]}")

    by_id = {item["id"]: item for item in items}
    results = []
    for row in rows:
        item = by_id.get(row.get("id"))
        if not item:
            continue
        scored = score_output(item, row)
        results.append({
            **{k: v for k, v in item.items() if k != "prompt"},
            "model": model_name,
            "returncode": row.get("returncode"),
            "prompt_tokens": row.get("prompt_tokens"),
            "generated_tokens": row.get("generated_tokens"),
            "prefill_tps": row.get("prefill_tps"),
            "generation_tps": row.get("generation_tps"),
            "prefill_seconds": row.get("prefill_seconds"),
            "decode_seconds": row.get("decode_seconds"),
            "batch_wall_seconds": wall,
            "output": (row.get("output") or "").strip(),
            **scored,
        })
    return results


def avg(vals):
    xs = [float(x) for x in vals if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(results, selected_models):
    summary = {}
    for model in selected_models:
        xs = [r for r in results if r["model"] == model]
        kind_summary = {}
        for kind, rows in group_by(xs, "kind").items():
            kind_summary[kind] = {
                "n": len(rows),
                "pass": sum(1 for r in rows if r["pass"]),
                "pass_rate": sum(1 for r in rows if r["pass"]) / len(rows) if rows else None,
                "avg_score": avg(r["score"] for r in rows),
            }
        summary[model] = {
            "n": len(xs),
            "pass": sum(1 for r in xs if r["pass"]),
            "pass_rate": sum(1 for r in xs if r["pass"]) / len(xs) if xs else None,
            "avg_score": avg(r["score"] for r in xs),
            "avg_prefill_tps": avg(r.get("prefill_tps") for r in xs),
            "avg_generation_tps": avg(r.get("generation_tps") for r in xs),
            "avg_generated_tokens": avg(r.get("generated_tokens") for r in xs),
            "kind_summary": kind_summary,
        }
    return summary


def group_by(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return dict(out)


def write_scores_csv(path: Path, rows):
    fields = [
        "id", "model", "kind", "pass", "score", "criteria_passed", "criteria_total",
        "title", "bullet_count", "step_count", "hangul_ratio",
        "prompt_tokens", "generated_tokens", "prefill_tps", "generation_tps",
        "prefill_seconds", "decode_seconds", "output",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary, models):
    lines = ["# DS4 Long Instruction v2 Report", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| model | pass | n | pass_rate | avg_score | prefill_tps | generation_tps | avg_generated_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in models:
        s = summary[model]
        lines.append(
            f"| {model} | {s['pass']} | {s['n']} | {s['pass_rate']:.3f} | {s['avg_score']:.3f} | "
            f"{s['avg_prefill_tps']:.2f} | {s['avg_generation_tps']:.2f} | {s['avg_generated_tokens']:.2f} |"
        )
    lines.append("")
    lines.append("## By Kind")
    lines.append("")
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| kind | pass | n | pass_rate | avg_score |")
        lines.append("|---|---:|---:|---:|---:|")
        for kind, s in sorted(summary[model]["kind_summary"].items()):
            lines.append(f"| {kind} | {s['pass']} | {s['n']} | {s['pass_rate']:.3f} | {s['avg_score']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,latestable5q4,layer10q4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model alias: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()
    write_jsonl(args.out / "long_instruction_v2_prompts.jsonl", prompts)

    all_results = []
    raw_path = args.out / "raw_results.jsonl"
    if args.rerun:
        raw_path.unlink(missing_ok=True)
    done = set()
    if raw_path.exists() and not args.rerun:
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["model"], row["id"]))
                all_results.append(row)

    with raw_path.open("a", encoding="utf-8") as raw_f:
        for model in selected:
            todo = [p for p in prompts if (model, p["id"]) not in done]
            if not todo:
                continue
            print(f"RUN long-instruction-v2 model={model} n={len(todo)}", flush=True)
            results = run_model(model, MODELS[model], todo, args.out, args.ctx)
            for row in results:
                raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            all_results.extend(results)

    write_scores_csv(args.out / "scores.csv", all_results)
    summary = summarize(all_results, selected)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out / "REPORT.md", summary, selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
