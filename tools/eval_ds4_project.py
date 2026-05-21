#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/kch3dri4n/llm_provide/ds4")
OUT = Path("/tmp/ds4-ko-cal/structured_eval_layerq4")

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


def hangul_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if re.match(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", c)) / len(chars)


def latin_ratio(text):
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if ("A" <= c <= "Z") or ("a" <= c <= "z")) / len(chars)


def cjk_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "\u4e00" <= c <= "\u9fff") / len(chars)


def sentence_count_ko(text):
    return len([x for x in re.split(r"[.!?。！？\n]+", text.strip()) if x.strip()])


def build_prompts():
    prompts = []

    situations = [
        ("group_late", "조별과제 팀원이 자료를 늦게 보내고 있습니다", "오늘 밤 9시까지 자료를 보내 달라고 요청"),
        ("prof_extension", "교수님께 과제 제출 가능 여부를 문의해야 합니다", "오늘 오후 5시까지 제출 가능 여부를 확인 요청"),
        ("club_room", "동아리 신입 부원이 회의 장소를 헷갈렸습니다", "학생회관 302호로 와 달라고 안내"),
        ("dorm_call", "룸메이트의 밤늦은 통화가 수면에 방해됩니다", "밤 11시 이후에는 복도에서 통화해 달라고 요청"),
        ("interview_change", "아르바이트 면접 시간을 바꿔야 합니다", "내일 오후 3시 또는 5시 가능 여부 확인 요청"),
    ]
    for i in range(20):
        sid, desc, ask = situations[i % len(situations)]
        prompts.append({
            "id": f"ko-held-daily-{i:03d}",
            "suite": "korean100",
            "kind": "daily",
            "n": 96,
            "prompt": f"상황: {desc}. 상대가 기분 나쁘지 않게 한국어 메시지를 3문장 이내로 작성하세요. 마지막 문장은 구체적인 요청으로 끝내세요. 요청 내용: {ask}.",
        })

    passages = [
        ("도시 열섬", "도시의 아스팔트와 콘크리트는 낮 동안 열을 저장하고 밤에도 천천히 방출한다. 차량과 냉방기의 폐열이 더해지면 주변보다 기온이 높아질 수 있다. 나무 그늘과 옥상 녹화는 열섬 완화에 도움이 된다."),
        ("온라인 수업", "온라인 수업은 이동 시간을 줄이고 녹화 강의를 반복해서 볼 수 있다는 장점이 있다. 그러나 학습 일정을 스스로 관리하지 못하면 집중도가 낮아질 수 있다. 실시간 질문과 짧은 퀴즈는 약점을 보완한다."),
        ("오픈소스", "오픈소스는 누구나 코드를 검토하고 수정할 수 있다는 특징이 있다. 버그를 빠르게 발견할 수 있지만 유지보수자가 부족하면 보안 패치가 늦어질 수 있다. 사용자는 업데이트 주기와 커뮤니티 상태도 확인해야 한다."),
        ("개인정보 보호", "개인정보 보호는 서비스를 만들 때 설계 단계부터 고려해야 한다. 필요한 정보만 수집하고 보관 기간을 제한해야 한다. 사용자가 자신의 정보를 확인하고 삭제할 수 있어야 신뢰를 지킬 수 있다."),
        ("학습 루틴", "공부 계획은 완벽함보다 지속 가능성이 중요하다. 하루에 많은 시간을 몰아서 공부하면 쉽게 지칠 수 있다. 짧은 복습과 주기적인 점검은 장기 기억에 도움이 된다."),
    ]
    for i in range(20):
        title, body = passages[i % len(passages)]
        prompts.append({
            "id": f"ko-held-summary-{i:03d}",
            "suite": "korean100",
            "kind": "summary",
            "n": 128,
            "prompt": f"다음 글을 읽고 한국어로 답하세요.\n\n[글: {title}]\n{body}\n\n작업: 제목을 붙이고 핵심 내용을 정확히 3문장으로 요약하세요. 글에 없는 내용을 추가하지 마세요.",
        })

    tech = [
        ("prepared_statement", "SQL Injection을 방어하는 prepared statement의 원리를 공격 절차 없이 설명해 주세요.", ["쿼리", "입력", "분리"]),
        ("hash_salt", "비밀번호를 평문으로 저장하면 안 되는 이유와 해시와 솔트의 개념을 설명해 주세요.", ["해시", "솔트", "평문"]),
        ("aslr", "ASLR이 메모리 공격을 어렵게 만드는 이유를 개념적으로 설명해 주세요.", ["주소", "무작위", "메모리"]),
        ("git", "Git에서 commit, branch, merge의 차이를 초보자에게 설명해 주세요.", ["commit", "branch", "merge"]),
        ("rest", "REST API에서 GET과 POST의 차이를 예시와 함께 설명해 주세요.", ["GET", "POST", "조회"]),
    ]
    for i in range(20):
        sid, task, keys = tech[i % len(tech)]
        prompts.append({
            "id": f"ko-held-tech-{i:03d}",
            "suite": "korean100",
            "kind": "tech",
            "n": 128,
            "keys": keys,
            "prompt": f"{task} 한국어로 5문장 이내로 간결하게 답하고, 불필요한 공격 절차나 과장된 표현은 넣지 마세요.",
        })

    copy_cases = [
        "가나다라마바사 | 숫자:{num} | 기호: !?.,;:()[]{{}}",
        "문장A: 나는 밥을 먹었다.\n문장B: 나는  밥을  먹었다.\n문장C: 나는\t밥을\t먹었다.\nID={num}",
        "{{\"lang\":\"ko\",\"task\":\"copy\",\"문장\":\"한국어 토큰을 보존하세요\",\"id\":{num}}}",
        "|항목|값|\n|---|---|\n|모델|DS4-KR|\n|번호|{num}|\n|비고|한글/영문 혼합|",
        "<ko><정확복사 id='{num}'>서울-대전-대구-부산 / 3.14159 / 끝.</정확복사></ko>",
    ]
    for i in range(20):
        expected = copy_cases[i % len(copy_cases)].format(num=2000 + i)
        prompts.append({
            "id": f"ko-held-copy-{i:03d}",
            "suite": "korean100",
            "kind": "exact",
            "n": 96,
            "expected": expected,
            "prompt": f"아래 [복사대상] 안의 내용을 한 글자도 바꾸지 말고 그대로 출력하세요. 설명을 붙이지 마세요.\n[복사대상]\n{expected}\n[/복사대상]",
        })

    long_topics = [
        ("수업 보고서", "복잡도와 순환 알고리즘 실습 결과를 보고서로 정리해야 합니다"),
        ("포트폴리오", "AI컴퓨터공학부 1학년 학생이 보안 전문가 진로를 설명해야 합니다"),
        ("회의록 정리", "팀 회의 의견을 실행 항목과 보류 항목으로 나누어야 합니다"),
        ("학습 계획", "자료구조, C언어, 리눅스 기초를 4주 동안 병행해서 공부해야 합니다"),
        ("기술 블로그", "한국어 LLM 양자화 실험을 독자가 이해하기 쉽게 정리해야 합니다"),
    ]
    for i in range(20):
        topic, bg = long_topics[i % len(long_topics)]
        prompts.append({
            "id": f"ko-held-long-{i:03d}",
            "suite": "korean100",
            "kind": "long",
            "n": 220,
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 답변을 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}.\n\n"
                "작성 조건:\n1. 첫 줄에 20자 이내 제목을 씁니다.\n2. 첫 문단은 문제 상황을 2문장으로 요약합니다.\n"
                "3. 핵심 목표를 정확히 3개 bullet로 정리합니다.\n4. 실행 계획은 '단계 N:' 형식으로 3단계 이상 작성합니다.\n"
                "5. 표는 사용하지 마세요.\n6. 마지막 문단은 바로 할 수 있는 첫 행동 1개로 끝내세요.\n7. 전체 답변은 한국어 공손체로 작성하세요."
            ),
        })

    # Degradation/control suites.
    for i in range(20):
        prompts.append({
            "id": f"en-control-{i:03d}",
            "suite": "control60",
            "kind": "english",
            "n": 96,
            "prompt": f"Answer in concise English. Explain one practical difference between caching and persistence for a software project. Use exactly {2 + (i % 2)} sentences.",
        })
    for i in range(20):
        prompts.append({
            "id": f"zh-control-{i:03d}",
            "suite": "control60",
            "kind": "chinese",
            "n": 96,
            "prompt": f"请用简体中文回答。用三句话解释为什么备份和版本控制不是同一件事。不要使用表格。编号：{i}",
        })
    for i in range(20):
        expected = f"CONTROL-{3000+i}: ABC xyz 12345 !? 한국어 中文"
        prompts.append({
            "id": f"fmt-control-{i:03d}",
            "suite": "control60",
            "kind": "control_exact",
            "n": 80,
            "expected": expected,
            "prompt": f"Copy the following line exactly and add no explanation:\n{expected}",
        })

    # Extra exact/long prompts beyond korean100.
    for i in range(20):
        expected = f"초성열: ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ / 모음열: ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ / 번호:{4000+i}"
        prompts.append({
            "id": f"extra-copy-{i:03d}",
            "suite": "exact_long_extra",
            "kind": "exact",
            "n": 96,
            "expected": expected,
            "prompt": f"다음 한글 자모 문자열을 그대로 반복해서 출력하세요.\n<{expected}>",
        })
    for i in range(10):
        prompts.append({
            "id": f"extra-long-{i:03d}",
            "suite": "exact_long_extra",
            "kind": "long",
            "n": 260,
            "prompt": (
                "다음 조건을 모두 만족하는 한국어 실행 계획을 작성하세요.\n"
                "상황: 한국어 모델 평가 결과를 발표 자료로 정리해야 합니다.\n"
                "조건: 첫 줄 제목은 15자 이내, 첫 문단은 2문장, 목표 bullet은 정확히 3개, "
                "각 단계는 '단계 N:'으로 시작, 위험 요소 2개와 완화 방법 2개 포함, 표 사용 금지, "
                "마지막 문장은 오늘 바로 할 수 있는 구체적 행동으로 끝내세요."
            ),
        })

    return prompts


def escape_tsv(text):
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def write_batch_tsv(path, items):
    system = "You are a helpful assistant"
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


def run_batch(model_name, model_path, items, suite_key):
    if not items:
        return []
    batch_in = OUT / f"batch_{suite_key}_{model_name}.tsv"
    batch_out = OUT / f"batch_{suite_key}_{model_name}.jsonl"
    batch_err = OUT / f"batch_{suite_key}_{model_name}.stderr.log"
    batch_out.unlink(missing_ok=True)
    batch_err.unlink(missing_ok=True)
    write_batch_tsv(batch_in, items)

    cmd = [
        "./ds4",
        "--metal",
        *model_cli_args(model_path),
        "-c",
        "4096",
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
    wall_seconds = time.time() - t0
    batch_err.write_text(proc.stderr, encoding="utf-8")
    rows = []
    if batch_out.exists():
        with batch_out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if proc.returncode != 0 and not rows:
        raise RuntimeError(f"batch generation failed for {model_name}: {proc.returncode}\n{proc.stderr[-2000:]}")

    item_by_id = {item["id"]: item for item in items}
    converted = []
    for row in rows:
        item = item_by_id.get(row.get("id"))
        if not item:
            continue
        seconds = (row.get("prefill_seconds") or 0.0) + (row.get("decode_seconds") or 0.0)
        converted.append({
            "id": item["id"],
            "suite": item["suite"],
            "kind": item["kind"],
            "model": model_name,
            "returncode": row.get("returncode", 1),
            "seconds": round(seconds, 3),
            "batch_wall_seconds": round(wall_seconds, 3),
            "prompt_tokens": row.get("prompt_tokens"),
            "generated_tokens": row.get("generated_tokens"),
            "prefill_tps": row.get("prefill_tps"),
            "generation_tps": row.get("generation_tps"),
            "output": (row.get("output") or "").strip(),
            "error": row.get("error"),
        })
    return converted


def score(item, output):
    kind = item["kind"]
    if kind in {"exact", "control_exact"}:
        return output == item.get("expected", "")
    if kind == "daily":
        return hangul_ratio(output) >= 0.65 and sentence_count_ko(output) <= 4 and ("요" in output or "습니다" in output)
    if kind == "summary":
        return hangul_ratio(output) >= 0.65 and sentence_count_ko(output) >= 3 and "제목" in output[:40]
    if kind == "tech":
        keys = item.get("keys", [])
        return hangul_ratio(output) >= 0.35 and sentence_count_ko(output) <= 7 and sum(k in output for k in keys) >= 2
    if kind == "long":
        lines = [x.rstrip() for x in output.splitlines() if x.strip()]
        title_ok = bool(lines) and len(lines[0]) <= 20 and "|" not in output
        bullets = len([x for x in lines if x.lstrip().startswith(("-", "*"))])
        steps = len(re.findall(r"단계\s*\d+:", output))
        return hangul_ratio(output) >= 0.55 and title_ok and bullets >= 3 and steps >= 3
    if kind == "english":
        return latin_ratio(output) >= 0.75 and cjk_ratio(output) < 0.05
    if kind == "chinese":
        return cjk_ratio(output) >= 0.35
    return bool(output.strip())


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", required=True)
    ap.add_argument("--models", default="base,worst5q4")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = [p for p in build_prompts() if p["suite"] in set(args.suite)]
    prompt_by_id = {p["id"]: p for p in prompts}
    (OUT / "prompts.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in prompts) + "\n")

    result_path = OUT / "raw_results.jsonl"
    if args.rerun and result_path.exists():
        result_path.unlink()
    done = set()
    if result_path.exists() and not args.rerun:
        with result_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("suite") in set(args.suite):
                    done.add((r["model"], r["id"]))

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    suite_key = "_".join(args.suite)
    with result_path.open("a") as out:
        for model_name in selected_models:
            todo = [item for item in prompts if (model_name, item["id"]) not in done]
            if not todo:
                continue
            print(f"RUN_BATCH suites={','.join(args.suite)} model={model_name} n={len(todo)}", flush=True)
            for res in run_batch(model_name, MODELS[model_name], todo, suite_key):
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()

    rows = []
    with result_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in prompt_by_id:
                continue
            item = prompt_by_id[r["id"]]
            rows.append({
                **{k: r.get(k) for k in ["id", "suite", "kind", "model", "returncode", "seconds", "prefill_tps", "generation_tps", "prompt_tokens", "generated_tokens"]},
                "pass": score(item, r.get("output", "")),
                "output_chars": len(r.get("output", "")),
                "hangul_ratio": round(hangul_ratio(r.get("output", "")), 4),
                "latin_ratio": round(latin_ratio(r.get("output", "")), 4),
                "cjk_ratio": round(cjk_ratio(r.get("output", "")), 4),
            })

    with (OUT / "scores.csv").open("w", newline="") as f:
        fields = ["id", "suite", "kind", "model", "returncode", "seconds", "prefill_tps", "generation_tps", "prompt_tokens", "generated_tokens", "pass", "output_chars", "hangul_ratio", "latin_ratio", "cjk_ratio"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for suite in sorted(set(r["suite"] for r in rows)):
        summary[suite] = {}
        for model in selected_models:
            xs = [r for r in rows if r["suite"] == suite and r["model"] == model]
            summary[suite][model] = {
                "n": len(xs),
                "pass": sum(1 for r in xs if r["pass"]),
                "pass_rate": sum(1 for r in xs if r["pass"]) / len(xs) if xs else None,
                "avg_prefill_tps": sum(r["prefill_tps"] or 0 for r in xs) / len(xs) if xs else None,
                "avg_generation_tps": sum(r["generation_tps"] or 0 for r in xs) / len(xs) if xs else None,
            }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
