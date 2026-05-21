#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/kch3dri4n/llm_provide/ds4")
DEFAULT_OUT = Path("/tmp/ds4-ko-cal/thinkmax_bench_20260518")

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


def last_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build_prompts():
    return [
        {
            "id": "tm-ko-daily-001",
            "suite": "korean",
            "kind": "daily",
            "n": 256,
            "prompt": (
                "상황: 조별과제 팀원이 마감 하루 전까지 자료를 보내지 않았습니다. "
                "한국어로 공손한 메시지를 3문장 이내로 작성하고, 마지막 문장은 오늘 밤 9시까지 자료를 보내 달라는 요청으로 끝내세요."
            ),
        },
        {
            "id": "tm-ko-summary-001",
            "suite": "korean",
            "kind": "summary",
            "n": 256,
            "prompt": (
                "다음 글을 읽고 한국어로 답하세요.\n\n"
                "[글: 데이터 편향]\n"
                "인공지능 모델은 학습 데이터의 분포를 반영한다. 특정 집단이나 언어가 데이터에서 적게 나타나면 모델의 답변 품질도 낮아질 수 있다. "
                "따라서 모델 평가에서는 전체 평균뿐 아니라 집단별 성능 차이를 따로 확인해야 한다.\n\n"
                "작업: 제목을 붙이고 핵심 내용을 정확히 3문장으로 요약하세요. 글에 없는 내용을 추가하지 마세요."
            ),
        },
        {
            "id": "tm-ko-tech-001",
            "suite": "korean",
            "kind": "tech",
            "n": 256,
            "keys": ["branch", "merge", "통합"],
            "prompt": "Git에서 branch와 merge의 차이를 초보자에게 한국어로 5문장 이내로 설명해 주세요.",
        },
        {
            "id": "tm-ko-security-001",
            "suite": "korean",
            "kind": "tech",
            "n": 256,
            "keys": ["쿼리", "입력", "분리"],
            "prompt": "SQL Injection을 방어하는 prepared statement의 원리를 공격 절차 없이 한국어로 설명해 주세요.",
        },
        {
            "id": "tm-ko-long-001",
            "suite": "long",
            "kind": "long",
            "n": 384,
            "prompt": (
                "다음 조건을 모두 만족하는 한국어 실행 계획을 작성하세요.\n"
                "상황: 한국어 모델 평가 결과를 발표 자료로 정리해야 합니다.\n"
                "조건: 첫 줄 제목은 15자 이내, 첫 문단은 2문장, 목표 bullet은 정확히 3개, "
                "각 단계는 '단계 N:'으로 시작, 위험 요소 2개와 완화 방법 2개 포함, 표 사용 금지, "
                "마지막 문장은 오늘 바로 할 수 있는 구체적 행동으로 끝내세요."
            ),
        },
        {
            "id": "tm-en-control-001",
            "suite": "control",
            "kind": "english",
            "n": 256,
            "prompt": "Answer in concise English. Explain one practical difference between caching and persistence for a software project. Use exactly 3 sentences.",
        },
        {
            "id": "tm-zh-control-001",
            "suite": "control",
            "kind": "chinese",
            "n": 256,
            "prompt": "请用简体中文回答。用三句话解释为什么备份和版本控制不是同一件事。不要使用表格。",
        },
        {
            "id": "tm-format-control-001",
            "suite": "control",
            "kind": "control_exact",
            "n": 256,
            "expected": "CONTROL-9001: ABC xyz 12345 !? 한국어 中文",
            "prompt": (
                "Think through the instruction if needed. The final non-empty line must copy the following line exactly and add no other characters on that final line:\n"
                "CONTROL-9001: ABC xyz 12345 !? 한국어 中文"
            ),
        },
        {
            "id": "tm-ko-exact-001",
            "suite": "exact",
            "kind": "exact",
            "n": 256,
            "expected": "가나다라마바사 | 숫자:9101 | 기호: !?.,;:()[]{}",
            "prompt": (
                "마지막 빈 줄이 아닌 줄에는 [복사대상] 안의 내용만 한 글자도 바꾸지 말고 그대로 출력하세요.\n"
                "[복사대상]\n가나다라마바사 | 숫자:9101 | 기호: !?.,;:()[]{}\n[/복사대상]"
            ),
        },
        {
            "id": "tm-ko-exact-002",
            "suite": "exact",
            "kind": "exact",
            "n": 256,
            "expected": "초성열: ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ / 모음열: ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ / 번호:9102",
            "prompt": (
                "마지막 빈 줄이 아닌 줄에는 다음 한글 자모 문자열만 그대로 출력하세요.\n"
                "<초성열: ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ / 모음열: ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ / 번호:9102>"
            ),
        },
    ]


def build_prompts_expanded30():
    prompts = []

    daily = [
        (
            "tm30-ko-daily-001",
            "조별과제 팀원이 마감 하루 전까지 자료를 보내지 않았습니다.",
            "오늘 밤 9시까지 자료를 보내 달라는 요청",
        ),
        (
            "tm30-ko-daily-002",
            "교수님께 출석 처리와 과제 제출 가능 여부를 정중히 물어봐야 합니다.",
            "출석 인정 가능 여부와 과제 제출 가능 기한 확인 요청",
        ),
    ]
    for pid, situation, ask in daily:
        prompts.append({
            "id": pid,
            "suite": "korean",
            "kind": "daily",
            "n": 256,
            "prompt": (
                f"상황: {situation} 한국어로 공손한 메시지를 3문장 이내로 작성하세요. "
                f"마지막 문장은 {ask}으로 끝내세요."
            ),
        })

    summaries = [
        (
            "tm30-ko-summary-001",
            "데이터 편향",
            "인공지능 모델은 학습 데이터의 분포를 반영한다. 특정 집단이나 언어가 데이터에서 적게 나타나면 모델의 답변 품질도 낮아질 수 있다. 따라서 모델 평가에서는 전체 평균뿐 아니라 집단별 성능 차이를 따로 확인해야 한다.",
        ),
        (
            "tm30-ko-summary-002",
            "긴 문맥 모델",
            "긴 문맥을 처리하는 언어 모델은 많은 문서를 한 번에 읽을 수 있다는 장점이 있다. 그러나 입력이 길어질수록 중요한 정보를 놓치거나 초반 내용을 잊는 문제가 생길 수 있다. 문맥 길이 확장은 검색, 요약, 위치 인식 능력까지 함께 평가해야 한다.",
        ),
    ]
    for pid, title, body in summaries:
        prompts.append({
            "id": pid,
            "suite": "korean",
            "kind": "summary",
            "n": 256,
            "prompt": (
                f"다음 글을 읽고 한국어로 답하세요.\n\n[글: {title}]\n{body}\n\n"
                "작업: 제목을 붙이고 핵심 내용을 정확히 3문장으로 요약하세요. 글에 없는 내용을 추가하지 마세요."
            ),
        })

    techs = [
        ("tm30-ko-tech-001", "Git에서 commit, branch, merge의 차이를 초보자에게 설명해 주세요.", ["commit", "branch", "merge"]),
        ("tm30-ko-tech-002", "REST API에서 GET과 POST의 차이를 예시와 함께 설명해 주세요.", ["GET", "POST", "조회"]),
        ("tm30-ko-tech-003", "Python 함수에 타입 힌트를 붙였을 때의 장점과 한계를 설명해 주세요.", ["타입", "힌트", "런타임"]),
        ("tm30-ko-security-001", "SQL Injection을 방어하는 prepared statement의 원리를 공격 절차 없이 설명해 주세요.", ["쿼리", "입력", "분리"]),
        ("tm30-ko-container-001", "Docker 컨테이너와 가상머신의 차이를 개발자 관점에서 설명해 주세요.", ["컨테이너", "가상머신", "커널"]),
        ("tm30-ko-llm-001", "LLM 추론에서 prefill과 decode 단계가 무엇인지 쉽게 설명해 주세요.", ["prefill", "decode", "토큰"]),
    ]
    for pid, prompt, keys in techs:
        prompts.append({
            "id": pid,
            "suite": "korean",
            "kind": "tech",
            "n": 256,
            "keys": keys,
            "prompt": f"{prompt} 한국어로 5문장 이내로 답하세요.",
        })

    long_topics = [
        ("tm30-ko-long-001", "수업 보고서", "복잡도와 순환 알고리즘 실습 결과를 보고서로 정리해야 합니다."),
        ("tm30-ko-long-002", "프로젝트 계획", "M4 Max 128GB에서 대형 MoE 모델을 실행하기 위한 양자화 실험 계획을 세워야 합니다."),
        ("tm30-ko-long-003", "동아리 발표", "신입생에게 CTF와 시스템 보안을 소개하는 5분 발표를 준비해야 합니다."),
        ("tm30-ko-long-004", "회의록 정리", "팀 회의에서 나온 의견을 실행 항목과 보류 항목으로 나누어야 합니다."),
        ("tm30-ko-long-005", "학습 계획", "자료구조, C언어, 리눅스 기초를 4주 동안 병행해서 공부해야 합니다."),
        ("tm30-ko-long-006", "기술 블로그", "한국어 LLM 양자화 실험을 독자가 이해하기 쉽게 블로그 글로 정리해야 합니다."),
        ("tm30-ko-long-007", "장학금 신청", "성적뿐 아니라 프로젝트 경험과 성장 가능성을 강조하는 자기소개 문단이 필요합니다."),
        ("tm30-ko-long-008", "모델 평가", "KMMLU와 장문 지시문 평가 결과를 기준으로 후보 모델을 비교해야 합니다."),
    ]
    for pid, topic, bg in long_topics:
        prompts.append({
            "id": pid,
            "suite": "long",
            "kind": "long",
            "n": 512,
            "prompt": (
                f"다음 조건을 모두 만족하는 한국어 답변을 작성하세요.\n\n상황: {topic}\n세부 배경: {bg}\n\n"
                "작성 조건:\n1. 첫 줄 제목은 20자 이내입니다.\n2. 첫 문단은 문제 상황을 2문장으로 요약합니다.\n"
                "3. 핵심 목표를 정확히 3개 bullet로 정리합니다.\n4. 실행 계획은 '단계 N:' 형식으로 3단계 이상 작성합니다.\n"
                "5. 위험 요소 2개와 완화 방법 2개를 포함합니다.\n6. 표는 사용하지 마세요.\n"
                "7. 마지막 문단은 오늘 바로 할 수 있는 첫 행동 1개로 끝내세요.\n8. 전체 답변은 한국어 공손체로 작성하세요."
            ),
        })

    exact_cases = [
        ("tm30-ko-exact-001", "가나다라마바사 | 숫자:9301 | 기호: !?.,;:()[]{}"),
        ("tm30-ko-exact-002", "경로: /Users/kch3drian/models/한국어 캘리브레이션/93/파일.txt"),
        ("tm30-ko-exact-003", "{\"lang\":\"ko\",\"task\":\"expert_trace\",\"문장\":\"한국어 토큰을 보존하세요\",\"id\":9303}"),
        ("tm30-ko-exact-004", "초성열: ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ / 모음열: ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ / 번호:9304"),
        ("tm30-ko-exact-005", "문장A: 나는 밥을 먹었다.\n문장B: 나는  밥을  먹었다.\n문장C: 나는\t밥을\t먹었다.\nID=9305"),
        ("tm30-ko-exact-006", "<ko><정확복사 id='9306'>서울-대전-대구-부산 / 3.14159 / 끝.</정확복사></ko>"),
    ]
    for pid, expected in exact_cases:
        prompts.append({
            "id": pid,
            "suite": "exact",
            "kind": "exact",
            "n": 256,
            "expected": expected,
            "prompt": (
                "마지막 빈 줄이 아닌 줄에는 [복사대상] 안의 내용만 한 글자도 바꾸지 말고 그대로 출력하세요.\n"
                f"[복사대상]\n{expected}\n[/복사대상]"
            ),
        })

    controls = [
        ("tm30-en-control-001", "english", "Answer in concise English. Explain why cache warming can change benchmark results. Use exactly 3 sentences.", None),
        ("tm30-en-control-002", "english", "Answer in concise English. Compare quantization and pruning in machine learning deployment. Use exactly 4 sentences.", None),
        ("tm30-zh-control-001", "chinese", "请用简体中文回答。用三句话解释为什么备份和版本控制不是同一件事。不要使用表格。", None),
        ("tm30-zh-control-002", "chinese", "请用简体中文回答。用三句话说明模型评估为什么需要保留测试集。不要使用表格。", None),
        ("tm30-format-control-001", "control_exact", "The final non-empty line must copy this exactly:\nCONTROL-9307: ABC xyz 12345 !? 한국어 中文", "CONTROL-9307: ABC xyz 12345 !? 한국어 中文"),
        ("tm30-format-control-002", "control_exact", "The final non-empty line must copy this exactly:\nCONTROL-9308: q4_K iq2_xxs 256 experts 한국어 中文", "CONTROL-9308: q4_K iq2_xxs 256 experts 한국어 中文"),
    ]
    for pid, kind, prompt, expected in controls:
        prompts.append({
            "id": pid,
            "suite": "control",
            "kind": kind,
            "n": 256,
            "expected": expected,
            "prompt": prompt,
        })

    assert len(prompts) == 30
    return prompts


def escape_tsv(text):
    return (
        str(text)
        .replace("\\", "\\\\")
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


def run_model(model_name, model_path, items, out_dir, ctx, suffix):
    stem = f"thinkmax{suffix}_{model_name}"
    batch_in = out_dir / f"{stem}.tsv"
    batch_out = out_dir / f"{stem}.jsonl"
    batch_err = out_dir / f"{stem}.stderr.log"
    batch_out.unlink(missing_ok=True)
    batch_err.unlink(missing_ok=True)
    write_batch_tsv(batch_in, items)

    cmd = [
        "./ds4",
        "--metal",
        *model_cli_args(model_path),
        "--ctx",
        str(ctx),
        "--think-max",
        "--temp",
        "0",
        "--seed",
        "7",
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
        timeout=72000,
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
        raise RuntimeError(f"Think MAX batch failed for {model_name}: {proc.returncode}\n{proc.stderr[-4000:]}")

    by_id = {item["id"]: item for item in items}
    converted = []
    for row in rows:
        item = by_id.get(row.get("id"))
        if not item:
            continue
        output = row.get("output") or ""
        converted.append({
            "id": item["id"],
            "suite": item["suite"],
            "kind": item["kind"],
            "model": model_name,
            "returncode": row.get("returncode", 1),
            "batch_wall_seconds": round(wall_seconds, 3),
            "prompt_tokens": row.get("prompt_tokens"),
            "generated_tokens": row.get("generated_tokens"),
            "prefill_seconds": row.get("prefill_seconds"),
            "decode_seconds": row.get("decode_seconds"),
            "prefill_tps": row.get("prefill_tps"),
            "generation_tps": row.get("generation_tps"),
            "output": output.strip(),
            "final_line": last_nonempty_line(output),
            "error": row.get("error"),
        })
    return converted


def score(item, result):
    output = result.get("output", "")
    final_line = result.get("final_line", "")
    if result.get("returncode") != 0 or not output.strip():
        return False
    kind = item["kind"]
    if kind in {"exact", "control_exact"}:
        expected = item.get("expected", "")
        return final_line == expected or output.strip() == expected
    if kind == "daily":
        tail = "\n".join(output.splitlines()[-4:])
        return hangul_ratio(tail) >= 0.55 and sentence_count_ko(tail) <= 5 and ("요" in tail or "습니다" in tail)
    if kind == "summary":
        return hangul_ratio(output) >= 0.45 and sentence_count_ko(output) >= 3 and "제목" in output
    if kind == "tech":
        keys = item.get("keys", [])
        return hangul_ratio(output) >= 0.25 and sum(k in output for k in keys) >= 2
    if kind == "long":
        return hangul_ratio(output) >= 0.35 and "단계 1:" in output and "위험" in output and "|" not in output
    if kind == "english":
        return latin_ratio(final_line or output) >= 0.6
    if kind == "chinese":
        return cjk_ratio(output) >= 0.25
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--ctx", type=int, default=393216)
    ap.add_argument("--models", default="base,worst5q4")
    ap.add_argument("--ids", default="")
    ap.add_argument("--token-multiplier", type=int, default=1)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--preset", choices=["standard", "expanded30"], default="standard")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts_expanded30() if args.preset == "expanded30" else build_prompts()
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        prompts = [p for p in prompts if p["id"] in wanted]
    if args.token_multiplier != 1:
        prompts = [{**p, "n": p["n"] * args.token_multiplier} for p in prompts]
    prompt_by_id = {p["id"]: p for p in prompts}
    suffix = f"_{args.suffix}" if args.suffix else ""
    (out_dir / f"thinkmax{suffix}_prompts.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=True) for p in prompts) + "\n",
        encoding="utf-8",
    )

    raw_path = out_dir / f"thinkmax{suffix}_raw_results.jsonl"
    if args.rerun:
        raw_path.unlink(missing_ok=True)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []
    done = set()
    if raw_path.exists() and not args.rerun:
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["model"], row["id"]))
                all_results.append(row)
    with raw_path.open("a", encoding="utf-8") as f:
        for model_name in selected_models:
            todo = [p for p in prompts if (model_name, p["id"]) not in done]
            if not todo:
                continue
            print(f"RUN_THINKMAX model={model_name} ctx={args.ctx} n={len(todo)}", flush=True)
            results = run_model(model_name, MODELS[model_name], todo, out_dir, args.ctx, suffix)
            for res in results:
                f.write(json.dumps(res, ensure_ascii=True) + "\n")
            f.flush()
            all_results.extend(results)

    score_rows = []
    for res in all_results:
        item = prompt_by_id[res["id"]]
        output = res.get("output", "")
        final_line = res.get("final_line", "")
        score_rows.append({
            "id": res["id"],
            "suite": res["suite"],
            "kind": res["kind"],
            "model": res["model"],
            "returncode": res["returncode"],
            "prompt_tokens": res["prompt_tokens"],
            "generated_tokens": res["generated_tokens"],
            "prefill_tps": res["prefill_tps"],
            "generation_tps": res["generation_tps"],
            "pass": score(item, res),
            "output_chars": len(output),
            "final_line_chars": len(final_line),
            "hangul_ratio": round(hangul_ratio(output), 4),
            "final_hangul_ratio": round(hangul_ratio(final_line), 4),
            "latin_ratio": round(latin_ratio(output), 4),
            "cjk_ratio": round(cjk_ratio(output), 4),
        })

    with (out_dir / f"thinkmax{suffix}_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "id", "suite", "kind", "model", "returncode", "prompt_tokens", "generated_tokens",
            "prefill_tps", "generation_tps", "pass", "output_chars", "final_line_chars",
            "hangul_ratio", "final_hangul_ratio", "latin_ratio", "cjk_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(score_rows)

    summary = {}
    for model_name in selected_models:
        xs = [r for r in score_rows if r["model"] == model_name]
        summary[model_name] = {
            "n": len(xs),
            "pass": sum(1 for r in xs if r["pass"]),
            "pass_rate": sum(1 for r in xs if r["pass"]) / len(xs) if xs else None,
            "avg_prefill_tps": sum((r["prefill_tps"] or 0) for r in xs) / len(xs) if xs else None,
            "avg_generation_tps": sum((r["generation_tps"] or 0) for r in xs) / len(xs) if xs else None,
            "avg_generated_tokens": sum((r["generated_tokens"] or 0) for r in xs) / len(xs) if xs else None,
        }
        for suite in sorted(set(r["suite"] for r in xs)):
            ss = [r for r in xs if r["suite"] == suite]
            summary[model_name][f"suite_{suite}"] = {
                "n": len(ss),
                "pass": sum(1 for r in ss if r["pass"]),
                "avg_generation_tps": sum((r["generation_tps"] or 0) for r in ss) / len(ss) if ss else None,
            }

    (out_dir / f"thinkmax{suffix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
