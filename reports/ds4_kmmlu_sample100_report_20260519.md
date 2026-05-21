# DS4 KMMLU Sample 100 Evaluation

날짜: 2026-05-19  
작업 디렉터리: `/Users/kch3dri4n/llm_provide/ds4`  
결과 디렉터리: `/tmp/ds4-ko-cal/kmmlu_sample100_20260519`

## 목적

KMMLU 원본 test split에서 seed 고정 샘플 100문항을 뽑아, 현재 로컬 GGUF 후보들이 한국어 지식형 객관식 문제에서 퇴화하는지 확인했습니다.

이번 평가는 전체 KMMLU 벤치마크가 아니라 100문항 smoke/regression 평가입니다. 따라서 leaderboard 점수로 해석하면 안 되고, base 대비 GGUF 변형 간 상대 변화만 보는 용도입니다.

## 데이터와 샘플링

- 데이터셋: `HAERAE-HUB/KMMLU`
- 사용 split: 각 subject의 `*-test.csv`
- 샘플 수: 100
- 샘플 seed: `20260519`
- 샘플 파일: `/tmp/ds4-ko-cal/kmmlu_sample100_20260519/kmmlu_sample100.jsonl`
- 총 포함 subject 수: 35

정답 분포는 다음과 같습니다.

| 정답 | 개수 |
|---:|---:|
| 1 | 14 |
| 2 | 17 |
| 3 | 34 |
| 4 | 35 |

샘플 수가 많은 subject는 `Industrial-Engineer` 8개, `Law` 6개, `Refrigerating-Machinery` 6개, `Nondestructive-Testing` 5개입니다. 나머지 subject는 대부분 1~4개입니다.

## 평가 설정

모든 모델은 같은 프롬프트와 같은 실행 조건으로 평가했습니다.

```text
backend: --metal
mode: --nothink
ctx: 4096
temperature: 0
seed: 1
max generated tokens per question: 8
```

프롬프트는 “정답 번호만 출력”하도록 제한했습니다. 채점은 모델 출력에서 첫 번째 `1~4` 또는 `A~D`를 추출해 KMMLU의 정답 번호와 비교했습니다.

## 비교 모델

| 이름 | GGUF |
|---|---|
| base | `/Users/kch3dri4n/llm_provide/ds4/ds4flash.gguf` |
| worst5q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf` |
| latestable5q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf` |

## 결과 요약

| 모델 | 정답 / 100 | 정확도 | invalid | 평균 prefill t/s | 평균 decode t/s | 평균 prompt tokens | 평균 생성 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 69 / 100 | 0.690 | 0 | 152.91 | 31.13 | 178.1 | 1.0 |
| worst5q4 | 71 / 100 | 0.710 | 0 | 149.86 | 30.84 | 178.1 | 1.0 |
| latestable5q4 | 69 / 100 | 0.690 | 0 | 151.30 | 31.00 | 178.1 | 1.0 |

속도는 세 모델이 사실상 같은 범위입니다. Worst5Q4와 LateStable5Q4는 base 대비 prefill/decode 모두 큰 손실이 없었습니다.

## Base 대비 변화

| 모델 | base가 틀리고 해당 모델이 맞힌 문항 | base가 맞고 해당 모델이 틀린 문항 | 순증감 |
|---|---:|---:|---:|
| worst5q4 | 7 | 5 | +2 |
| latestable5q4 | 2 | 2 | 0 |

세 모델이 모두 맞힌 문항은 63개, 세 모델이 모두 틀린 문항은 22개였습니다.

## 해석

이번 KMMLU 100문항 smoke sample에서는 `worst5q4`가 base보다 2문항 높게 나왔고, `latestable5q4`는 base와 동률이었습니다. 다만 100문항이고 subject별 표본 수가 작기 때문에, Worst5Q4가 실제로 KMMLU에서 우수하다고 단정할 수는 없습니다.

중요한 점은 이전 한국어 장문 지시문 평가에서 Worst5Q4가 크게 약했던 것과 달리, KMMLU식 짧은 객관식 지식 문제에서는 퇴화 신호가 보이지 않았다는 점입니다. 즉 Worst5Q4의 문제는 “한국어 전반 지식 능력”보다는 장문 지시 이행, 형식 유지, 생성 안정성 쪽에 더 가까워 보입니다.

LateStable5Q4는 KMMLU sample에서 base와 동률이고 속도 손실도 작았습니다. 하지만 이전 Think MAX와 장문 지시 평가까지 합치면 아직 최종 후보로 승격하기에는 근거가 부족합니다.

## 산출물

```text
/tmp/ds4-ko-cal/kmmlu_sample100_20260519/kmmlu_sample100.jsonl
/tmp/ds4-ko-cal/kmmlu_sample100_20260519/raw_results.jsonl
/tmp/ds4-ko-cal/kmmlu_sample100_20260519/scores.csv
/tmp/ds4-ko-cal/kmmlu_sample100_20260519/summary.json
/tmp/ds4-ko-cal/kmmlu_sample100_20260519/REPORT.md
/Users/kch3dri4n/llm_provide/ds4/tools/eval_kmmlu_sample.py
```

## 다음 권장 작업

1. 같은 스크립트로 seed를 3개 이상 바꿔 `3 x 100` 또는 `5 x 100` 반복 측정합니다.
2. subject별 stratified sample을 만들어 특정 분야 과표집 영향을 줄입니다.
3. `KR-Layer10Q4`를 만들 경우, 같은 KMMLU sample으로 base/Worst5Q4/LateStable5Q4/Layer10Q4를 동시에 비교합니다.
4. KMMLU에서 좋아져도 장문 지시문이 무너지면 채택하지 않는 기준을 유지합니다.
