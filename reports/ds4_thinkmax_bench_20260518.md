# DS4 Think MAX 동작 확인 및 벤치 보고서

작성일: 2026-05-18  
작업 경로: `/Users/kch3dri4n/llm_provide/ds4`  
결과 경로: `/tmp/ds4-ko-cal/thinkmax_bench_20260518`

## 1. 결론

`--think-max`는 두 GGUF 모두에서 정상 동작했습니다.

- Base GGUF: 정상 실행
- Worst5Q4 GGUF: 정상 실행
- `ctx=393216`에서는 실제 Think MAX 경로로 실행됨
- `ctx=4096`에서는 의도대로 warning 후 normal thinking/high 경로로 downgrade됨
- Metal graph backend, batch generation, expert usage trace 모두 정상 실행됨
- 평균 decode 속도는 base와 Worst5Q4가 거의 동일함

다만 Think MAX는 현재 구현상 “Reasoning Effort: Absolute maximum...” prefix를 넣고 `<think>`로 진입하는 방식입니다. 그래서 128~512 token 예산에서는 최종 답변보다 reasoning/planning 텍스트가 길게 생성되는 경우가 많았습니다. 즉, Think MAX는 동작과 속도는 양호하지만, exact-copy나 짧은 형식 준수 작업에는 nothink보다 부적합합니다.

## 2. 확인한 모델

| 모델 | 경로 | 크기 |
|---|---:|---:|
| Base | `/Users/kch3dri4n/llm_provide/ds4/ds4flash.gguf` | 80.76 GiB |
| Worst5Q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf` | 89.20 GiB |

Worst5Q4는 L23, L25, L28, L34, L36 routed expert layer를 Q4로 올린 GGUF입니다.

## 3. Think MAX 조건

코드상 Think MAX 최소 context는 다음 값입니다.

```text
DS4_THINK_MAX_MIN_CONTEXT = 393216
```

`--think-max --ctx 393216` 이상에서는 Think MAX prefix가 들어가고, 그보다 작으면 normal thinking으로 downgrade됩니다.

컨텍스트 버퍼 추정치는 다음과 같습니다.

| ctx | backend | context buffers |
|---:|---|---:|
| 4096 | Metal | 263.46 MiB |
| 393216 | Metal | 6889.71 MiB |
| 524288 | Metal | 9121.71 MiB |

## 4. Smoke Test

### 실제 Think MAX, ctx=393216

짧은 단일 prompt로 두 모델 모두 실행했습니다.

| 모델 | context buffer | mapped model | prefill | decode |
|---|---:|---:|---:|---:|
| Base | 6889.71 MiB | 82697.67 MiB | 108.83 tok/s | 33.96 tok/s |
| Worst5Q4 | 6889.71 MiB | 91337.67 MiB | 110.17 tok/s | 33.63 tok/s |

둘 다 downgrade warning 없이 실제 Think MAX 경로로 실행됐습니다.

### 낮은 ctx에서 downgrade 확인, ctx=4096

`--think-max --ctx 4096`에서는 아래 warning이 출력되고 normal thinking으로 내려갔습니다.

```text
ds4: warning: --think-max needs --ctx >= 393216; ctx=4096 uses normal thinking instead
```

| 모델 | context buffer | prefill | decode |
|---|---:|---:|---:|
| Base | 263.46 MiB | 79.77 tok/s | 33.40 tok/s |
| Worst5Q4 | 263.46 MiB | 79.64 tok/s | 33.35 tok/s |

## 5. Think MAX 답변 예산 확인

Git 설명 prompt를 base에서 128 tokens와 256 tokens로 비교했습니다.

- 128 tokens: reasoning이 대부분을 차지해서 최종 한국어 답변까지 도달하지 못함
- 256 tokens: 최종 한국어 답변까지 도달함

따라서 Think MAX 평가는 최소 256 tokens 이상으로 잡아야 합니다. 긴 지시문이나 exact-copy류는 512~768 tokens에서도 reasoning이 길어지는 경우가 있어, 형식 준수 평가는 nothink와 분리해야 합니다.

## 6. Batch Bench, 10 Prompts, ctx=393216

벤치 구성:

- 한국어 일상 메시지 1개
- 한국어 요약 1개
- 한국어 Git 설명 1개
- 한국어 prepared statement 설명 1개
- 장문 지시문 1개
- 영어 control 1개
- 중국어 control 1개
- format control exact 1개
- 한국어 exact-copy 2개

토큰 예산:

- 일반 prompt: 256 tokens
- 장문 prompt: 384 tokens

### 전체 결과

| 모델 | n | pass | avg prefill | avg decode | avg generated |
|---|---:|---:|---:|---:|---:|
| Base | 10 | 4/10 | 137.94 tok/s | 31.47 tok/s | 261.9 |
| Worst5Q4 | 10 | 4/10 | 136.90 tok/s | 31.35 tok/s | 268.8 |

### Suite별 결과

| 모델 | suite | n | pass | avg decode |
|---|---|---:|---:|---:|
| Base | Korean | 4 | 2 | 31.48 tok/s |
| Base | Control | 3 | 2 | 31.48 tok/s |
| Base | Exact | 2 | 0 | 31.47 tok/s |
| Base | Long | 1 | 0 | 31.40 tok/s |
| Worst5Q4 | Korean | 4 | 2 | 31.35 tok/s |
| Worst5Q4 | Control | 3 | 2 | 31.36 tok/s |
| Worst5Q4 | Exact | 2 | 0 | 31.34 tok/s |
| Worst5Q4 | Long | 1 | 0 | 31.31 tok/s |

낮은 pass rate는 모델 실행 실패가 아니라 Think MAX의 출력 방식 때문입니다. 많은 실패 케이스에서 모델은 정답을 만들기 전에 지시 해석과 출력 계획을 길게 쓰다가 token budget이 끝났습니다.

## 7. Extended 2x Bench, 실패 취약 6개 재측정

256-token 벤치에서 실패하기 쉬웠던 6개 prompt를 2배 토큰 예산으로 재실행했습니다.

토큰 예산:

- 일반/정확복사/control: 512 tokens
- 장문: 768 tokens

| 모델 | n | pass | avg prefill | avg decode | avg generated |
|---|---:|---:|---:|---:|---:|
| Base | 6 | 0/6 | 141.62 tok/s | 31.35 tok/s | 554.7 |
| Worst5Q4 | 6 | 2/6 | 141.04 tok/s | 31.25 tok/s | 554.7 |

이 결과는 “Worst5Q4가 더 좋다”라기보다, Think MAX에서 token budget과 final-answer delimiter가 없으면 자동 채점이 불안정하다는 의미가 큽니다. 특히 exact-copy와 format-copy는 마지막 줄에 답을 쓰라고 해도 reasoning이 계속 이어지는 경향이 있어 Think MAX 평가 과제로 적합하지 않았습니다.

## 8. Expert Activation Check

Think MAX 형태로 렌더링한 10개 prompt에 대해 decode 64 tokens routing trace를 생성했습니다.

Trace 조건:

- ctx: 393216
- decode tokens: 64 per prompt
- prompts: 10
- prompt tokens: 1550
- decode tokens: 640
- routed expert observations: 165120 per model

확인 대상 stable experts:

```text
L40:E037
L41:E184
L38:E021
L37:E025
L42:E032
```

### Base에서의 활성화

| Expert | selected_count | count_share | weight_share |
|---|---:|---:|---:|
| L40:E037 | 454 | 11.82% | 16.49% |
| L41:E184 | 448 | 11.67% | 15.09% |
| L38:E021 | 442 | 11.51% | 18.72% |
| L37:E025 | 437 | 11.38% | 16.30% |
| L42:E032 | 334 | 8.70% | 12.88% |

### Worst5Q4에서의 활성화

| Expert | selected_count | count_share | weight_share |
|---|---:|---:|---:|
| L40:E037 | 446 | 11.61% | 18.85% |
| L41:E184 | 440 | 11.46% | 14.66% |
| L38:E021 | 438 | 11.41% | 21.01% |
| L37:E025 | 435 | 11.33% | 17.93% |
| L42:E032 | 373 | 9.71% | 18.56% |

결론: 이전에 지목한 stable experts는 Think MAX decode trace에서도 모두 강하게 활성화됐습니다. 특히 L40:E037, L41:E184, L38:E021, L37:E025는 두 모델 모두에서 global top권에 반복 등장했습니다.

## 9. 속도 판단

Think MAX 실제 실행의 decode 속도는 약 31.2~34.0 tok/s 범위였습니다.

- 짧은 smoke: 33.6~34.0 tok/s
- 10 prompt batch: 31.3~31.5 tok/s
- extended 2x batch: 31.25~31.35 tok/s

Worst5Q4는 Base 대비 decode 기준 약 0.1~0.2 tok/s 낮은 수준이라, 현재 측정에서는 실사용상 큰 속도 차이로 보기 어렵습니다. prefill도 같은 프롬프트 기준으로 거의 같은 범위입니다.

## 10. 한계와 해석 주의

- Think MAX는 reasoning을 길게 쓰므로, nothink와 같은 채점 기준을 그대로 적용하면 과소평가됩니다.
- exact-copy와 짧은 형식 준수는 Think MAX보다 nothink가 맞습니다.
- 이번 Think MAX bench는 10개 + 확장 6개로 작은 smoke/bench 성격입니다.
- 한국어 품질의 본평가는 held-out 100개 nothink 결과를 더 신뢰해야 합니다.
- Think MAX 품질을 제대로 보려면 final answer delimiter, 충분한 token budget, reasoning 제거 후 채점 파이프라인이 필요합니다.
- expert activation trace는 decode 64 token 샘플이므로, 장기 생성 전체 routing 분포를 대표한다고 단정하면 안 됩니다.

## 11. 생성 산출물

주요 결과 파일:

```text
/tmp/ds4-ko-cal/thinkmax_bench_20260518/thinkmax_summary.json
/tmp/ds4-ko-cal/thinkmax_bench_20260518/thinkmax_scores.csv
/tmp/ds4-ko-cal/thinkmax_bench_20260518/thinkmax_raw_results.jsonl
/tmp/ds4-ko-cal/thinkmax_bench_20260518/thinkmax_extended2x_summary.json
/tmp/ds4-ko-cal/thinkmax_bench_20260518/thinkmax_extended2x_scores.csv
/tmp/ds4-ko-cal/thinkmax_bench_20260518/expert_usage_thinkmax_base_decode64.csv
/tmp/ds4-ko-cal/thinkmax_bench_20260518/expert_usage_thinkmax_worst5q4_decode64.csv
/tmp/ds4-ko-cal/thinkmax_bench_20260518/expert_activation_thinkmax_decode64_summary.json
/tmp/ds4-ko-cal/thinkmax_bench_20260518/hot_experts_thinkmax_base_decode64_top5_by_layer.csv
/tmp/ds4-ko-cal/thinkmax_bench_20260518/hot_experts_thinkmax_worst5q4_decode64_top5_by_layer.csv
/Users/kch3dri4n/llm_provide/ds4/tools/bench_thinkmax_ds4.py
```

다운로드용 압축본:

```text
/Users/kch3dri4n/Downloads/ds4_thinkmax_bench_20260518.zip
```
