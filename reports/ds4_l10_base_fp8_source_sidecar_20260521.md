# DS4 L10 Base-FP8 Source Sidecar 평가 보고서

작성일: 2026-05-21  
작업 위치: `/Users/kch3dri4n/llm_provide/ds4`  
실험 산출물 위치: `/Volumes/Back_UP_LLM/ds4-source-fp8-sidecar`

## 결론

이번 실험의 핵심 결론은 명확합니다. `deepseek-ai/DeepSeek-V4-Flash-Base`의 official Base FP8 shard에서 Layer 10 routed expert를 직접 읽어 Q4_K sidecar로 만드는 파이프라인은 성공했습니다. 세 sidecar 모두 로드, inspect, Think MAX routing trace, structured eval, KMMLU300, Think MAX 30, long instruction v2까지 런타임 안정성은 통과했습니다.

하지만 품질 기준으로는 새 Base-FP8 L10 sidecar가 최종 추천 모델이 아닙니다. 일반 chat/nothink는 `base` 또는 기존 `LateStable5Q4` 유지가 맞고, Think MAX 및 KMMLU 계열은 기존 `Layer10Q4`가 여전히 가장 강합니다. 새 Base-FP8 L10 sidecar는 실사용 추천 모델이 아니라, source-based sidecar writer/runtime 파이프라인을 검증한 실험 산출물로 보관하는 것이 맞습니다.

## 생성한 sidecar

| artifact | coverage | size | SHA256 |
|---|---:|---:|---|
| `DeepSeek-V4-Flash-KR-ThinkTop64-L10-BaseFP8-Q4.sidecar.gguf` | L10 top64 routed experts | 864M | `fb3c755c658e39287424dfe85cc9bfe0fbcc8a4bfbaf775ed0263e4640de2f0e` |
| `DeepSeek-V4-Flash-KR-ThinkTop128-L10-BaseFP8-Q4.sidecar.gguf` | L10 top128 routed experts | 1.7G | `74bc248b0ff480f4a56066a73693b15e8dcfdb8e0d5608119ada730f58db888b` |
| `DeepSeek-V4-Flash-KR-Full256-L10-BaseFP8-Q4.sidecar.gguf` | L10 all 256 routed experts | 3.4G | `b22a14ab2bedf72ef31968168186f8c937c229c3e2bd38b9fb55346b903ee94a` |

원본 source shard는 `model-00012-of-00046.safetensors`이며, Layer 10 routed expert tensor는 `F8_E4M3` weight와 `F32` block scale 조합입니다. 따라서 이번 실험은 full BF16 원본 기반이 아니라 official Base FP8 source 기반입니다.

## 로드 및 routing trace

`./ds4 -m ds4flash.gguf --bitlift-sidecar ... --inspect` 기준 세 artifact 모두 정상 로드되었습니다.

| artifact | sidecar mapped memory | loaded layers | expert slots |
|---|---:|---:|---:|
| Top64 | 864.02 MiB | 1 | 64 |
| Top128 | 1728.02 MiB | 1 | 128 |
| Full256 | 3456.02 MiB | 1 | 256 |

Think MAX routing trace도 정상 동작했습니다.

| artifact | route events | base routes | sidecar routes | sidecar route rate | unique sidecar slots | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top64 | 266 | 553 | 1043 | 0.654 | 55 | 114.06 | 30.91 |
| Top128 | 265 | 411 | 1179 | 0.742 | 91 | 115.13 | 30.99 |
| Full256 | 266 | 0 | 1596 | 1.000 | 166 | 118.87 | 31.10 |

해석: 전문가 선택 경로는 정상적으로 sidecar를 타고 있습니다. 특히 Full256은 L10 routed expert가 전부 sidecar에 존재하므로 route hit rate가 100%였습니다. 즉 문제는 dispatch/로드가 아니라, 어떤 source와 어느 layer/expert를 올렸을 때 실제 품질이 좋아지는지의 문제입니다.

## Structured 평가

평가 파일: `runs/20260521_l10_base_fp8_source/eval_structured/summary.json`

| suite | model | pass | prefill t/s | decode t/s |
|---|---|---:|---:|---:|
| korean100 | base | 88/100 | 123.87 | 32.11 |
| korean100 | Top64 BaseFP8 sidecar | 85/100 | 122.93 | 31.33 |
| korean100 | Top128 BaseFP8 sidecar | 85/100 | 123.24 | 31.37 |
| korean100 | Full256 BaseFP8 sidecar | 80/100 | 123.53 | 31.32 |
| control60 | base | 60/60 | 59.43 | 32.59 |
| control60 | Top64 BaseFP8 sidecar | 60/60 | 58.98 | 31.69 |
| control60 | Top128 BaseFP8 sidecar | 60/60 | 59.35 | 31.70 |
| control60 | Full256 BaseFP8 sidecar | 60/60 | 59.42 | 31.71 |
| exact_long_extra | base | 10/30 | 137.76 | 31.81 |
| exact_long_extra | Top64 BaseFP8 sidecar | 10/30 | 137.06 | 31.16 |
| exact_long_extra | Top128 BaseFP8 sidecar | 10/30 | 137.64 | 31.18 |
| exact_long_extra | Full256 BaseFP8 sidecar | 10/30 | 137.55 | 31.16 |

해석: control 퇴화와 exact-copy 추가 악화는 관찰되지 않았습니다. 다만 korean100에서 base 대비 Top64/Top128은 -3, Full256은 -8로 떨어졌습니다. 일반 한국어 chat/nothink 개선 후보로는 탈락입니다.

## KMMLU 300

평가 파일: `runs/20260521_l10_base_fp8_source/eval_kmmlu300/summary.json`

| model | correct | accuracy | invalid | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 209/300 | 0.697 | 0 | 155.38 | 30.96 |
| Layer10Q4 | 212/300 | 0.707 | 1 | 152.42 | 30.60 |
| Top64 BaseFP8 sidecar | 205/300 | 0.683 | 1 | 154.50 | 30.75 |
| Top128 BaseFP8 sidecar | 205/300 | 0.683 | 0 | 154.60 | 30.77 |
| Full256 BaseFP8 sidecar | 205/300 | 0.683 | 0 | 154.77 | 30.71 |

해석: KMMLU에서는 기존 `Layer10Q4`가 최상입니다. 새 Base-FP8 sidecar 3종은 모두 base보다 낮습니다. 원본 source에서 sidecar를 만들었다는 점만으로 품질이 좋아지지는 않았고, L10 단일 layer coverage 확대는 이 평가에서 효과가 없었습니다.

## Think MAX 30

평가 파일: `runs/20260521_l10_base_fp8_source/eval_thinkmax30/thinkmax_basefp8_cmp_summary.json`

| model | pass | korean subset | long subset | control subset | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 10/30 | 3/10 | 3/8 | 4/6 | 148.16 | 31.55 |
| Layer10Q4 | 17/30 | 8/10 | 5/8 | 4/6 | 145.62 | 31.12 |
| Top64 BaseFP8 sidecar | 14/30 | 5/10 | 5/8 | 4/6 | 147.37 | 31.26 |
| Top128 BaseFP8 sidecar | 12/30 | 2/10 | 6/8 | 4/6 | 147.50 | 31.25 |
| Full256 BaseFP8 sidecar | 10/30 | 0/10 | 6/8 | 4/6 | 148.56 | 31.19 |

해석: Think MAX에서는 `Layer10Q4`가 확실히 우세합니다. Top128/Full256은 long subset은 괜찮지만 korean subset이 크게 무너졌습니다. 특히 Full256은 korean subset 0/10이라 실사용 후보로 두기 어렵습니다.

## Long Instruction v2

평가 파일: `runs/20260521_l10_base_fp8_source/eval_longv2/summary.json`

| model | pass | avg score | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|
| base | 41/60 | 0.850 | 165.72 | 31.44 |
| Layer10Q4 | 27/60 | 0.820 | 163.98 | 31.08 |
| Top64 BaseFP8 sidecar | 34/60 | 0.819 | 165.87 | 31.15 |
| Top128 BaseFP8 sidecar | 30/60 | 0.806 | 164.71 | 31.10 |
| Full256 BaseFP8 sidecar | 36/60 | 0.821 | 164.42 | 31.23 |

해석: 장문 지시문은 base가 가장 좋습니다. Layer10Q4는 Think MAX/KMMLU는 좋지만 장문 형식 준수에서 손해가 큽니다. Full256 Base-FP8 sidecar는 새 sidecar 중 장문에서는 가장 낫지만 base에는 못 미칩니다.

## 속도와 메모리

세 Base-FP8 sidecar 모두 decode 속도는 대략 31 tok/s 선에서 안정적이었습니다. base 대비 큰 속도 붕괴는 없었습니다. sidecar 추가 메모리는 Top64 약 864 MiB, Top128 약 1.7 GiB, Full256 약 3.4 GiB입니다. 실제 총 mmap 기준은 base GGUF 약 80.76 GiB에 sidecar mapped memory가 추가되는 구조입니다.

## 최종 운영 추천

1. 일반 chat/nothink: `base` 또는 기존 `LateStable5Q4` 유지.
2. Think MAX 한국어/KMMLU 실험: `Layer10Q4` 유지.
3. 새 Base-FP8 L10 Top64/Top128/Full256 sidecar: 실사용 추천 모델로 채택하지 않음.
4. sidecar runtime/writer: 유지할 가치 있음. 로드, inspect, routing hit, Q4_K tensor dispatch 기반은 검증됨.
5. 다음 실험: L10 단일 layer coverage 확대보다, Layer10Q4가 실제로 올린 multi-layer 구성의 원본 source 기반 재현이 더 논리적입니다. 즉 `L23/L25/L28/L34/L36/L37/L38/L40/L41/L42` 중심으로 source-based sidecar를 만드는 방향이 다음 후보입니다.

## PR에 포함할 내용

- FP8 E4M3 + F32 block scale을 Q4_K로 변환하는 quant helper.
- HF Base-FP8 safetensors shard에서 routed expert tensor를 읽어 sidecar GGUF를 만드는 writer.
- 기존 eval/bench 스크립트에 Base-FP8 sidecar alias 추가.
- 본 보고서와 평가 결과 요약.

## HF 업로드 정책

이번 산출물은 private HF repo에 업로드하되, 모델 카드에서 “최종 추천 모델”이 아니라 “source-based sidecar 파이프라인 검증 artifact”로 명시해야 합니다. 품질상 winner는 새 sidecar가 아니며, 현 시점 winner는 용도별로 base/LateStable5Q4/Layer10Q4입니다.
