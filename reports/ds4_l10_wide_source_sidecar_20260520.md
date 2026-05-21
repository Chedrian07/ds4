# DS4 L10 Wide Source Sidecar Report

작성일: 2026-05-20  
작업 경로: `/Users/kch3dri4n/llm_provide/ds4`  
실험 대상: DeepSeek-V4-Flash JANGTQ-K 기반 L10 routed expert source sidecar

## 1. 결론

이번 실험의 핵심 결론은 **L10 top32보다 L10 coverage를 넓히는 방향은 맞지만, L10 full256을 기본 채팅 후보로 쓰는 것은 아직 이르다**입니다.

가장 균형 잡힌 후보는 `ThinkTop128-L10-HF-FP4`입니다. Think MAX 30개와 held-out 한국어 100개에서 가장 좋았고, KMMLU300에서도 198/300으로 이번 wide 후보 중 최고였습니다. 다만 장문 지시문 v2에서는 `ThinkTop64-L10-HF-FP4`가 41/60으로 가장 높아서, 장문 지시 추종만 놓고 보면 top64가 더 안정적입니다.

`Full256-L10-HF-FP4`는 L10 routed expert 전체를 sidecar로 덮는 가장 깨끗한 구현 검증입니다. 실제 trace에서 L10 base route가 0이고 sidecar route가 100%로 확인되었습니다. 그러나 한국어 생성형 held-out에서는 76/100으로 크게 밀렸기 때문에, full coverage가 항상 품질 개선으로 이어지지는 않았습니다.

최종 추천은 다음과 같습니다.

- 일반 한국어 chat/nothink: 기존 `base` 또는 `LateStable5Q4` 유지
- Think MAX 한국어 실험: `ThinkTop128-L10-HF-FP4`를 1순위 후보로 유지
- 장문 지시문 특화: `ThinkTop64-L10-HF-FP4` 별도 후보 유지
- 객관식/KMMLU 계열: `ThinkTop128-L10-HF-FP4`와 `Full256-L10-HF-FP4`를 둘 다 유지
- 다음 구현: L10 full256을 바로 채팅 기본값으로 올리지 말고, L10 top64/top128/full256을 runtime selectable sidecar policy로 분리

## 2. 생성한 sidecar

세 sidecar 모두 base GGUF에서 잘라낸 것이 아니라, 로컬 HF source shard에서 L10 expert tensor를 읽어 다시 sidecar GGUF로 쓴 것입니다. 단, 현재 확보한 source shard는 `HF-FP4` 경로이므로 “BF16/FP16 고정밀 원본”은 아닙니다. 이번 결과는 **base GGUF 복원 기반보다 source shard 기반에 가까운 sidecar**라는 의미로 해석해야 합니다.

source shard:

```text
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00012-of-00046.safetensors
```

생성 파일:

| 후보 | L10 expert coverage | 파일 크기 | 경로 |
|---|---:|---:|---|
| ThinkTop64 | 64/256 | 864M | `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop64-L10-HF-FP4.sidecar.gguf` |
| ThinkTop128 | 128/256 | 1.7G | `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop128-L10-HF-FP4.sidecar.gguf` |
| Full256 | 256/256 | 3.4G | `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-Full256-L10-HF-FP4.sidecar.gguf` |

repo 내 symlink:

```text
gguf/DeepSeek-V4-Flash-KR-ThinkTop64-L10-HF-FP4.sidecar.gguf
gguf/DeepSeek-V4-Flash-KR-ThinkTop128-L10-HF-FP4.sidecar.gguf
gguf/DeepSeek-V4-Flash-KR-Full256-L10-HF-FP4.sidecar.gguf
```

## 3. Expert 선정 방식

기존 uploaded trace 기반 `think_priority_top32`를 prefix로 유지했습니다. 이는 이전 측정에서 Think MAX decode 쪽 신호가 강했고, top32 후보가 이미 의미 있게 검증된 상태였기 때문입니다.

top32 이후의 확장은 이번에 로컬에서 새로 수집한 L10 usage trace로 채웠습니다.

수집한 trace:

```text
runs/20260520_l10_wide_source/usage/prefill_think_usage.csv
runs/20260520_l10_wide_source/usage/decode_think128_usage.csv
```

확장 ranking:

```text
runs/20260520_l10_wide_source/manifests/l10_local_think_usage_extended_ranking.csv
```

생성 manifest:

```text
runs/20260520_l10_wide_source/manifests/l10_think_priority_top64_source_hf_fp4.json
runs/20260520_l10_wide_source/manifests/l10_think_priority_top128_source_hf_fp4.json
runs/20260520_l10_wide_source/manifests/l10_full256_source_hf_fp4.json
```

sidecar plan:

```text
runs/20260520_l10_wide_source/sidecar_plan/l10_think_priority_top64_source_hf_fp4_sidecar_plan.json
runs/20260520_l10_wide_source/sidecar_plan/l10_think_priority_top128_source_hf_fp4_sidecar_plan.json
runs/20260520_l10_wide_source/sidecar_plan/l10_full256_source_hf_fp4_sidecar_plan.json
```

## 4. 로딩 검증

세 sidecar 모두 `./ds4 --inspect`에서 정상 로딩됐습니다.

inspect 결과 요약:

| 후보 | loaded layers | expert slots | tensor triplets | qtype |
|---|---:|---:|---:|---|
| ThinkTop64 | 1 | 64 | 3 | q4_k |
| ThinkTop128 | 1 | 128 | 3 | q4_k |
| Full256 | 1 | 256 | 3 | q4_k |

관련 로그:

```text
runs/20260520_l10_wide_source/top64_l10_hf_fp4_inspect.out
runs/20260520_l10_wide_source/top128_l10_hf_fp4_inspect.out
runs/20260520_l10_wide_source/full256_l10_hf_fp4_inspect.out
```

초기 병렬 inspect 실패는 sidecar 문제가 아니라 `ds4` 단일 실행 잠금 때문이었습니다. 순차 inspect에서는 모두 성공했습니다.

## 5. Runtime activation trace

Think MAX smoke prompt에서 `DS4_BITLIFT_TRACE_HITS=1`로 L10 route split을 확인했습니다.

| 후보 | trace events | base routes | sidecar routes | sidecar route rate | unique sidecar slots observed |
|---|---:|---:|---:|---:|---:|
| ThinkTop64 | 230 | 479 | 901 | 65.29% | 54 |
| ThinkTop128 | 230 | 290 | 1090 | 78.99% | 90 |
| Full256 | 230 | 0 | 1380 | 100.00% | 160 |

해석:

- top64는 L10 route의 약 65%를 sidecar Q4로 받았습니다.
- top128은 약 79%까지 올라갔습니다.
- full256은 L10 routed expert 전체를 sidecar로 커버하므로 base route가 0입니다.
- full256에서 base route 0 상태로 prefill/decode가 정상 동작했으므로, runtime partition path는 빈 base partition도 처리합니다.

trace summary:

```text
runs/20260520_l10_wide_source/top64_l10_hf_fp4_thinkmax_trace_summary.json
runs/20260520_l10_wide_source/top128_l10_hf_fp4_thinkmax_trace_summary.json
runs/20260520_l10_wide_source/full256_l10_hf_fp4_thinkmax_trace_summary.json
```

## 6. 속도

짧은 Think MAX smoke:

| 후보 | prefill t/s | generation t/s |
|---|---:|---:|
| ThinkTop64 | 113.62 | 30.78 |
| ThinkTop128 | 114.40 | 30.68 |
| Full256 | 117.21 | 31.07 |

Think MAX 30:

| 후보 | avg prefill t/s | avg generation t/s |
|---|---:|---:|
| ThinkTop64 | 148.36 | 31.38 |
| ThinkTop128 | 148.64 | 31.43 |
| Full256 | 149.12 | 31.37 |

KMMLU300:

| 후보 | avg prefill t/s | avg generation t/s |
|---|---:|---:|
| ThinkTop64 | 153.85 | 31.02 |
| ThinkTop128 | 153.73 | 31.01 |
| Full256 | 154.21 | 30.99 |

장문 v2:

| 후보 | avg prefill t/s | avg generation t/s |
|---|---:|---:|
| ThinkTop64 | 166.46 | 31.53 |
| ThinkTop128 | 166.21 | 31.82 |
| Full256 | 165.48 | 31.36 |

속도 결론:

세 후보 간 decode 속도 차이는 의미 있게 크지 않았습니다. sidecar 폭을 top64에서 full256까지 키워도 이번 측정에서는 generation throughput이 대략 31 tok/s 부근으로 유지되었습니다. 즉 이번 단계의 병목은 속도보다 품질/지시추종 안정성입니다.

## 7. 품질 평가 요약

기존 기준선:

| 평가 | base | Layer10Q4 | L10 source top32 | L8-L12 source top32 |
|---|---:|---:|---:|---:|
| Think MAX 30 | 10/30 | 17/30 | 9/30 | 14/30 |
| KMMLU300 | 197/300 | 210/300 | 197/300 | 182/300 |
| Korean100 | 88/100 | 84/100 | 85/100 | 83/100 |
| control60 | 60/60 | 60/60 | 60/60 | 60/60 |
| exact_long_extra | 10/30 | 10/30 | 9/30 | 8/30 |
| long v2 | 42/60 | 27/60 | 38/60 | 41/60 |

이번 L10 wide 후보:

| 평가 | ThinkTop64 | ThinkTop128 | Full256 |
|---|---:|---:|---:|
| Think MAX 30 | 13/30 | 15/30 | 11/30 |
| KMMLU100 | 69/100 | 70/100 | 74/100 |
| KMMLU300 | 194/300 | 198/300 | 197/300 |
| Korean100 | 83/100 | 87/100 | 76/100 |
| control60 | 60/60 | 60/60 | 60/60 |
| exact_long_extra | 9/30 | 9/30 | 9/30 |
| long v2 | 41/60 | 33/60 | 36/60 |

## 8. 평가별 해석

### Think MAX 30

`ThinkTop128`이 15/30으로 wide 후보 중 가장 좋았습니다. 이전 `L10 source top32`가 9/30이었으므로, L10 coverage를 top128까지 넓힌 것은 Think MAX 쪽에서 유의미한 개선입니다.

다만 기존 `Layer10Q4` 전체 레이어 변환 후보는 17/30이었습니다. 따라서 현 sidecar 방식은 top32 대비 개선됐지만, 아직 layer 전체 Q4 모델의 Think MAX 성능을 완전히 따라잡지는 못했습니다.

### KMMLU

KMMLU100에서는 full256이 74/100으로 앞섰습니다. 그러나 KMMLU300에서는 top128이 198/300, full256이 197/300으로 거의 동률입니다. 즉 full256의 KMMLU100 우세는 작은 표본에서 과대평가된 면이 있습니다.

기존 `Layer10Q4`는 KMMLU300에서 210/300이었으므로, knowledge benchmark에서는 아직 전체 L10Q4가 가장 좋습니다.

### Korean100

일반 한국어 held-out에서는 top128이 87/100으로 최고입니다. base 88/100보다는 1점 낮지만, Layer10Q4 84/100, L10 source top32 85/100보다 낫습니다.

반대로 full256은 76/100으로 크게 하락했습니다. L10의 모든 expert를 source sidecar로 교체하는 것이 한국어 생성형 안정성에는 오히려 해로울 수 있다는 신호입니다.

### Control60

세 후보 모두 60/60입니다. 이번 L10 sidecar는 영어/중국어/control prompt에서 즉각적인 퇴화는 보이지 않았습니다.

### Exact/Long Extra

세 후보 모두 9/30입니다. 기존 base와 Layer10Q4의 10/30보다 1점 낮습니다. exact-copy 계열은 여전히 취약하고, 이번 L10 wide sidecar만으로 해결되지 않았습니다.

### Long V2

장문 지시문 v2에서는 top64가 41/60으로 가장 좋습니다. 이 값은 기존 L8-L12 source top32의 41/60과 동률이고, base 42/60에는 1점 낮습니다.

top128은 33/60으로 오히려 낮습니다. L10 coverage를 top128까지 넓히면 일반 한국어/Think MAX에는 도움이 되지만, 장문 지시문 구조 보존에는 방해가 될 수 있습니다.

## 9. 최종 판단

하나만 고르라면 `ThinkTop128-L10-HF-FP4`입니다.

이유:

- top32 source sidecar보다 Think MAX가 개선됨: 9/30 → 15/30
- Korean100에서 wide 후보 중 최고: 87/100
- KMMLU300에서도 wide 후보 중 최고: 198/300
- control60 퇴화 없음: 60/60
- 속도 손해 없음: decode 약 31 tok/s 유지

단, 장문 지시문만 보면 `ThinkTop64-L10-HF-FP4`가 낫습니다. 따라서 실제 runtime policy는 다음처럼 나누는 것이 좋습니다.

```text
default chat / nothink: base 또는 LateStable5Q4
Think MAX general Korean: ThinkTop128-L10-HF-FP4
long instruction / report generation: ThinkTop64-L10-HF-FP4
KMMLU / objective QA experiment: ThinkTop128-L10-HF-FP4 또는 Full256-L10-HF-FP4
```

`Full256-L10-HF-FP4`는 버리면 안 됩니다. proof-of-runtime으로 매우 중요하고, L10 base route 0 처리가 검증됐습니다. 하지만 채팅 기본값으로 올리기에는 Korean100 하락이 큽니다.

## 10. 한계

이번 source sidecar는 `HF-FP4` shard에서 만들었습니다. 진짜 BF16/FP16 원본 expert를 가져온 것은 아닙니다. 따라서 “고정밀 원본 weight 기반”의 최종판으로 보기에는 아직 한 단계 부족합니다.

또한 sidecar는 L10만 대상으로 했습니다. L10이 현재 가장 논리적인 승부처인 것은 맞지만, L10 하나로 Layer10Q4 전체 모델의 KMMLU/Think MAX 성능을 완전히 재현하지는 못했습니다.

이번 평가는 prompt 수가 꽤 늘었지만, 여전히 내부 자동 채점 규칙에 의존합니다. exact-copy와 장문 지시문은 특히 채점 함수의 엄격도에 영향을 받습니다.

마지막으로 한국어 특이 expert와 범용 reasoning expert를 완전히 분리하지는 못했습니다. full256의 KMMLU 강세와 Korean100 약세가 갈린 것을 보면, L10 내부에도 task별로 다른 expert subset이 필요한 상태입니다.

## 11. 다음 액션

다음 단계는 `sidecar policy routing`입니다. 하나의 sidecar만 고정하는 대신, runtime에서 task mode별로 L10 sidecar coverage를 선택하게 만드는 편이 지금 데이터와 가장 잘 맞습니다.

추천 구현 순서:

1. CLI 옵션 추가: `--bitlift-policy chat|think|long|qa`
2. 정책별 sidecar 선택:
   - `chat`: sidecar off 또는 LateStable5Q4
   - `think`: L10 top128
   - `long`: L10 top64
   - `qa`: L10 top128 또는 full256
3. 동일 prompt set에서 policy switching benchmark 실행
4. BF16/FP16 원본 shard 확보 시 L10 top64/top128/full256을 같은 manifest로 재생성
5. Layer10Q4 전체 모델과 source sidecar 간 차이가 나는 expert/tensor를 diff 추적

## 12. 산출물

주요 산출물:

```text
runs/20260520_l10_wide_source/
reports/ds4_l10_wide_source_sidecar_20260520.md
```

중요 결과 파일:

```text
runs/20260520_l10_wide_source/thinkmax30_l10_wide_source/thinkmax_l10_wide_hf_fp4_summary.json
runs/20260520_l10_wide_source/kmmlu100_l10_wide_source/summary.json
runs/20260520_l10_wide_source/kmmlu300_l10_wide_source/summary.json
runs/20260520_l10_wide_source/project_eval_l10_wide_source/summary.json
runs/20260520_l10_wide_source/longv2_l10_wide_source/summary.json
```

sidecar 파일:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop64-L10-HF-FP4.sidecar.gguf
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop128-L10-HF-FP4.sidecar.gguf
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-Full256-L10-HF-FP4.sidecar.gguf
```
