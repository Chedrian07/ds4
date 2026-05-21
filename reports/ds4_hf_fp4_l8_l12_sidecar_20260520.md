# DS4 한국어 Bit-Lift Source Sidecar 실험 보고서

작성일: 2026-05-20  
작업 디렉터리: `/Users/kch3dri4n/llm_provide/ds4`  
대상 모델: DeepSeek-V4-Flash JANGTQ-K / DS4 GGUF runtime  
이번 실험명: `ThinkTop32 L10 / L8-L12 HF-FP4 source sidecar`

## 1. 결론

이번 실험의 핵심 결론은 명확합니다.

**원본 shard 기반 sidecar 파이프라인은 성공했습니다.**  
L10 단독 sidecar와 L8-L12 sidecar 모두 GGUF로 생성했고, runtime에서 sidecar expert route가 실제로 활성화되며 계산 경로에 들어가는 것을 trace로 확인했습니다.

하지만 **품질 기준으로는 아직 실전 후보가 아닙니다.**  
현재까지의 최강 후보는 여전히 `Layer10Q4` 전체 GGUF입니다. 일반 chat/nothink에서는 base가 가장 안정적이고, Think MAX 및 KMMLU에서는 `Layer10Q4`가 가장 강합니다.

이번 source sidecar 중에서는 `L8-L12 HF-FP4`가 가장 나았습니다. 다만 KMMLU 300에서 base보다도 낮아졌고, Think MAX 30에서도 `Layer10Q4`를 넘지 못했습니다. 따라서 `L8-L12 HF-FP4`는 실전 배포 후보라기보다 “source-based sidecar가 동작하며 일부 장문 형식 준수는 보존된다”는 검증 결과로 보는 것이 맞습니다.

최종 운영 판단은 다음과 같습니다.

- 일반 chat / nothink: `base` 유지
- Think MAX 한국어 실험: `Layer10Q4` 유지
- source sidecar 후보: `L8-L12 HF-FP4`는 보관, 실전 주력 아님
- L10 단독 source sidecar: 작동 검증용, 품질 후보에서는 제외
- 다음 승부: top32만 올리는 방식이 아니라 `Layer10 전체 expert` 또는 더 넓은 expert coverage를 source-sidecar로 검증

## 2. 생성된 산출물

### 2.1 L10 source sidecar

파일:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-L10-HF-FP4.sidecar.gguf
```

요약:

```text
크기: 432M
layer_count: 1
layers: L10
expert_slot_count: 32
tensor_count: 4
source shard: model-00012-of-00046.safetensors
source format: packed_fp4_i8_plus_f8_e8m0_scales
qtype: Q4_K sidecar payload
sha256: 2032fb69e0f297290fcaa30c1b3dda8f96bfeb509a1a0c31f6005eac160600cf
```

빌드 summary:

```text
runs/20260520_hf_fp4_l8_l12/thinktop32_l10_hf_fp4.summary.json
```

### 2.2 L8-L12 source sidecar

파일:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-L8-L12-HF-FP4.sidecar.gguf
```

요약:

```text
크기: 2.1G
layer_count: 5
layers: L8, L9, L10, L11, L12
expert_slot_count: 160
tensor_count: 20
source shards: model-00010-of-00046.safetensors ... model-00014-of-00046.safetensors
source format: packed_fp4_i8_plus_f8_e8m0_scales
qtype: Q4_K sidecar payload
sha256: ed87c85c9fd655c1f581b4f2a14f4b74e633f668adbdba6c8121551fe023f38c
```

빌드 summary:

```text
runs/20260520_hf_fp4_l8_l12/thinktop32_l8_l12_hf_fp4.summary.json
```

### 2.3 사용한 source shards

다운로드 위치:

```text
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5
```

Layer mapping:

```text
L8  -> model-00010-of-00046.safetensors
L9  -> model-00011-of-00046.safetensors
L10 -> model-00012-of-00046.safetensors
L11 -> model-00013-of-00046.safetensors
L12 -> model-00014-of-00046.safetensors
```

주의: 여기서 “source-based”는 기존 GGUF에서 다시 뽑은 값이 아니라 Hugging Face source shard에서 직접 sidecar를 만들었다는 뜻입니다. 다만 해당 shard 자체가 full FP16 원본이 아니라 `packed_fp4_i8_plus_f8_e8m0_scales` 형식입니다. 즉 “고정밀 원본 weight”라기보다는 “상위 source artifact 기반”입니다.

## 3. Runtime activation 검증

### 3.1 L10 source sidecar

Trace 파일:

```text
runs/20260520_hf_fp4_l8_l12/l10_hf_fp4_thinkmax_trace_summary.json
```

Think MAX smoke prompt에서 L10 sidecar가 실제 routing 계산에 들어갔습니다.

```text
trace rows: 219
L10 base routes: 519
L10 sidecar routes: 795
L10 sidecar rate: 60.50%
```

상위 sidecar slots:

```text
L10 slot 1: 198 hits
L10 slot 0: 111 hits
L10 slot 5: 44 hits
L10 slot 2: 42 hits
L10 slot 6: 41 hits
```

판정: sidecar가 장식으로 붙은 것이 아니라 실제 dispatch에서 선택됩니다.

### 3.2 L8-L12 source sidecar

Trace 파일:

```text
runs/20260520_hf_fp4_l8_l12/l8_l12_hf_fp4_thinkmax_trace_summary.json
```

Think MAX smoke prompt에서 다섯 layer 모두 sidecar route가 활성화되었습니다.

```text
L8  sidecar rate: 66.21%
L9  sidecar rate: 50.00%
L10 sidecar rate: 55.10%
L11 sidecar rate: 51.75%
L12 sidecar rate: 65.98%
```

상위 sidecar slots:

```text
L10 slot 1: 199 hits
L8  slot 0: 157 hits
L12 slot 1: 150 hits
L8  slot 3: 132 hits
L11 slot 0: 122 hits
```

판정: L8-L12 sidecar도 runtime에서 안정적으로 활성화됩니다. 다만 활성화율이 높다는 사실이 품질 개선을 보장하지는 않았습니다.

## 4. Think MAX 30 평가

평가 경로:

```text
runs/20260520_hf_fp4_l8_l12/thinkmax30_base_layer10_l10src_l8l12src
```

요약:

| 모델 | pass | pass rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 10/30 | 33.33% | 148.99 | 31.65 |
| Layer10Q4 | 17/30 | 56.67% | 148.33 | 31.19 |
| L10 HF-FP4 sidecar | 9/30 | 30.00% | 148.88 | 31.33 |
| L8-L12 HF-FP4 sidecar | 14/30 | 46.67% | 146.46 | 30.69 |

세부:

```text
base Korean suite: 3/10
Layer10Q4 Korean suite: 8/10
L10 HF-FP4 Korean suite: 2/10
L8-L12 HF-FP4 Korean suite: 5/10
```

해석:

- `Layer10Q4`가 Think MAX에서는 확실한 1등입니다.
- `L8-L12 HF-FP4`는 base보다는 낫지만 `Layer10Q4`와 격차가 큽니다.
- `L10 HF-FP4`는 활성화율 60%에도 불구하고 품질이 base보다 낮았습니다.
- 속도는 대체로 양호합니다. L8-L12 source sidecar도 decode 기준 약 3% 이내 하락입니다.

## 5. KMMLU 100 평가

평가 경로:

```text
runs/20260520_hf_fp4_l8_l12/kmmlu100_base_layer10_l10src_l8l12src
```

요약:

| 모델 | correct | accuracy | invalid | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 70/100 | 70.00% | 0 | 153.12 | 31.23 |
| Layer10Q4 | 79/100 | 79.00% | 0 | 149.48 | 31.13 |
| L10 HF-FP4 sidecar | 67/100 | 67.00% | 0 | 151.55 | 31.19 |
| L8-L12 HF-FP4 sidecar | 66/100 | 66.00% | 1 | 148.72 | 30.61 |

해석:

- `Layer10Q4`가 100개 샘플에서 크게 우세합니다.
- source sidecar 둘은 base보다 낮습니다.
- L8-L12는 Think MAX에서는 base를 넘었지만 KMMLU에서는 base보다 낮아졌습니다.

## 6. KMMLU 300 평가

평가 경로:

```text
runs/20260520_hf_fp4_l8_l12/kmmlu300_base_layer10_l10src_l8l12src
```

요약:

| 모델 | correct | accuracy | invalid | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 197/300 | 65.67% | 0 | 156.28 | 30.90 |
| Layer10Q4 | 210/300 | 70.00% | 1 | 153.42 | 30.63 |
| L10 HF-FP4 sidecar | 197/300 | 65.67% | 2 | 155.54 | 30.67 |
| L8-L12 HF-FP4 sidecar | 182/300 | 60.67% | 2 | 152.62 | 30.05 |

해석:

- 표본을 300개로 늘려도 `Layer10Q4` 우세가 유지됩니다.
- `L10 HF-FP4 sidecar`는 base와 같은 correct 수지만 invalid가 2개 있어 실제 안정성은 base보다 낮습니다.
- `L8-L12 HF-FP4 sidecar`는 KMMLU에서 명확히 퇴화했습니다.
- 이 결과 때문에 L8-L12 source sidecar를 “한국어 지식형 성능 개선 후보”로 채택하면 안 됩니다.

## 7. 한국어 held-out / control / exact-long 평가

평가 경로:

```text
runs/20260520_hf_fp4_l8_l12/project_eval_base_layer10_l10src_l8l12src
```

### 7.1 한국어 held-out 100

| 모델 | pass | pass rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 88/100 | 88.00% | 122.99 | 31.88 |
| Layer10Q4 | 84/100 | 84.00% | 121.24 | 31.66 |
| L10 HF-FP4 sidecar | 85/100 | 85.00% | 122.79 | 31.81 |
| L8-L12 HF-FP4 sidecar | 83/100 | 83.00% | 120.18 | 31.04 |

해석:

- 일반 nothink 한국어 작성/요약/기술/복사/계획에서는 base가 가장 안정적입니다.
- L10 source sidecar는 Layer10Q4보다 1점 높지만 base보다 낮습니다.
- L8-L12 source sidecar는 네 모델 중 가장 낮습니다.

### 7.2 영어/중국어/control 60

| 모델 | pass | pass rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 60/60 | 100.00% | 59.21 | 32.28 |
| Layer10Q4 | 60/60 | 100.00% | 58.37 | 31.94 |
| L10 HF-FP4 sidecar | 60/60 | 100.00% | 59.07 | 32.07 |
| L8-L12 HF-FP4 sidecar | 60/60 | 100.00% | 57.71 | 31.13 |

해석:

- 영어/중국어/control 퇴화는 이 샘플에서는 관측되지 않았습니다.
- L8-L12 source sidecar의 decode 속도는 base 대비 약 3.6% 낮습니다.

### 7.3 exact-copy + extra long 30

| 모델 | pass | pass rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 10/30 | 33.33% | 137.33 | 31.84 |
| Layer10Q4 | 10/30 | 33.33% | 133.52 | 31.41 |
| L10 HF-FP4 sidecar | 9/30 | 30.00% | 136.25 | 31.54 |
| L8-L12 HF-FP4 sidecar | 8/30 | 26.67% | 134.35 | 30.67 |

해석:

- exact-copy는 전체적으로 약점입니다.
- source sidecar가 exact-copy를 개선하지 못했고 오히려 낮췄습니다.
- 한글 자모/공백/태그 복사 계열은 별도 보정 대상입니다.

## 8. 장문 지시문 v2 평가

평가 경로:

```text
runs/20260520_hf_fp4_l8_l12/longv2_base_layer10_l10src_l8l12src
```

요약:

| 모델 | pass | pass rate | avg score | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 42/60 | 70.00% | 0.8556 | 168.46 | 31.40 |
| Layer10Q4 | 27/60 | 45.00% | 0.8196 | 166.05 | 30.95 |
| L10 HF-FP4 sidecar | 38/60 | 63.33% | 0.8455 | 167.67 | 31.11 |
| L8-L12 HF-FP4 sidecar | 41/60 | 68.33% | 0.8426 | 164.38 | 30.45 |

해석:

- 장문 형식 준수에서는 base가 가장 강합니다.
- `L8-L12 HF-FP4`는 41/60으로 base와 1점 차이까지 접근했습니다.
- `Layer10Q4`는 Think MAX/KMMLU에서는 강하지만 장문 형식 준수에서는 크게 낮아집니다.
- 따라서 Layer10Q4는 “thinking/high + 지식형 한국어” 후보이지, 모든 한국어 작업의 전역 기본값은 아닙니다.

## 9. 속도 평가

전반적인 속도는 실사용 가능한 범위입니다.

### Think MAX 30 기준

```text
base decode: 31.65 t/s
Layer10Q4 decode: 31.19 t/s
L10 source sidecar decode: 31.33 t/s
L8-L12 source sidecar decode: 30.69 t/s
```

### KMMLU 300 기준

```text
base decode: 30.90 t/s
Layer10Q4 decode: 30.63 t/s
L10 source sidecar decode: 30.67 t/s
L8-L12 source sidecar decode: 30.05 t/s
```

### 장문 v2 기준

```text
base decode: 31.40 t/s
Layer10Q4 decode: 30.95 t/s
L10 source sidecar decode: 31.11 t/s
L8-L12 source sidecar decode: 30.45 t/s
```

판정:

- sidecar runtime overhead는 크지 않습니다.
- L8-L12 sidecar도 대략 2-4% 수준의 decode 하락에 머뭅니다.
- 병목은 속도가 아니라 품질입니다.

## 10. 왜 활성화되는데 품질은 안 오르는가

이번 실험에서 가장 중요한 학습은 이것입니다.

```text
high route hit rate != quality gain
```

L10 source sidecar는 Think MAX smoke에서 sidecar route rate가 60.50%였습니다. L8-L12 source sidecar도 L8/L12에서 66% 수준의 sidecar route rate를 보였습니다. 그런데 품질은 `Layer10Q4`를 넘지 못했습니다.

가능한 원인은 다음과 같습니다.

1. Top32 expert만으로는 Layer10Q4 효과를 재현하기 부족합니다.
2. Layer10Q4의 개선은 특정 hot expert 일부가 아니라 layer 전체 expert 분포의 정밀도 변화에서 나온 것일 수 있습니다.
3. HF source shard가 full FP16이 아니라 packed FP4이므로 “source 기반”이라도 정보량이 제한됩니다.
4. sidecar Q4_K 변환이 기존 JANGTQ-K 본체의 양자화/스케일링 특성과 정확히 맞지 않을 수 있습니다.
5. Think MAX에서 일부 smoke output이 영어 meta reasoning으로 시작했습니다. 이는 routing 변경이 출력 스타일 안정성에도 영향을 줄 수 있음을 시사합니다.
6. KMMLU처럼 1-token multiple choice 답변에서는 작은 logit shift가 정답률을 크게 흔들 수 있습니다.

## 11. 이번 실험의 한계

### 11.1 source shard는 full precision 원본이 아닙니다

사용한 HF shard는 `packed_fp4_i8_plus_f8_e8m0_scales`입니다. 기존 GGUF에서 재추출한 것보다는 낫지만, full FP16/BF16 원본 weight에서 Q4로 올린 실험은 아닙니다.

### 11.2 expert coverage가 좁습니다

이번 source sidecar는 ThinkTop32 후보만 올렸습니다.

```text
L10: 32 experts
L8-L12: 5 layers * 32 experts = 160 experts
```

Layer10Q4 전체 모델은 layer 전체 routed expert 정밀도를 바꿉니다. 따라서 효과가 특정 top32에만 있는지, 전체 layer의 분포 안정화에 있는지 아직 분해되지 않았습니다.

### 11.3 평가 기준은 자동 휴리스틱입니다

한국어 held-out, exact-copy, long-v2는 자동 scoring rule입니다. 사람 평가와 완전히 같지는 않습니다. 다만 같은 프롬프트, 같은 러너, 같은 seed에서 모델 간 상대 비교를 하는 데는 충분히 유용합니다.

### 11.4 Think MAX output 스타일 문제

source sidecar smoke에서 한국어 지시에도 영어 meta 문장이 출력 초반에 나타난 사례가 있었습니다. 이는 고정밀 후보가 “정답률”뿐 아니라 응답 스타일에도 영향을 준다는 신호입니다.

### 11.5 단일 seed / 단일 샘플링 온도

대부분 `temp=0`, 고정 seed, 고정 prompt set입니다. 안정적인 비교에는 좋지만, 다양한 sampling 설정에서의 회복력은 아직 별도로 보지 않았습니다.

## 12. 디스크 상태와 정리

현재 외장디스크:

```text
/Volumes/Back_UP: 466Gi total, 450Gi used, 15Gi available
```

이번에 보존한 핵심 artifact:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-L10-HF-FP4.sidecar.gguf
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-L8-L12-HF-FP4.sidecar.gguf
```

보존할 가치:

- L10 source sidecar: 작은 smoke/regression test용
- L8-L12 source sidecar: source sidecar 중 가장 의미 있는 비교 후보
- L8-L12 source shards: 다음 full-layer 또는 top64/top128 sidecar 실험에 필요

삭제 후보:

- L10 source sidecar는 품질 후보가 아니므로 공간이 급하면 삭제 가능
- Late5 HF-FP4 sidecar는 proof-of-pipeline 성격이므로 보관 우선순위 낮음

## 13. 다음 단계 제안

이번 결과를 보고 다음 단계는 두 갈래 중 하나로 가야 합니다.

### A. 실사용 기준

바로 쓸 모델 기준은 다음이 맞습니다.

```text
nothink 일반 한국어: base
Think MAX / 한국어 지식형: Layer10Q4
장문 형식 준수: base 우선, L8-L12 source는 연구 후보
```

### B. 연구 기준

source-sidecar 방향을 계속 간다면 다음은 top32 확장이 아니라 coverage를 늘려야 합니다.

1. `Layer10 full expert source sidecar` 생성  
   L10의 256 routed experts 전체를 source shard에서 Q4 sidecar로 생성합니다. 예상 크기는 L10 top32의 8배 수준, 대략 3.4GiB 전후입니다.

2. `Layer10 top64/top128 source sidecar` 생성  
   top32가 너무 좁았을 가능성을 검증합니다.

3. `L8-L12 top64`는 후순위  
   KMMLU 300에서 L8-L12 top32가 크게 낮아졌으므로, 무작정 layer 폭을 넓히는 것보다 L10 coverage를 넓히는 쪽이 더 논리적입니다.

4. source format 재검증  
   가능하다면 full precision 또는 더 높은 정밀도의 upstream shard가 있는지 확인해야 합니다. 현재 source는 packed FP4라서 “고정밀 원본” 효과를 완전히 검증하지 못했습니다.

## 14. 최종 판정

이번 실험은 pipeline 관점에서는 성공, 모델 후보 관점에서는 부분 실패입니다.

성공한 것:

- L8-L12 source shards 확보
- HF packed FP4 source에서 sidecar GGUF 생성
- L10 및 L8-L12 sidecar runtime 로딩
- sidecar expert route 활성화 trace 확인
- Think MAX, KMMLU 100/300, 한국어 held-out, control, exact-copy, 장문 v2 평가 완료
- 속도 overhead가 작다는 점 확인

실패 또는 보류:

- source sidecar가 `Layer10Q4` 품질을 넘지 못함
- L8-L12 source sidecar가 KMMLU 300에서 base보다 낮음
- L10 source sidecar는 Think MAX와 KMMLU에서 약함
- exact-copy 개선 없음

따라서 지금 당장 채택할 운영 전략은 다음입니다.

```text
base는 일반 nothink 기본값으로 유지합니다.
Layer10Q4는 Think MAX 한국어 실험 후보로 유지합니다.
L8-L12 source sidecar는 연구 artifact로 보관하되 실전 기본값으로 쓰지 않습니다.
다음 실험은 Layer10 full/top64/top128 source sidecar로 진행합니다.
```

