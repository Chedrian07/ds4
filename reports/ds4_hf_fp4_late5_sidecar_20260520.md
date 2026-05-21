# DS4 HF-FP4 Late5 Sidecar 결과 보고서

작성일: 2026-05-20  
작업 디렉터리: `/Users/kch3dri4n/llm_provide/ds4`

## 1. 최종 결론

고정밀 원본 weight 기반 sidecar 경로는 성공했습니다. 기존 `base GGUF -> Q4 sidecar` 재포장 방식은 품질이 크게 무너졌지만, 이번에는 로컬에 남아 있던 HF 원본 계열 shard의 `packed FP4 + F8_E8M0 scale`에서 직접 Q4_K sidecar를 생성했고, Metal 런타임에서 실제 routed expert dispatch까지 정상 동작했습니다.

다만 품질 결론은 보수적입니다.

- 일반 chat / nothink 기본 모델로는 여전히 `base` 또는 기존 안정 후보를 유지하는 것이 맞습니다.
- Think MAX 실험 후보로는 `Layer10Q4`가 아직 1순위입니다.
- 새 `ThinkTop32-Late5-HF-FP4` sidecar는 base보다 Think MAX에서 개선을 보였지만, Layer10Q4를 넘지는 못했습니다.
- 이번 성과의 핵심은 "source-weight sidecar 생성 및 runtime 계산 경로가 된다"는 증명입니다. 다음 승부는 Late5가 아니라 Layer10 근방 또는 Think MAX trace 기반 early/mid layer 원본 shard를 확보해서 같은 방식으로 sidecar를 만드는 것입니다.

## 2. 생성한 sidecar

생성 파일:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf
```

로컬 링크:

```text
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf
```

크기와 체크섬:

```text
file size: 2,264,926,752 bytes, about 2.1 GiB
sha256: 6a13f9cf493b2683f3ee4c5fce905732e042d3a3e9025d6e03d53121e87a9309
```

구성:

```text
source: HF packed FP4 I8 weight + F8_E8M0 scale
layers: L37, L38, L40, L41, L42
expert slots: 160
experts per layer: 32
tensors: 20
sidecar qtype: Q4_K
payload bytes: 2,264,924,800
```

사용한 local source shards:

```text
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00039-of-00046.safetensors
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00040-of-00046.safetensors
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00042-of-00046.safetensors
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00043-of-00046.safetensors
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5/model-00044-of-00046.safetensors
```

주의: 이것은 BF16/FP16 원본이 아니라, 현재 로컬에 확보된 HF upstream 계열의 FP4+E8M0 source입니다. 그래도 기존 base GGUF의 routed expert Q2/IQ2 재포장보다 source 손실이 적은 입력입니다.

## 3. 구현 내용

새 writer:

```text
/Users/kch3dri4n/llm_provide/ds4/tools/write_bitlift_sidecar_from_hf_fp4.py
```

핵심 동작:

- safetensors index와 shard header를 직접 읽습니다.
- HF tensor 이름 `layers.N.ffn.experts.E.w1/w2/w3.{weight,scale}`를 sidecar tensor로 매핑합니다.
- `w1 -> gate`, `w3 -> up`, `w2 -> down`으로 매핑합니다.
- packed FP4 nibble을 E8M0 scale로 dequant한 뒤 Q4_K로 재양자화합니다.
- compact sidecar GGUF tensor를 씁니다.

추가한 quantizer 경로:

```text
/Users/kch3dri4n/llm_provide/ds4/gguf-tools/quants.c
/Users/kch3dri4n/llm_provide/ds4/gguf-tools/quants.h
```

핵심 함수:

```text
ds4q_quantize_fp4_e8m0_to_q4_k_chunk(...)
```

런타임 쪽 기존 sidecar loader/Metal dispatch는 새 sidecar를 정상 인식했습니다.

## 4. 런타임 검증

inspect 결과:

```text
bitlift sidecar: layers=5 expert_slots=160 tensor_triplets=15 qtype=q4_k
Metal sidecar mapped: about 2160.02 MiB
```

smoke sidecar도 따로 생성했습니다.

```text
runs/20260520_hf_fp4_sidecar/smoke_layer37_2_from_hf_fp4.sidecar.gguf
size: about 27 MiB
layer: 37
expert slots: 2
```

생성 테스트:

- `--nothink` 생성 정상
- `--think-max --ctx 393216` 생성 정상
- `DS4_BITLIFT_TRACE_HITS=1`에서 sidecar route hit 로그 정상

## 5. Expert 활성화 확인

### nothink trace

파일:

```text
runs/20260520_hf_fp4_sidecar/late5_hf_fp4_nothink_trace_summary.json
```

요약:

| layer | rows/tokens | base routes | sidecar routes | sidecar rate |
|---:|---:|---:|---:|---:|
| 37 | 139 | 243 | 591 | 70.86% |
| 38 | 139 | 209 | 625 | 74.94% |
| 40 | 139 | 237 | 597 | 71.58% |
| 41 | 139 | 210 | 624 | 74.82% |
| 42 | 139 | 246 | 588 | 70.50% |

### Think MAX trace

파일:

```text
runs/20260520_hf_fp4_sidecar/late5_hf_fp4_thinkmax_trace_summary.json
```

요약:

| layer | rows/tokens | base routes | sidecar routes | sidecar rate |
|---:|---:|---:|---:|---:|
| 37 | 219 | 639 | 675 | 51.37% |
| 38 | 219 | 434 | 880 | 66.97% |
| 40 | 219 | 520 | 794 | 60.43% |
| 41 | 219 | 514 | 800 | 60.88% |
| 42 | 219 | 655 | 659 | 50.15% |

해석:

- sidecar expert는 실제로 꽤 자주 활성화됩니다.
- nothink에서는 late layer 후보가 매우 자주 잡힙니다.
- Think MAX에서는 활성률이 낮아지지만 여전히 절반 이상 경로에서 sidecar가 사용됩니다.
- 따라서 품질이 부족한 원인은 "sidecar가 안 쓰여서"가 아니라 "Late5 top32가 최적 품질 위치가 아니어서"로 보는 것이 맞습니다.

## 6. 구조화 평가 결과

평가 경로:

```text
runs/20260520_hf_fp4_sidecar/project_eval_base_layer10_late5hf
```

모델:

```text
base
layer10q4
thinktop32_late5_hf_fp4
```

### korean100

| model | pass | rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 88/100 | 88% | 123.01 | 31.85 |
| layer10q4 | 84/100 | 84% | 122.96 | 31.28 |
| thinktop32_late5_hf_fp4 | 84/100 | 84% | 120.12 | 30.94 |

### control60

| model | pass | rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 60/60 | 100% | 59.25 | 32.08 |
| layer10q4 | 60/60 | 100% | 58.85 | 31.79 |
| thinktop32_late5_hf_fp4 | 60/60 | 100% | 58.08 | 31.30 |

### exact_long_extra

| model | pass | rate | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|
| base | 10/30 | 33.33% | 138.25 | 31.50 |
| layer10q4 | 10/30 | 33.33% | 134.12 | 31.38 |
| thinktop32_late5_hf_fp4 | 10/30 | 33.33% | 133.41 | 30.72 |

해석:

- 새 sidecar는 control 퇴화를 만들지 않았습니다.
- 속도는 base 대비 decode 약 2.9%, prefill 약 2.3% 정도 낮습니다.
- 한국어 held-out 100에서는 base를 못 이겼습니다.
- exact/long 추가 세트는 개선 없음입니다.

## 7. Think MAX 30 결과

평가 경로:

```text
runs/20260520_hf_fp4_sidecar/thinkmax30_base_layer10_late5hf
```

| model | pass | rate | avg prefill t/s | avg decode t/s | avg generated tokens |
|---|---:|---:|---:|---:|---:|
| base | 10/30 | 33.33% | 149.79 | 31.47 | 304.57 |
| layer10q4 | 17/30 | 56.67% | 149.94 | 31.05 | 319.70 |
| thinktop32_late5_hf_fp4 | 14/30 | 46.67% | 148.13 | 30.51 | 291.27 |

Suite별:

| model | control | exact | korean | long |
|---|---:|---:|---:|---:|
| base | 4/6 | 0/6 | 3/10 | 3/8 |
| layer10q4 | 4/6 | 0/6 | 8/10 | 5/8 |
| thinktop32_late5_hf_fp4 | 4/6 | 0/6 | 6/10 | 4/8 |

해석:

- 새 sidecar는 Think MAX에서 base보다 좋습니다.
- 하지만 Layer10Q4에는 못 미칩니다.
- Think MAX 후보로 "가능성 있음"이지만, 현재 best는 아닙니다.

## 8. KMMLU 100 결과

평가 경로:

```text
runs/20260520_hf_fp4_sidecar/kmmlu100_base_layer10_late5hf
```

| model | correct | accuracy | invalid | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 70/100 | 70% | 0 | 152.94 | 31.08 |
| layer10q4 | 79/100 | 79% | 0 | 150.19 | 30.83 |
| thinktop32_late5_hf_fp4 | 72/100 | 72% | 0 | 150.22 | 30.21 |

해석:

- 새 sidecar는 KMMLU 100에서 base보다 +2점입니다.
- Layer10Q4는 +9점으로 더 강합니다.
- invalid prediction은 0이라 객관식 출력 안정성은 괜찮습니다.

## 9. 속도 판단

전체적으로 새 sidecar는 속도 손실이 작습니다.

- nothink structured eval decode: base 31.85 t/s, sidecar 30.94 t/s
- Think MAX decode: base 31.47 t/s, sidecar 30.51 t/s
- KMMLU decode: base 31.08 t/s, sidecar 30.21 t/s

대략 2.5~3.5% 수준의 decode 손실입니다. 2.1GiB sidecar를 추가 매핑했지만 Metal runtime 경로는 안정적으로 유지됐습니다.

## 10. 한계

현재 local source shard가 late 5개 layer뿐입니다.

```text
available: L37, L38, L40, L41, L42
missing for full Mixed32/ThinkTop32: 대부분의 early/mid layer shards
```

이번 결과만으로는 "한국어 expert bit-lift의 최종 답"을 고를 수 없습니다. 오히려 실험 결과는 Layer10 근방이 더 중요하다는 쪽을 지지합니다.

또한 현재 source는 BF16 원본이 아니라 HF FP4+E8M0 source입니다. base GGUF보다 높은 출발점이지만, 진짜 BF16/FP16에서 Q4로 직접 내린 결과와는 다를 수 있습니다.

## 11. 추천 다음 단계

1. 일반 chat/nothink 기본값은 `base` 또는 기존 안정 후보 유지.
2. Think MAX 한국어 실험은 계속 `Layer10Q4`를 1순위로 유지.
3. 이번 `ThinkTop32-Late5-HF-FP4`는 폐기하지 말고 source-sidecar pipeline 검증용 baseline으로 보관.
4. 다음 생성 후보는 `Layer10` 또는 `L8-L12` 원본 shard 기반 sidecar.
5. 가능하면 전체 checkpoint를 저장하지 말고 필요한 shard만 순차 다운로드/변환/삭제하는 streaming 방식으로 진행.
6. 이후 `KMMLU 300`, `Think MAX 30`, `long-instruction v2`를 Layer10 source-sidecar 후보와 다시 비교.

## 12. 실사용 명령

inspect:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf \
  --inspect
```

nothink:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf \
  --nothink -n 256 --temp 0 \
  -p '한국어로 답하세요. 자료구조에서 스택과 큐의 차이를 설명해 주세요.'
```

Think MAX:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf \
  --think-max --ctx 393216 -n 512 --temp 0 \
  -p '한국어로 답하세요. 보안 업데이트가 중요한 이유를 설명해 주세요.'
```

trace:

```bash
DS4_BITLIFT_TRACE_HITS=1 ./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Late5-HF-FP4.sidecar.gguf \
  --think-max --ctx 393216 -n 128 --temp 0 \
  -p '한국어로 짧게 답하세요. 입력 검증과 출력 인코딩의 차이는 무엇인가요?'
```

