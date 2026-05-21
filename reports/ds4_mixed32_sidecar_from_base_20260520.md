# DS4 KR-Mixed32 Sidecar GGUF 구현 및 평가 보고서

작성일: 2026-05-20  
작업 루트: `/Users/kch3dri4n/llm_provide/ds4`

## 1. 요약 결론

이번 작업에서는 `KR-Mixed32` expert-level sidecar GGUF를 실제로 생성하고, 기존 DS4 Metal runtime에서 sidecar expert가 선택되어 계산 경로에 들어가는 것까지 확인했습니다. 즉, “sidecar GGUF를 안전하게 읽는 설계 문서” 단계가 아니라, 실제 `./ds4 --bitlift-sidecar ...` 실행에서 38개 sidecar layer가 모두 route hit를 받는 단계까지 도달했습니다.

하지만 품질 평가는 명확히 부정적입니다. 현재 만든 `Mixed32 sidecar`는 고정밀 원본 weight에서 2bit expert를 4bit로 올린 것이 아니라, 이미 양자화된 base GGUF의 `IQ2_XXS/Q2_K` expert slice를 dequantize한 뒤 `Q4_K`로 다시 저장한 것입니다. 따라서 손실된 정보가 복구되지 않으며, 실제 평가에서도 `base` 및 `Layer10Q4`보다 한국어, KMMLU, exact-copy, 장문 지시문 안정성이 크게 낮았습니다.

최종 권장 사항은 다음과 같습니다.

- 일반 chat/nothink: `base` 또는 기존 `LateStable5Q4` 계열 유지
- Think MAX 한국어 실험: `Layer10Q4` 후보 유지
- 이번 `KR-Mixed32-from-base.sidecar.gguf`: runtime 검증용/실험 아티팩트로 보존, 기본 사용 비추천
- 다음 실제 bit-lift: base GGUF가 아니라 원본 BF16/FP16 또는 원래 JANG/MXTQ 변환 전 고정밀 expert weight에서 sidecar를 생성해야 함

## 2. 생성된 주요 산출물

### Full Mixed32 sidecar GGUF

- 실제 파일: `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-Mixed32-from-base.sidecar.gguf`
- 작업 디렉터리 symlink: `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-KR-Mixed32-from-base.sidecar.gguf`
- 크기: `17,213,440,736 bytes`, 약 `16.03 GiB`
- base GGUF symlink 대상 크기: `86,720,111,488 bytes`, 약 `80.78 GiB`
- base + sidecar 합산: 약 `96.81 GiB`

### 생성 요약

- sidecar 대상 layer 수: `38`
- layer 목록: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,27,29,30,31,32,33,35,37,38,39,40,41,42`
- expert slot 수: `1216` = `38 layers * 32 experts/layer`
- sidecar tensor 수: `152`
- tensor 구성: 각 layer마다 `gate/up/down/ids`
- source type id: `16 = IQ2_XXS`, `10 = Q2_K`
- output type: `Q4_K`
- source payload: 약 `8.01 GiB`
- Q4 sidecar payload: 약 `16.03 GiB`

관련 파일:

- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/mixed32_from_base_fixed.summary.json`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/mixed32_from_base_fixed.build.log`

## 3. 구현 내용

### 새로 추가한 writer

파일: `/Users/kch3dri4n/llm_provide/ds4/tools/write_bitlift_sidecar_from_base_gguf.py`

기능:

- base GGUF에서 selected expert slice를 직접 읽음
- `Q2_K`, `IQ2_XXS`, `Q4_K`, `F16`, `BF16`, `F32` 입력을 처리할 수 있도록 quants dylib 호출
- source expert slice를 chunk 단위로 dequantize
- selected expert를 `Q4_K`로 quantize
- sidecar GGUF에 `blk.L.ffn_*_exps.bitlift_q4.weight` 및 expert id tensor 기록
- 대형 모델 전체를 메모리에 올리지 않고 `row_chunk=128` 단위로 처리

### quants dequantize API 보강

수정 파일:

- `/Users/kch3dri4n/llm_provide/ds4/gguf-tools/quants.h`
- `/Users/kch3dri4n/llm_provide/ds4/gguf-tools/quants.c`

추가 API:

```c
bool ds4q_can_dequantize(ds4q_type type);
size_t ds4q_dequantize_chunk(ds4q_type type, const void *src, float *dst,
                             int64_t nrows, int64_t ncols);
```

지원한 dequantize type:

- `F32`
- `F16`
- `BF16`
- `Q2_K`
- `Q4_K`
- `IQ2_XXS`

### 중요한 버그 수정

초기 smoke sidecar에서 한국어 출력이 중국어/깨진 토큰으로 흔들렸고, 원인은 `IQ2_XXS` dequantize 구현이었습니다.

문제:

- `IQ2_XXS` grid raw 값 `1/3/5`를 그대로 magnitude로 사용함
- 실제 runtime dot-product 계열은 grid 값에 대해 magnitude table `8/25/43`을 사용함
- 이 차이 때문에 dequantize 후 Q4_K 재양자화가 원래 값을 심하게 왜곡함

수정:

- `grid[j] == 1 -> 8`
- `grid[j] == 3 -> 25`
- `grid[j] == 5 -> 43`

검증:

- 수정 전 IQ2 roundtrip MSE: 약 `0.388`
- 수정 후 IQ2 roundtrip MSE: 약 `0.062`

수정 후 smoke 출력:

```text
개인정보 보호에서 가장 중요한 원칙은 정보주체의 동의와 자기결정권입니다.
```

수정 후 full sidecar sanity 출력:

```text
개인정보 보호에서 가장 중요한 원칙은 **데이터 최소화**입니다. 필요한 정보만 수집하고 보관해야 합니다.
```

## 4. Runtime 활성화 검증

실행 방식:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-Mixed32-from-base.sidecar.gguf \
  --nothink -n 48 --temp 0 \
  -p '한국어로 두 문장 이내로 답하세요. 개인정보 보호에서 가장 중요한 원칙은 무엇인가요?'
```

runtime 로그:

- sidecar loaded layers: `38`
- mapped sidecar size: 약 `16416.03 MiB`
- sanity prefill: `60.09 tok/s`
- sanity generation: `26.47 tok/s`

Route trace 요약:

- sidecar layer count: `38`
- 모든 sidecar layer에서 실제 sidecar route hit 발생
- 평균 sidecar top-k share: `0.651`
- 최소/최대 sidecar top-k share: `0.205 / 0.779`
- 평균 row hit rate: `0.961`
- 최소/최대 row hit rate: `0.674 / 1.000`
- layer당 hit된 unique sidecar slot 평균: `23.9`

의미:

- GGUF가 단순히 load만 된 것이 아니라, routed-MoE top-k dispatch에서 sidecar expert가 실제로 선택됨
- expert-level sidecar runtime 경로 자체는 기능적으로 작동함
- 품질 문제는 sidecar dispatch 미작동 때문이 아니라, sidecar weight 품질/후보 선택/업퀀트 한계 문제로 보는 것이 맞음

관련 파일:

- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/mixed32_fixed_trace_summary.json`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/mixed32_fixed_trace_stderr.log`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/mixed32_fixed_trace_stdout.txt`

## 5. 평가 결과

### 5.1 프로젝트 평가 190개

평가 구성:

- held-out Korean 100
- English/Chinese/control 60
- exact-copy + long extra 30

결과:

| suite | model | pass / n | pass rate | avg prefill tok/s | avg decode tok/s |
|---|---|---:|---:|---:|---:|
| korean100 | base | 88 / 100 | 0.880 | 124.26 | 31.69 |
| korean100 | mixed32_sidecar | 39 / 100 | 0.390 | 105.09 | 26.13 |
| control60 | base | 60 / 60 | 1.000 | 59.51 | 32.03 |
| control60 | mixed32_sidecar | 46 / 60 | 0.767 | 52.17 | 26.41 |
| exact_long_extra | base | 10 / 30 | 0.333 | 137.96 | 31.49 |
| exact_long_extra | mixed32_sidecar | 4 / 30 | 0.133 | 117.00 | 26.05 |

속도 변화:

- `mixed32_sidecar` prefill은 base 대비 약 `84.6~87.7%`
- `mixed32_sidecar` decode는 base 대비 약 `82.4~82.7%`
- sidecar 계산 경로는 정상 동작하지만, 16GiB 추가 매핑과 Q4 sidecar expert dispatch 때문에 decode 속도가 약 17~18% 낮아짐

관찰:

- 일반 한국어도 pass rate가 `0.88 -> 0.39`로 크게 낮아짐
- exact-copy와 한글 자모 복사에서 mojibake, 설명 끼어들기, 잘림, 안전 토큰 파편이 나타남
- 영어/중국어/control에서도 `base`보다 명확히 퇴화

관련 파일:

- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/project_eval_base_vs_mixed32/summary.json`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/project_eval_base_vs_mixed32/scores.csv`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/project_eval_base_vs_mixed32/raw_results.jsonl`

### 5.2 KMMLU 100개

평가 구성:

- `HAERAE-HUB/KMMLU`
- deterministic local sample 100개
- greedy, `--nothink`
- 답변 형식: `1~4` 숫자만 출력

결과:

| model | correct / n | accuracy | invalid | avg prefill tok/s | avg decode tok/s |
|---|---:|---:|---:|---:|---:|
| base | 69 / 100 | 0.690 | 0 | 152.74 | 31.05 |
| mixed32_sidecar | 25 / 100 | 0.250 | 20 | 129.26 | 24.58 |

관찰:

- `mixed32_sidecar`는 accuracy가 `0.69 -> 0.25`로 하락
- invalid prediction이 `20/100` 발생
- 일부 출력에서 숫자 대신 설명 파편, `pp<ds_safety>...` 같은 이상 토큰이 발생
- 이 정도면 객관식 지식형에서도 실사용 후보가 아님

관련 파일:

- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/kmmlu100_base_vs_mixed32/summary.json`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/kmmlu100_base_vs_mixed32/REPORT.md`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/kmmlu100_base_vs_mixed32/scores.csv`

### 5.3 Think MAX expanded30

평가 구성:

- 30 prompt expanded set
- suites: Korean, long, exact, English/Chinese/control
- 모델: `base`, `Layer10Q4`, `mixed32_sidecar`
- mode: `--think-max`

결과:

| model | pass / n | pass rate | avg prefill tok/s | avg decode tok/s | Korean pass | Long pass | Control pass | Exact pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10 / 30 | 0.333 | 150.83 | 31.38 | 3 / 10 | 3 / 8 | 4 / 6 | 0 / 6 |
| Layer10Q4 | 17 / 30 | 0.567 | 150.76 | 31.06 | 8 / 10 | 5 / 8 | 4 / 6 | 0 / 6 |
| mixed32_sidecar | 10 / 30 | 0.333 | 129.28 | 25.98 | 5 / 10 | 2 / 8 | 3 / 6 | 0 / 6 |

판단:

- Think MAX에서는 `Layer10Q4`가 가장 좋은 후보
- `mixed32_sidecar`는 base와 전체 pass rate가 같지만, 한국어/장문/컨트롤의 안정성이 Layer10Q4보다 낮음
- 속도도 `Layer10Q4`는 base와 거의 같지만, `mixed32_sidecar`는 decode가 약 17% 느림
- exact-copy는 세 모델 모두 취약하므로 별도 prompt/decoding/format-control 개선이 필요

관련 파일:

- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/thinkmax30_base_layer10_mixed32/thinkmax_full_sidecar_summary.json`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/thinkmax30_base_layer10_mixed32/thinkmax_full_sidecar_scores.csv`
- `/Users/kch3dri4n/llm_provide/ds4/runs/20260520_full_sidecar_from_base/thinkmax30_base_layer10_mixed32/thinkmax_full_sidecar_raw_results.jsonl`

## 6. 왜 Mixed32가 나빠졌는가

### 6.1 “업퀀트”는 bit-lift가 아님

이번 파일은 base GGUF에서 selected expert를 뽑았습니다. base GGUF의 routed expert 대부분은 이미 `IQ2_XXS` 또는 `Q2_K`입니다.

따라서 절차는 다음과 같습니다.

```text
이미 손실된 2bit 계열 expert
→ float로 dequantize
→ Q4_K로 다시 quantize
→ sidecar로 dispatch
```

이 과정은 저장 형식만 Q4가 될 뿐, 원래 4bit에 해당하는 정보를 복원하지 못합니다. 오히려 dequantize/requantize 과정과 runtime dispatch 차이 때문에 원래 base의 조정된 양자화 특성이 깨질 수 있습니다.

### 6.2 top32/layer가 너무 넓음

38개 layer 전체에서 32개 expert/layer를 sidecar로 바꾼 것은 route trace 관점에서는 강하게 활성화되지만, 품질 관점에서는 영향 범위가 큽니다. trace에서는 평균 sidecar top-k share가 `65.1%`였으므로, 생성 중 상당수 routed expert 계산이 새 sidecar 값으로 대체됩니다.

이 값이 “고정밀 개선 expert”라면 긍정적일 수 있지만, 이번에는 “base에서 재포장한 Q4 expert”이기 때문에 넓은 대체 범위가 오히려 품질 위험으로 작동했습니다.

### 6.3 후보 선정은 한국어 routing이지 한국어 특이 expert가 아님

Mixed32 후보는 한국어 prefill/decode/think routing trace를 반영했습니다. 하지만 이것은 “한국어에서 자주 쓰인 expert”이지 “한국어만 개선하는 expert”가 아닙니다.

정확히 분리하려면 다음 점수가 필요합니다.

```text
ko_usage_score / control_usage_score
```

또는 layer별로 영어/중국어/control에서 함께 쓰이는 expert를 제외하는 penalty가 필요합니다.

## 7. 현재 프로젝트 한계

- 원본 고정밀 weight가 없어서 진짜 bit-lift를 수행하지 못했습니다.
- sidecar writer는 base GGUF에서 재양자화하므로 품질 개선보다는 runtime 검증에 가깝습니다.
- exact-copy 평가는 세 모델 모두 취약하므로 모델만 바꿔 해결할 수 있는 문제가 아닐 수 있습니다.
- KMMLU 100개는 local regression signal이며 공개 벤치 수준의 대표성은 없습니다.
- Think MAX expanded30도 수가 작아 경향 확인용입니다.
- Back_UP 외장 디스크 여유가 약 `20GiB`뿐이라 추가 16GiB급 sidecar를 여러 개 만드는 것은 위험합니다.

## 8. 다음 실행 계획

### P0. Mixed32-from-base는 보존하되 기본 후보에서 제외

현재 sidecar는 runtime proof artifact로 유지합니다.

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-Mixed32-from-base.sidecar.gguf
```

기본 사용 후보로 올리지는 않습니다.

### P1. Think MAX는 Layer10Q4를 우선 후보로 유지

현재 Think MAX 30개 평가에서:

```text
Layer10Q4: 17/30 pass
base:      10/30 pass
Mixed32:   10/30 pass
```

속도도 Layer10Q4가 base와 거의 같으므로, Think MAX 한국어 실험은 계속 Layer10Q4 중심으로 가는 것이 맞습니다.

### P2. 진짜 bit-lift sidecar는 고정밀 원본에서 다시 생성

필요한 입력:

- BF16/FP16 expert tensors
- 또는 JANG/MXTQ 변환 전 중간 산출물
- 또는 최소 Q4 이상 원본 expert checkpoint

그 다음에야 다음 경로가 의미 있습니다.

```text
고정밀 expert
→ 한국어/control differential trace로 후보 선정
→ selected expert만 Q4_K/Q5_K sidecar
→ runtime dispatch
→ 한국어 + control + KMMLU + exact-copy 재평가
```

### P3. 후보 수를 줄인 sidecar 실험

고정밀 원본을 구하기 전에는 큰 sidecar 대신 작은 실험만 권장합니다.

- stable core 5~16 experts/layer
- late-layer only
- Layer10Q4처럼 특정 layer 단위 변형
- exact-copy/control에서 자주 쓰이는 expert는 제외

### P4. 평가 체계 개선

다음 평가에서는 다음 세트를 고정해야 합니다.

- KMMLU 300개
- Think MAX 30개
- 장문 지시문 v2 60개
- exact-copy 강화 50개
- 영어/중국어/control 각각 50개
- route trace diff: Korean vs control

## 9. 최종 판단

이번 작업의 기술적 성과는 분명합니다. expert-level sidecar GGUF writer와 runtime dispatch 검증이 되었고, route trace로 38개 sidecar layer가 모두 실제 활성화됨을 확인했습니다.

하지만 `KR-Mixed32-from-base`는 품질 후보로는 탈락입니다. 현재 결과는 “한국어 expert 후보를 실제 GGUF sidecar로 만들 수 있다”는 엔지니어링 가능성을 입증했지만, “base GGUF에서 뽑아 Q4로 재포장하면 한국어 품질이 좋아진다”는 가설은 기각했습니다.

다음 성공 가능성이 높은 방향은 `Layer10Q4`를 Think MAX 후보로 유지하면서, 진짜 고정밀 source weight 기반의 작은 sidecar부터 다시 bit-lift하는 것입니다.
