# DS4 Korean Bit-Lift LateStable5Q4 결과 보고

작성일: 2026-05-18  
작업 디렉터리: `/Users/kch3dri4n/llm_provide/ds4`

## 1. 이번 작업의 결론

`KR-LateStable5Q4` GGUF 생성은 완료됐고, 런타임 로딩 문제까지 수정해 실제 추론, batch 평가, Think MAX 벤치, expert usage trace까지 수행했습니다.

다만 모델 품질 관점에서는 `LateStable5Q4`를 바로 본 모델로 승격하기는 이릅니다. `Worst5Q4`보다는 한국어 long instruction에서 확실히 나아졌지만, 전체 held-out 한국어 점수는 base보다 낮습니다.

핵심 판단은 다음과 같습니다.

- 생성 성공: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf`
- Q4 적용 layer: `L37`, `L38`, `L40`, `L41`, `L42`
- 파일 크기: `95,779,807,840 bytes`, Finder 기준 약 `89G`, 약 `89.20 GiB`
- 한국어 100개 평가: base `88/100`, Worst5Q4 `80/100`, LateStable5Q4 `84/100`
- 영어/중국어/control 평가: 세 모델 모두 `60/60`
- exact/long extra: 세 모델 모두 `10/30`
- Think MAX: 속도는 정상, LateStable5Q4 자동 점수는 `2/10`으로 base/Worst의 `4/10`보다 낮음
- expert trace: Q4로 올린 late layer들은 실제로 넓게 활성화됨. think trace에서는 stable core 5개 모두 선택됨.

최종적으로는 `LateStable5Q4`는 “실험적으로 의미 있는 후보”이지만, 현재 데이터 기준 추천 순위는 아직 `base > LateStable5Q4 > Worst5Q4`입니다.

## 2. 생성한 GGUF

생성 파일:

```text
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf
```

크기 비교:

```text
base       86,720,111,488 bytes
Worst5Q4  95,779,807,840 bytes
LateStable5Q4 95,779,807,840 bytes
```

`LateStable5Q4`는 base 대비 약 `+8.44 GiB`입니다. `Worst5Q4`와 같은 수의 routed expert layer 5개를 Q4_K로 올렸기 때문에 총 크기는 동일합니다.

## 3. Q4 적용 검증

검증 결과:

```text
LateStable5Q4
L23: gate=iq2_xxs down=q2_K up=iq2_xxs
L25: gate=iq2_xxs down=q2_K up=iq2_xxs
L28: gate=iq2_xxs down=q2_K up=iq2_xxs
L34: gate=iq2_xxs down=q2_K up=iq2_xxs
L36: gate=iq2_xxs down=q2_K up=iq2_xxs
L37: gate=q4_K down=q4_K up=q4_K
L38: gate=q4_K down=q4_K up=q4_K
L40: gate=q4_K down=q4_K up=q4_K
L41: gate=q4_K down=q4_K up=q4_K
L42: gate=q4_K down=q4_K up=q4_K
```

즉 기존 `Worst5Q4`의 `L23/L25/L28/L34/L36`과는 겹치지 않고, 이번 후보는 late layer stable 후보 5개만 Q4_K로 올라갔습니다.

## 4. 구현 변경 사항

### 4.1 partial HF shard 기반 생성

기존 `deepseek4-quantize`는 전체 GGUF를 만들 때 모든 tensor를 원본 HF safetensors에서 다시 생성하는 구조였습니다. 원본 DeepSeek-V4-Flash 전체 safetensors는 약 `159.6GB`라 로컬 디스크와 반복 실험에 부담이 컸습니다.

이번에는 quantizer에 `--copy-unchanged` 옵션을 추가했습니다.

동작 방식:

- type이 바뀌지 않는 tensor는 기존 base GGUF에서 그대로 스트리밍 복사
- Q4_K로 바뀌는 tensor만 원본 HF shard에서 재생성
- 따라서 이번 생성에는 late 5개 layer에 해당하는 원본 shard 5개만 필요

받은 원본 shard:

```text
model-00039-of-00046.safetensors  L37
model-00040-of-00046.safetensors  L38
model-00042-of-00046.safetensors  L40
model-00043-of-00046.safetensors  L41
model-00044-of-00046.safetensors  L42
model.safetensors.index.json
```

현재 이 partial HF 원본은 외장 디스크로 이동했습니다.

```text
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5
```

프로젝트 내부에는 symlink만 남겼습니다.

```text
/Users/kch3dri4n/llm_provide/ds4/hf-partial/DeepSeek-V4-Flash-late5
```

### 4.2 Metal runtime view overlap 수정

최초 `LateStable5Q4` smoke test에서 다음 에러가 발생했습니다.

```text
ds4: Metal model range 79.89..81.01 GiB is not covered by mapped model views
```

원인:

- 기존 Metal model view overlap 기준은 `704,643,072 bytes`, 즉 약 `672 MiB`
- Q4_K routed expert tensor 하나는 `1,152 MiB`
- `blk.42.ffn_up_exps.weight`가 view 경계를 걸치면서 하나의 Metal buffer view 안에 완전히 들어가지 못함

수정:

```c
#define DS4_METAL_MODEL_MAX_TENSOR_BYTES (1152ull * 1024ull * 1024ull)
```

수정 후 smoke test:

```text
LateStable5Q4 ctx=4096
prefill: 68.15 t/s
generation: 32.84 t/s
output: 안녕하세요, 무엇을 도와드릴까요?
```

이후 batch 평가와 Think MAX에서도 런타임 에러 없이 동작했습니다.

## 5. 구조 평가 결과

평가 디렉터리:

```text
/tmp/ds4-ko-cal/structured_eval_latestable5q4
```

평가 조건:

- `--metal`
- `ctx=4096`
- `--nothink`
- `temperature=0`
- 동일 prompt 190개
- 모델 3개: base, Worst5Q4, LateStable5Q4

### 5.1 suite별 요약

| suite | base | Worst5Q4 | LateStable5Q4 |
|---|---:|---:|---:|
| korean100 | 88/100 | 80/100 | 84/100 |
| control60 | 60/60 | 60/60 | 60/60 |
| exact_long_extra | 10/30 | 10/30 | 10/30 |

### 5.2 한국어 100개 세부

| kind | base | Worst5Q4 | LateStable5Q4 |
|---|---:|---:|---:|
| daily | 16/20 | 20/20 | 16/20 |
| exact | 20/20 | 20/20 | 20/20 |
| long | 12/20 | 0/20 | 8/20 |
| summary | 20/20 | 20/20 | 20/20 |
| tech | 20/20 | 20/20 | 20/20 |

해석:

- `Worst5Q4`는 daily는 좋아졌지만 long instruction이 완전히 무너졌습니다.
- `LateStable5Q4`는 long instruction을 `0/20 -> 8/20`으로 회복했습니다.
- 그러나 daily는 base와 같은 `16/20`으로 내려갔고, base의 long `12/20`에는 못 미쳤습니다.
- summary, tech, exact-copy 기본 세트는 세 모델 모두 안정적입니다.

### 5.3 control 퇴화 확인

| kind | base | Worst5Q4 | LateStable5Q4 |
|---|---:|---:|---:|
| chinese | 20/20 | 20/20 | 20/20 |
| english | 20/20 | 20/20 | 20/20 |
| control_exact | 20/20 | 20/20 | 20/20 |

control suite에서는 퇴화가 보이지 않았습니다.

### 5.4 exact/long extra

| kind | base | Worst5Q4 | LateStable5Q4 |
|---|---:|---:|---:|
| exact | 0/20 | 0/20 | 0/20 |
| long | 10/10 | 10/10 | 10/10 |

extra exact는 세 모델 모두 실패했습니다. 주로 한글 자모와 label 보존이 엄격한 exact-copy 문제입니다. 이건 bit-lift 후보 간 차이보다 현재 모델/프롬프트/디코딩 체계의 공통 약점으로 보는 게 맞습니다.

## 6. 속도 결과

### 6.1 구조 평가 평균 속도

| suite | model | avg prefill t/s | avg generation t/s |
|---|---|---:|---:|
| korean100 | base | 122.22 | 32.58 |
| korean100 | Worst5Q4 | 121.41 | 31.66 |
| korean100 | LateStable5Q4 | 123.33 | 31.41 |
| control60 | base | 58.86 | 32.15 |
| control60 | Worst5Q4 | 58.81 | 32.05 |
| control60 | LateStable5Q4 | 59.46 | 31.79 |
| exact_long_extra | base | 135.98 | 31.64 |
| exact_long_extra | Worst5Q4 | 135.66 | 31.32 |
| exact_long_extra | LateStable5Q4 | 137.18 | 31.24 |

속도 해석:

- LateStable5Q4의 prefill은 base/Worst와 동급입니다.
- decode는 base보다 약 `3.6%` 낮고 Worst와 거의 비슷합니다.
- Q4_K late layer 5개 추가로 인한 속도 손실은 실사용상 크지 않습니다.

## 7. Think MAX 결과

벤치 디렉터리:

```text
/tmp/ds4-ko-cal/thinkmax_bench_latestable5q4_20260518
```

조건:

- `--think-max`
- `ctx=393216`
- 10 prompt
- base, Worst5Q4, LateStable5Q4

요약:

| model | pass | avg prefill t/s | avg generation t/s | avg generated tokens |
|---|---:|---:|---:|---:|
| base | 4/10 | 138.21 | 31.35 | 261.9 |
| Worst5Q4 | 4/10 | 137.06 | 31.25 | 268.8 |
| LateStable5Q4 | 2/10 | 137.05 | 31.24 | 256.2 |

suite별:

```text
base:          control 2/3, exact 0/2, korean 2/4, long 0/1
Worst5Q4:      control 2/3, exact 0/2, korean 2/4, long 0/1
LateStable5Q4: control 2/3, exact 0/2, korean 0/4, long 0/1
```

해석:

- Think MAX 런타임은 세 모델 모두 정상 동작합니다.
- 속도도 거의 동일합니다.
- 다만 LateStable5Q4는 Think MAX 한국어 자동 점수에서 좋지 않았습니다.
- 이 scoring은 final answer extraction이 완벽하지 않아서 절대값보다 상대 신호로 봐야 합니다.
- 그래도 LateStable을 Think MAX 주력 후보로 바로 쓰기는 어렵습니다.

## 8. Expert Usage Trace

trace 디렉터리:

```text
/tmp/ds4-ko-cal/expert_usage_latestable5q4
```

대상 stable core:

```text
L40:E037
L41:E184
L38:E021
L37:E025
L42:E032
```

### 8.1 nothink decode64, 10 prompts

조건:

```text
dataset: rendered_prompts_nothink.txt
ctx=4096
max_prompts=10
decode_tokens=64
prompt_tokens=15310
decode_tokens_done=325
routed expert observations=83850
```

stable core counts:

| expert | selected_count | count_share | weight_share |
|---|---:|---:|---:|
| L40:E037 | 1 | 0.000513 | 0.000423 |
| L41:E184 | 9 | 0.004615 | 0.003490 |
| L38:E021 | 0 | 0.000000 | 0.000000 |
| L37:E025 | 0 | 0.000000 | 0.000000 |
| L42:E032 | 32 | 0.016410 | 0.013919 |

target layer active expert 수:

```text
L37 active experts: 200 / 256
L38 active experts: 193 / 256
L40 active experts: 179 / 256
L41 active experts: 200 / 256
L42 active experts: 198 / 256
```

### 8.2 think decode64, 10 prompts

조건:

```text
dataset: rendered_prompts_think.txt
ctx=4096
max_prompts=10
decode_tokens=64
prompt_tokens=15539
decode_tokens_done=258
routed expert observations=66564
```

stable core counts:

| expert | selected_count | count_share | weight_share |
|---|---:|---:|---:|
| L40:E037 | 18 | 0.011628 | 0.009122 |
| L41:E184 | 9 | 0.005814 | 0.004125 |
| L38:E021 | 1 | 0.000646 | 0.000391 |
| L37:E025 | 1 | 0.000646 | 0.000412 |
| L42:E032 | 37 | 0.023902 | 0.020571 |

target layer active expert 수:

```text
L37 active experts: 170 / 256
L38 active experts: 174 / 256
L40 active experts: 164 / 256
L41 active experts: 183 / 256
L42 active experts: 174 / 256
```

해석:

- Q4로 올린 late layer들은 실제로 많은 expert가 선택됩니다.
- think trace에서는 stable core 5개가 모두 활성화됐습니다.
- nothink trace에서는 stable core 중 3개만 활성화됐습니다.
- 즉 “stable core가 항상 모든 짧은 샘플에서 뜬다”는 식으로 해석하면 안 됩니다.
- layer-level Q4는 expert-level sidecar보다 더 넓은 보호막을 주지만, 비용도 더 큽니다.

## 9. 한계점

이번 결과의 한계는 분명합니다.

1. Q4_K 변경 tensor에는 별도 imatrix를 적용하지 않았습니다.  
   기존 base에서 복사된 tensor는 기존 quant 상태를 유지하지만, 새로 생성된 Q4_K tensor는 이번 partial HF 기반 quantizer에서 imatrix 없이 생성됐습니다. 다음 고품질 변환에서는 late layer 대상 imatrix를 새로 만들어 넣는 편이 더 정직합니다.

2. 평가 점수는 휴리스틱입니다.  
   특히 long instruction과 Think MAX는 형식 조건을 기계적으로 채점합니다. 실제 사람이 보면 일부 실패가 쓸 만할 수도 있고, 반대로 pass가 품질적으로 빈약할 수도 있습니다.

3. Think MAX 출력은 scoring과 맞지 않습니다.  
   Think MAX는 reasoning output이 길어지기 때문에 final answer extraction이 중요합니다. 현재 평가는 “실제 reasoning 모드 품질”이라기보다 “이 scoring harness에서의 통과율”입니다.

4. 한국어 특이 expert와 일반 고활성 expert가 아직 분리되지 않았습니다.  
   이번 layer 후보는 한국어 trace 기반이지만, 영어/중국어/control 대비 특이성 비율까지 반영한 것은 아닙니다.

5. 이번 후보는 layer-level lift입니다.  
   사용자가 원한 방향대로 Mixed32 같은 expert-level lift는 피했습니다. 대신 layer 전체를 올렸기 때문에 특정 layer 안의 모든 expert가 비용을 먹습니다.

## 10. 다음 의사결정

현재 결과로는 다음 순서를 권합니다.

1. `Worst5Q4`는 단독 후보에서 내립니다.  
   daily는 좋아졌지만 long instruction이 `0/20`이라 너무 위험합니다.

2. `LateStable5Q4`는 보관하되 본 후보로 승격하지 않습니다.  
   Worst보다 long이 나아졌고 control 퇴화가 없지만, base보다 한국어 총점과 Think MAX 점수가 낮습니다.

3. 다음 실험은 layer-level을 유지한다면 `KR-Layer10Q4`가 가장 합리적입니다.  
   즉 `L23/L25/L28/L34/L36`과 `L37/L38/L40/L41/L42`를 합친 10개 layer Q4_K입니다. Worst 쪽 daily 개선과 LateStable 쪽 long 회복이 같이 나타나는지 확인할 수 있습니다.

4. `KR-Layer10Q4`도 같은 190 prompt와 Think MAX 10 prompt로 평가해야 합니다.  
   base를 이기지 못하면 더 큰 Q4 확장은 중단하고, expert-level sidecar 또는 imatrix 개선으로 돌아가는 편이 낫습니다.

5. 최종 목표가 한국어 품질이면 다음 trace는 반드시 control 대비 비율로 뽑아야 합니다.  
   `ko_usage / en_usage`, `ko_usage / zh_usage`, `ko_usage / generic_code_usage`를 layer와 expert별로 계산해야 진짜 한국어 특화 후보를 고를 수 있습니다.

## 11. 산출물 경로

GGUF:

```text
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf
```

구조 평가:

```text
/tmp/ds4-ko-cal/structured_eval_latestable5q4/summary.json
/tmp/ds4-ko-cal/structured_eval_latestable5q4/scores.csv
/tmp/ds4-ko-cal/structured_eval_latestable5q4/raw_results.jsonl
```

Think MAX:

```text
/tmp/ds4-ko-cal/thinkmax_bench_latestable5q4_20260518/thinkmax_latestable5q4_summary.json
/tmp/ds4-ko-cal/thinkmax_bench_latestable5q4_20260518/thinkmax_latestable5q4_scores.csv
/tmp/ds4-ko-cal/thinkmax_bench_latestable5q4_20260518/thinkmax_latestable5q4_raw_results.jsonl
```

Expert usage:

```text
/tmp/ds4-ko-cal/expert_usage_latestable5q4/latestable5q4_decode64_nothink_10prompts.csv
/tmp/ds4-ko-cal/expert_usage_latestable5q4/latestable5q4_decode64_think_10prompts.csv
```

Partial original HF shards:

```text
/Volumes/Back_UP/hf-cache-offload/deepseek-ai/DeepSeek-V4-Flash-late5
```

