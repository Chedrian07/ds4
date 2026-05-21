# DS4 한국어 Expert Trace 및 GGUF 1차 결과 보고서

작성일: 2026-05-18  
작업 경로: `/Users/kch3dri4n/llm_provide/ds4`

## 1. 요약

이번 작업의 목표는 한국어 캘리브레이션 데이터셋을 사용해 DeepSeek V4 Flash 계열 모델의 routed expert 사용 패턴을 측정하고, 실제 trace 기반으로 한국어 품질 개선 후보 expert를 선정한 뒤, 이를 GGUF 변환 실험까지 연결하는 것이었습니다.

현재까지 완료된 핵심 산출물은 다음과 같습니다.

- 한국어 calibration prompt 512개 기반 prefill/decode expert usage trace 기능 구현
- nothink prefill, think prefill, nothink decode, think/high decode routing trace 측정
- trace 기반 hot expert 및 bit-lift 추천 manifest 생성
- 현재 GGUF/runtime 구조에서 즉시 실행 가능한 layer-level Q4 lift GGUF 생성
- 생성 GGUF의 로딩, 실제 생성, routing coverage, 한국어 품질, prefill/decode 속도 검증

최종 생성 GGUF는 다음 파일입니다.

```text
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf
```

파일 크기는 다음과 같습니다.

```text
95,779,807,840 bytes
89.20 GiB
```

## 2. 데이터셋

사용한 한국어 calibration 데이터셋은 총 512개 prompt로 구성되었습니다. 주요 범주는 다음과 같습니다.

- 일상 대화 및 문자 작성
- 한국어 문체 변환
- 독해 요약 및 근거 추출
- 한국어 문화/지식 문항
- 코딩/보안 설명
- 정확 복사 및 형식 보존
- 장문 지시문 준수

대표적인 평가 목적은 다음과 같습니다.

- 한국어 자연성
- 공손체 및 문자 작성 품질
- 긴 문맥 지시문 준수
- 한글/자모/공백/표/JSON 형식 보존
- 코딩/보안 개념 설명 품질
- prefill과 decode 단계에서의 routed expert 사용 차이 확인

원본 데이터와 렌더링 결과는 다음 위치에 있습니다.

```text
/tmp/ds4-ko-cal/ko_cal_prompts.jsonl
/tmp/ds4-ko-cal/rendered/rendered_prompts_nothink.txt
/tmp/ds4-ko-cal/rendered-think/rendered_prompts_think.txt
```

## 3. 구현 변경

expert usage trace를 위해 다음 파일을 수정했습니다.

```text
/Users/kch3dri4n/llm_provide/ds4/ds4.c
/Users/kch3dri4n/llm_provide/ds4/ds4.h
/Users/kch3dri4n/llm_provide/ds4/ds4_cli.c
/Users/kch3dri4n/llm_provide/ds4/gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

추가된 주요 CLI 옵션은 다음과 같습니다.

```text
--expert-usage-out FILE
--expert-usage-decode-tokens N
```

기능 요약:

- routed MoE expert 선택 count 수집
- router weight sum 수집
- prefill prompt token 기준 trace
- greedy decode token 기준 trace
- CSV 형식으로 layer/expert별 count share 및 weight share 출력

## 4. Routing Trace 측정 결과

측정한 1차 trace 파일은 다음과 같습니다.

```text
/tmp/ds4-ko-cal/expert_usage.csv
/tmp/ds4-ko-cal/expert_usage_think.csv
/tmp/ds4-ko-cal/expert_usage_decode_nothink_128.csv
/tmp/ds4-ko-cal/expert_usage_decode_think_128.csv
```

decode trace 토큰 수는 다음과 같이 관측되었습니다.

```text
nothink decode: 52,315 decode tokens
think/high decode: 64,693 decode tokens
```

중요한 관찰은 다음과 같습니다.

- prefill hot expert만 보고 bit-lift 대상을 고르는 것은 위험합니다.
- prefill think/nothink의 top4는 전 layer에서 거의 동일했지만, decode nothink와 decode think/high의 top4는 37/43 layers에서 달라졌습니다.
- prefill과 decode 간 top expert 일치도가 낮아서, decode 품질 개선 후보는 prefill-only trace로는 많이 누락됩니다.

반복적으로 강하게 등장한 stable core expert는 다음입니다.

```text
L40:E037
L41:E184
L38:E021
L37:E025
L42:E032
```

## 5. Bit-Lift Manifest 산출물

추천 manifest는 다음 위치에 생성했습니다.

```text
/tmp/ds4-ko-cal/bitlift_recommendation/
```

주요 manifest는 다음과 같습니다.

```text
bitlift_mixed_top32_skip_existing4.json
bitlift_mixed_top24_skip_existing4.json
bitlift_think_priority_top32_skip_existing4.json
bitlift_nothink_priority_top32_skip_existing4.json
bitlift_union_top8_prefill_decode_nothink_decode_think_skip_existing4.json
bitlift_union_top12_prefill_decode_nothink_decode_think_skip_existing4.json
bitlift_stable_core_intersection_top8_all4modes_skip_existing4.json
global_stable_top100_all4modes.csv
```

추천 순위는 다음과 같습니다.

| 단계 | Manifest | Pairs | 예상 추가 용량 | 용도 |
|---|---|---:|---:|---|
| P0 | union_top8 | 479 | 약 +2.57 GiB | 변환 파이프라인 smoke test |
| P1 | mixed_top24 | 912 | 약 +4.90 GiB | 메모리 절약형 실전 후보 |
| P2 | mixed_top32 | 1216 | 약 +6.53 GiB | 균형형 한국어 품질 후보 |
| P2-think | think_priority_top32 | 1216 | 약 +6.53 GiB | thinking/high 위주 |
| P2-nothink | nothink_priority_top32 | 1216 | 약 +6.53 GiB | 일반 chat/nothink 위주 |

가장 추천하는 장기 목표는 다음입니다.

```text
DeepSeek-V4-Flash-JANGTQ-KR-Mixed32
```

다만 이 목표는 expert 단위 Q4 sidecar 또는 mixed tensor runtime 지원이 필요합니다.

## 6. GGUF 생성

원본 HF 모델은 다음에서 다운로드했습니다.

```text
deepseek-ai/DeepSeek-V4-Flash
```

다운로드 결과:

```text
46 safetensors
약 149 GiB
```

현재 quantizer와 runtime은 routed expert tensor 전체 단위 qtype을 사용합니다. 따라서 `mixed_top32`처럼 layer 내부의 특정 expert만 4bit로 올리는 GGUF는 현재 구조로 바로 표현할 수 없습니다.

그래서 1차 GGUF는 즉시 실행 가능한 layer 단위 Q4 lift로 생성했습니다.

Q4_K 적용 layer:

```text
L23
L25
L28
L34
L36
```

각 layer에서 다음 3개 routed tensor를 모두 Q4_K로 올렸습니다.

```text
ffn_gate_exps.weight
ffn_down_exps.weight
ffn_up_exps.weight
```

GGUF inspect 결과:

```text
file size: 89.20 GiB
tensor bytes described by GGUF: 89.20 GiB
logical parameters: 284.33 B

tensor types:
  f32        492 tensors, 0.00 GiB
  f16        359 tensors, 2.04 GiB
  q8_0       345 tensors, 6.15 GiB
  q2_k        38 tensors, 24.94 GiB
  q4_k        15 tensors, 16.88 GiB
  iq2_xxs     76 tensors, 39.19 GiB
  i32          3 tensors, 0.01 GiB
```

qtype 검증 결과:

```text
L23: gate=Q4_K down=Q4_K up=Q4_K
L25: gate=Q4_K down=Q4_K up=Q4_K
L28: gate=Q4_K down=Q4_K up=Q4_K
L34: gate=Q4_K down=Q4_K up=Q4_K
L36: gate=Q4_K down=Q4_K up=Q4_K
non_target_q4_count=0
bad_count=0
```

## 7. GGUF Smoke Test

짧은 한국어 생성 테스트를 실행했습니다.

Prompt:

```text
한국어로 한 문장만 답해 주세요. 오늘 작업 상태는?
```

Output:

```text
오늘 작업 상태는 순조롭게 진행 중입니다.
```

속도:

```text
prefill: 77.57 t/s
generation: 32.06 t/s
```

이 결과로 모델 로딩, Metal backend, prefill, decode, 한국어 출력이 모두 정상임을 확인했습니다.

## 8. 새 GGUF 기준 Routing Coverage

평가 결과 위치:

```text
/tmp/ds4-ko-cal/gguf_eval_worst5q4/
```

prefill trace:

```text
128 prompts
10,107 prompt tokens
2,607,606 routed expert observations
```

decode trace:

```text
32 prompts
1,932 prompt tokens
3,986 decode tokens
1,028,388 routed expert observations
```

manifest 후보 활성화율:

```text
prefill mixed_top32: 1204 / 1216 = 99.01%
decode  mixed_top32: 1204 / 1216 = 99.01%

prefill union_top8: 474 / 479 = 98.96%
decode  union_top8: 474 / 479 = 98.96%
```

Q4로 올린 layer의 expert 활성 폭:

| Mode | L23 | L25 | L28 | L34 | L36 |
|---|---:|---:|---:|---:|---:|
| prefill | 235/256 | 227/256 | 231/256 | 237/256 | 236/256 |
| decode | 227/256 | 233/256 | 236/256 | 219/256 | 218/256 |

stable core top5 활성화:

```text
L40:E037 prefill=9175 decode=3984
L41:E184 prefill=9134 decode=3976
L38:E021 prefill=9185 decode=3979
L37:E025 prefill=9146 decode=3974
L42:E032 prefill=9151 decode=3975
```

판단:

- 우리가 지목한 주요 expert들은 실제 prefill/decode에서 강하게 활성화됩니다.
- Q4 적용 layer들도 특정 expert 몇 개만 쓰이는 것이 아니라 layer당 218~237개 expert가 관측되었습니다.
- layer-level Q4 lift가 최소한 “죽은 layer를 올린 것”은 아니며, 실제 routing에서 자주 쓰이는 영역을 포함합니다.

## 9. 속도 비교

비교 대상:

```text
base: ds4flash.gguf
worst5q4: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf
```

짧은 decode 128 테스트:

| Model | Prefill t/s | Decode t/s |
|---|---:|---:|
| base | 71.34 | 31.76 |
| worst5q4 | 70.30 | 31.64 |

prefill-heavy 테스트:

| Model | Prefill t/s |
|---|---:|
| base | 327.49 |
| worst5q4 | 327.18 |

판단:

- Q4 layer 5개 추가로 인한 decode 속도 저하는 거의 없습니다.
- prefill 속도도 거의 동일합니다.
- M4 Max 128GB 환경에서 현재 89.20 GiB GGUF는 실행 가능하고 속도도 양호합니다.

## 10. 한국어 품질 점검

품질 샘플 결과:

```text
/tmp/ds4-ko-cal/gguf_eval_worst5q4/quality_samples.json
/tmp/ds4-ko-cal/gguf_eval_worst5q4/quality_deep_samples.json
```

exact-copy strict pass:

```text
6 / 7
```

통과 항목:

```text
single_line_copy
multiline_copy
json_like_copy
markdown_table_copy
whitespace_copy
tag_copy
```

실패 항목:

```text
jamo_copy
```

실패 내용:

```text
expected: 초성열: ...
got:      <초성열: ...
```

즉 자모 자체는 보존했지만, 원래 제거해야 했던 꺾쇠괄호 `<`를 같이 복사했습니다.

일상 대화 샘플은 공손하고 자연스러웠습니다. 예를 들어 조별과제 자료 요청 문항에서는 짧고 구체적인 요청으로 끝났습니다.

독해 요약 샘플도 원문에 없는 내용을 크게 추가하지 않고 3문장 요약 형식을 지켰습니다.

코딩/보안 설명은 개념 설명 품질은 양호했습니다. 다만 SQL prepared statement 예시에서 설명용 입력 문자열이 공격 예시처럼 보일 수 있어, 엄격한 방어-only 평가에서는 감점 여지가 있습니다.

장문 지시문 샘플은 전체 구조는 괜찮았지만, “표 사용 금지” 조건이 있는 상황에서 마지막 문장에 “표로 정리”를 언급했습니다. 따라서 long instruction compliance는 추가 개선/재평가가 필요합니다.

## 11. 현재 한계

가장 큰 한계는 GGUF/runtime 표현 단위입니다.

현재 구조:

```text
layer 전체 routed expert tensor 하나가 하나의 qtype을 가짐
```

원하는 최종 구조:

```text
같은 layer 안에서 특정 expert만 Q4, 나머지는 Q2/IQ2 유지
```

이 차이 때문에 `mixed_top32` manifest를 그대로 GGUF에 반영하지 못했습니다. 현재 만든 GGUF는 `mixed_top32`가 아니라, 실행 가능한 1차 smoke 모델인 `Worst5Q4`입니다.

true expert-level bit-lift를 하려면 다음이 필요합니다.

- Q4 sidecar expert tensor 포맷
- sidecar expert id mapping metadata
- runtime layer weight 구조 확장
- routing 결과에서 base tensor와 sidecar tensor를 나누는 split path
- Metal MoE kernel 또는 dispatch path 수정
- CPU/GPU validation test

## 12. 다음 계획

### P0: 현재 Worst5Q4 보존 및 비교 평가

현재 GGUF는 보존 가치가 있습니다. 속도 저하 없이 Q4 layer lift가 가능함을 확인했기 때문입니다.

해야 할 일:

- 기존 base vs Worst5Q4 held-out 100개 한국어 prompt 비교
- exact-copy 세트 확대
- 장문 지시문 준수율 측정
- 영어/중국어/control prompt degradation check

### P1: Expert-Level Mixed32 포맷 설계

다음 목표는 `mixed_top32`를 실제로 반영하는 것입니다.

권장 설계:

```text
base tensor: 기존 IQ2/Q2 full expert tensor 유지
sidecar tensor: selected expert만 Q4_K로 별도 저장
metadata: layer별 selected expert id list 저장
runtime: routing된 expert id가 sidecar에 있으면 Q4 path 사용
```

예상 sidecar tensor 이름:

```text
blk.N.ffn_gate_exps.bitlift_q4.weight
blk.N.ffn_down_exps.bitlift_q4.weight
blk.N.ffn_up_exps.bitlift_q4.weight
blk.N.ffn_exps.bitlift_q4.ids
```

### P2: KR-Mixed32 생성

목표 모델명:

```text
DeepSeek-V4-Flash-JANGTQ-KR-Mixed32
```

내용:

```text
기존 IQ2XXS-w2Q2K 기반 유지
나머지 38개 2bit layer에서 mixed score 상위 32 experts/layer만 Q4 sidecar 생성
L23/L25/L28/L34/L36은 별도 정책 적용
```

### P3: 최종 평가

최소 평가 항목:

- 한국어 held-out 100개
- exact-copy 64개 이상
- 장문 지시문 40개 이상
- 한국어 코딩/보안 설명 40개 이상
- 영어/중국어 degradation 50~100개
- prefill/decode speed
- stable expert routing coverage
- sidecar hit ratio

## 13. 최종 판단

현재 단계의 결론은 다음과 같습니다.

```text
1. 한국어 expert trace 기능은 정상 동작한다.
2. prefill-only 기준은 부족하고 decode trace가 반드시 필요하다.
3. trace 기반 mixed_top32 후보는 실제 routing에서 거의 전부 활성화된다.
4. 현재 runtime에서 바로 가능한 layer-level Q4 GGUF 생성은 성공했다.
5. 생성 GGUF는 로딩, 생성, 속도, routing coverage 모두 양호하다.
6. 한국어 일반 품질은 괜찮지만 exact-copy와 장문 지시문 준수는 추가 검증이 필요하다.
7. 진짜 목표인 expert-level KR-Mixed32는 GGUF/runtime sidecar 설계가 다음 핵심 작업이다.
```

현재 결과는 “감으로 한국어 expert를 찍는 단계”를 넘어서, trace 기반으로 lift 후보를 고르고 실제 GGUF 실험까지 연결한 1차 성공 상태입니다. 다음 작업의 핵심은 layer 단위 lift가 아니라 expert 단위 mixed lift를 런타임에 실제로 먹이는 것입니다.
