# DS4 KR Follow-up: KMMLU300, Think MAX30, Long-v2, Sidecar Runtime

작성일: 2026-05-19 KST

## 1. 결론

이번 재부팅 이후 `/tmp` 산출물이 사라져서 평가를 프로젝트 내부 `runs/20260519_followup/`로 옮기고 다시 완료했습니다. 결과는 꽤 분명합니다.

- 일반 chat/nothink 기본 후보는 여전히 `base` 또는 `LateStable5Q4`입니다.
- KMMLU 300에서는 `LateStable5Q4`가 214/300으로 1등이지만, `Worst5Q4` 213/300, `Layer10Q4` 212/300, `base` 209/300이라 차이는 작습니다.
- Think MAX 30에서는 `Layer10Q4`가 17/30으로 `base`와 `LateStable5Q4`의 10/30을 크게 앞섭니다.
- 장문 지시문 v2 nothink에서는 `base`가 42/60으로 가장 안정적이고, `LateStable5Q4` 39/60, `Layer10Q4` 27/60입니다.
- 속도는 모든 축에서 decode 약 31 tok/s 전후로 유지되어, 현재 layer 단위 Q4 후보들의 속도 퇴화는 실사용 판단을 뒤집을 정도가 아닙니다.

따라서 운영 정책은 다음처럼 가져가는 게 맞습니다.

```text
일반 chat/nothink: base 우선, KMMLU/일반 지식이 더 중요하면 LateStable5Q4도 후보
Think MAX 한국어: Layer10Q4 유지
Mixed32: 지금 당장 기본 후보로 올리지 말고 sidecar 런타임 완성 뒤 재평가
```

## 2. 산출물 위치

| 항목 | 경로 |
|---|---|
| 전체 run 디렉터리 | `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup` |
| KMMLU 300 | `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/kmmlu300` |
| Think MAX 30 | `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/thinkmax30` |
| 장문 지시문 v2 | `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/longv2` |
| sidecar plan | `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/sidecar_plan` |
| 본 보고서 | `/Users/kch3dri4n/llm_provide/ds4/reports/ds4_followup_kmmlu300_thinkmax30_sidecar_20260519.md` |

## 3. 모델 파일 상태

| 모델 | 경로 | 크기 |
|---|---|---:|
| base | `/Users/kch3dri4n/llm_provide/ds4/ds4flash.gguf` | 80.76 GiB |
| worst5q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf` | 89.20 GiB |
| latestable5q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-LateStable5Q4-chat-v2.gguf` | 89.20 GiB |
| layer10q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf` | 97.64 GiB |

## 4. KMMLU 300 결과

| model | correct | n | accuracy | invalid | prefill tok/s | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 209 | 300 | 69.67% | 0 | 153.50 | 31.46 |
| worst5q4 | 213 | 300 | 71.00% | 0 | 150.80 | 31.32 |
| latestable5q4 | 214 | 300 | 71.33% | 0 | 152.87 | 31.23 |
| layer10q4 | 212 | 300 | 70.67% | 1 | 150.82 | 31.19 |

해석:

- `LateStable5Q4`가 base 대비 +5문항입니다.
- `Layer10Q4`도 base 대비 +3문항이라 KMMLU 자체가 나빠졌다고 보기는 어렵습니다.
- 다만 `Layer10Q4`는 invalid prediction 1개가 있어 객관식 format 안정성은 `base/LateStable5Q4`보다 약간 낮습니다.
- KMMLU 300만 보면 `LateStable5Q4`가 가장 깔끔합니다.

## 5. Think MAX 30 결과

| model | pass | n | pass_rate | prefill tok/s | decode tok/s | avg generated tokens | korean subset | long subset | exact subset | control subset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10 | 30 | 33.33% | 150.12 | 31.71 | 304.57 | 3/10 | 3/8 | 0/6 | 4/6 |
| latestable5q4 | 10 | 30 | 33.33% | 149.71 | 31.32 | 295.37 | 3/10 | 3/8 | 0/6 | 4/6 |
| layer10q4 | 17 | 30 | 56.67% | 150.03 | 31.04 | 319.70 | 8/10 | 5/8 | 0/6 | 4/6 |

해석:

- `Layer10Q4`는 Think MAX에서 base/latestable 대비 +7개 pass입니다.
- 한국어 subset만 보면 `Layer10Q4` 8/10, base 3/10, latestable 3/10입니다. 이건 우연이라기보다 Layer10Q4가 thinking/high 계열에 더 맞는 신호입니다.
- long subset도 `Layer10Q4` 5/8로 base/latestable 3/8보다 좋습니다.
- exact subset은 세 모델 모두 0/6입니다. Think MAX에서 exact-copy는 아직 별도 프롬프트/디코딩 전략 문제가 큽니다.
- decode 속도는 `Layer10Q4`가 31.04 tok/s로 약간 낮지만, 품질 차이를 감안하면 Think MAX 후보로 유지할 만합니다.

## 6. 장문 지시문 v2 결과

| model | pass | n | pass_rate | avg_score | prefill tok/s | decode tok/s | avg generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 42 | 60 | 70.00% | 0.856 | 168.70 | 31.38 | 353.30 |
| latestable5q4 | 39 | 60 | 65.00% | 0.835 | 165.54 | 31.35 | 357.25 |
| layer10q4 | 27 | 60 | 45.00% | 0.816 | 164.53 | 31.18 | 364.33 |

세부 kind별 pass:

| model | basic_plan | risk_plan | term_explain | validation_plan |
|---|---:|---:|---:|---:|
| base | 15/15 | 6/15 | 15/15 | 6/15 |
| latestable5q4 | 15/15 | 6/15 | 12/15 | 6/15 |
| layer10q4 | 12/15 | 0/15 | 9/15 | 6/15 |

해석:

- 일반 nothink 장문은 `base`가 가장 안정적입니다.
- `LateStable5Q4`는 base보다 약간 낮지만 KMMLU가 좋아서 일반 지식형 chat 후보로는 남길 수 있습니다.
- `Layer10Q4`는 risk_plan 0/15로 명확한 약점이 있습니다. Think MAX에서는 강하지만 일반 nothink 장문 기본값으로 올리면 안 됩니다.
- 장문 v2는 기존 장문 평가보다 까다롭습니다. 제목 길이, bullet 개수, 단계 수, 표 금지, 위험/완화, 검증 필요 항목, 전문용어 괄호 설명을 개별 criterion으로 봅니다.

## 7. Expert-level Sidecar GGUF/runtime 진행 상황

이번에 runtime 쪽에는 sidecar-aware loader/validation 기반을 붙였습니다.

추가한 런타임 계약:

```text
blk.N.ffn_gate_exps.bitlift_q4.weight
blk.N.ffn_up_exps.bitlift_q4.weight
blk.N.ffn_down_exps.bitlift_q4.weight
blk.N.ffn_exps.bitlift_q4.ids
```

동작:

- layer별 optional sidecar tensor 4종을 탐지합니다.
- 4종 중 일부만 있으면 로드 시점에서 실패시킵니다.
- `ids`는 1D `i32`, expert id 범위 0..255, 중복 없음으로 검증합니다.
- gate/up/down sidecar tensor는 `Q4_K`이고 각각 `[4096, 2048, sidecar_count]`, `[4096, 2048, sidecar_count]`, `[2048, 4096, sidecar_count]` 형태로 검증합니다.
- 런타임에 `expert_id -> sidecar_slot` lookup을 layer별로 구축합니다.
- 기존 GGUF에서는 `./ds4 --inspect -m ds4flash.gguf` 기준 `bitlift sidecar: none`으로 정상 로드됨을 확인했습니다.

생성된 sidecar plan:

| plan | layers | expert slots | tensors to add | manifest sha256 |
|---|---:|---:|---:|---|
| mixed_top32 | 38 | 1216 | 152 | `f62e149254c11cf5ffeb3399eb560f1ec5e327e32c24715a827d9e125ee399b1` |
| think_priority_top32 | 38 | 1216 | 152 | `a1c86014ef8b35ed8bf3952f10f0d7a001887f5c661e41dba331e5cde194972d` |

중요한 한계:

- 이번 패치는 sidecar tensor를 안전하게 인식하고 검증하는 loader/runtime 기반입니다.
- 실제 Metal fused MoE dispatch가 sidecar Q4 slice를 선택해 계산하는 단계는 아직 남아 있습니다.
- 따라서 아직 `Mixed32 sidecar GGUF`를 넣으면 품질 평가에 쓰면 안 됩니다. 다음 패치는 Metal decode/prefill routed MoE에서 selected expert별 base/sidecar dispatch를 구현해야 합니다.

## 8. 성능 판단

- KMMLU decode: 31.19~31.46 tok/s
- Think MAX decode: 31.04~31.71 tok/s
- 장문 v2 decode: 31.18~31.38 tok/s

Q4 layer 변형 간 decode 차이는 작습니다. 품질 차이가 모델 선택을 결정하고, 속도는 현재 후보군에서 보조 지표입니다.

## 9. 최종 의사결정

```text
일반 chat/nothink 기본값: base
일반 지식/KMMLU 중시 후보: LateStable5Q4
Think MAX 한국어 후보: Layer10Q4
장문 nothink 기본값: Layer10Q4 제외
Mixed32: sidecar dispatch 구현 전까지 보류
```

## 10. 다음 작업

1. Metal decode MoE에서 selected expert별 sidecar slot 분기 구현
2. Metal prefill batch MoE에서 active expert를 base/sidecar 그룹으로 나누는 dispatch 구현
3. sidecar GGUF writer에서 plan JSON을 읽어 Q4_K sidecar tensor와 i32 ids tensor 생성
4. `think_priority_top32` sidecar를 먼저 생성해 Think MAX 30을 재평가
5. `mixed_top32` sidecar는 KMMLU/장문 v2까지 포함해 기본값 후보인지 별도 검증

## 11. 체크섬/재현성 메모

- KMMLU sample: `n=300`, `seed=20260519`
- Think MAX: `ctx=393216`, `preset=expanded30`, `seed=7`
- Long v2: `n=60`, `ctx=4096`, greedy/nothink
- 모든 결과는 재시작 후에도 이어서 쓸 수 있게 raw JSONL 기반 resume 형태로 저장했습니다.
