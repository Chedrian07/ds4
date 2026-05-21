# DS4 Bit-Lift Sidecar Runtime / GGUF Writer 진행 보고서

작성 시각: 2026-05-19  
작업 위치: `/Users/kch3dri4n/llm_provide/ds4`

## 1. 이번 단계의 결론

이번 작업으로 기존의 “sidecar GGUF를 안전하게 읽고 검증하는 단계”를 넘어, 실제 Metal routed-MoE 계산 경로에서 sidecar Q4 expert slice를 선택해 계산하는 1차 런타임 경로를 구현했습니다. 또한 compact sidecar GGUF를 생성하는 writer를 추가했고, 실제 sidecar 파일을 외장 디스크에 생성해 base GGUF와 함께 로딩, inspect, nothink 생성, Think MAX 생성까지 스모크 테스트했습니다.

단, 현재 구현은 기능 검증과 correctness 우선의 1차 경로입니다. sidecar 대상 layer에서 router top-k 결과를 CPU로 읽어 base expert와 sidecar expert를 분리한 뒤 Metal routed-MoE를 두 번 호출하고 결과를 합산하므로, 최종 고속 경로로 보려면 GPU-native route partition/remap 커널 또는 bitlift-aware fused MoE 경로가 추가로 필요합니다.

## 2. 구현된 기능

### 2.1 CLI 및 engine 옵션

새 옵션을 추가했습니다.

```bash
./ds4 -m ds4flash.gguf --bitlift-sidecar path/to/sidecar.gguf ...
```

`--bitlift-sidecar`는 base GGUF와 별도의 compact Q4 routed-expert sidecar GGUF를 로드합니다. base 모델은 그대로 두고, sidecar에 들어 있는 layer/expert만 런타임에서 Q4 sidecar tensor로 대체 계산합니다.

### 2.2 sidecar GGUF contract

sidecar GGUF는 다음 tensor 이름을 사용합니다.

```text
blk.N.ffn_gate_exps.bitlift_q4.weight
blk.N.ffn_up_exps.bitlift_q4.weight
blk.N.ffn_down_exps.bitlift_q4.weight
blk.N.ffn_exps.bitlift_q4.ids
```

각 layer는 gate/up/down Q4_K compact tensor 3개와 expert id i32 tensor 1개를 가집니다. `ids` tensor는 compact slot index와 원래 expert id의 대응표입니다.

### 2.3 런타임 계산 경로

sidecar가 있는 layer에서는 다음 순서로 동작합니다.

1. router가 기존처럼 top-6 expert id와 weight를 계산합니다.
2. 런타임이 top-6 중 sidecar에 포함된 expert를 compact slot id로 remap합니다.
3. base expert는 기존 base GGUF tensor로 계산합니다.
4. sidecar expert는 sidecar GGUF의 Q4_K compact tensor로 계산합니다.
5. base 결과와 sidecar 결과를 더해 최종 routed-MoE 출력으로 사용합니다.

Metal routed-MoE 함수는 기존 256 experts 고정 구조에서 벗어나, compact sidecar tensor의 expert 수를 받을 수 있도록 `tensor_n_expert` 인자를 추가했습니다.

### 2.4 sidecar GGUF writer

추가된 writer:

```text
/Users/kch3dri4n/llm_provide/ds4/tools/write_bitlift_sidecar_gguf.py
```

주요 옵션:

```bash
tools/write_bitlift_sidecar_gguf.py \
  --source-q4 SOURCE_Q4.gguf \
  --plan PLAN.json \
  --out OUTPUT.sidecar.gguf \
  --allow-missing-source-q4 \
  --summary SUMMARY.json
```

이 writer는 기존 Q4 GGUF에서 필요한 expert slice만 복사해 compact sidecar GGUF를 생성합니다.

## 3. 생성된 sidecar 산출물

### 3.1 실제 sidecar 파일

외장 디스크에 생성된 파일:

```text
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf
```

로컬 symlink:

```text
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf
```

요약:

```text
layer_count: 5
layers: 37, 38, 40, 41, 42
expert_slot_count: 160
tensor_count: 20
file size: 약 2.1 GiB
payload size: 약 2.109 GiB
```

주의할 점은 이 sidecar가 “Layer10Q4 전체”가 아니라는 것입니다. 현재 사용한 `bitlift_think_priority_top32_skip_existing4` 계열 plan은 이미 4bit로 간주된 L23/L25/L28/L34/L36을 skip합니다. 따라서 Layer10Q4 source에서 실제 sidecar로 복사된 것은 추가 후보 layer인 L37/L38/L40/L41/L42의 top32 experts/layer입니다.

### 3.2 smoke sidecar

작은 검증용 sidecar:

```text
/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/sidecar_plan/bitlift_smoke_layer37_2.gguf
```

요약:

```text
layer_count: 1
expert_slot_count: 2
file size: 약 27 MiB
```

## 4. 검증 결과

### 4.1 빌드

전체 빌드 통과:

```bash
make
```

생성 확인:

```text
ds4
ds4-server
ds4-bench
ds4-eval
```

### 4.2 CLI 확인

`./ds4 --help`에서 다음 옵션이 확인되었습니다.

```text
--bitlift-sidecar FILE
    Optional GGUF containing compact Q4 routed-expert sidecar tensors.
```

### 4.3 inspect 확인

명령:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf \
  --inspect
```

결과:

```text
bitlift sidecar loaded: ... (layers=5)
Metal base model mapped: 82697.67 MiB
Metal sidecar mapped: 2160.02 MiB
bitlift sidecar: layers=5 expert_slots=160 tensor_triplets=15 qtype=q4_k
```

### 4.4 nothink base vs sidecar 스모크 속도

프롬프트:

```text
한국어로 한 문장만 답하세요. 오늘 상태는?
```

결과:

| 모델 | prefill | decode/generation | 비고 |
|---|---:|---:|---|
| base | 73.19 tok/s | 32.52 tok/s | 정상 생성 |
| base + ThinkTop32 sidecar | 63.52 tok/s | 29.36 tok/s | 정상 생성 |

sidecar는 현재 구현 기준 decode가 약 10% 느립니다. 원인은 sidecar layer에서 route id/weight를 CPU로 읽어 분기하는 1차 구현 방식입니다.

### 4.5 Think MAX 스모크

명령:

```bash
./ds4 -m ds4flash.gguf \
  --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf \
  --think-max --ctx 393216 \
  -p '한국어로 한 문장만 답하세요. 복잡도와 순환 알고리즘 보고서는 어떻게 시작하면 좋을까요?' \
  -n 16 --temp 0
```

결과:

```text
ctx=393216
context buffers: 6889.71 MiB
prefill: 91.22 tok/s
generation: 30.80 tok/s
```

Think MAX 경로는 로딩과 생성이 정상 동작했습니다. 다만 `-n 16`의 짧은 smoke 출력은 답변 품질을 판단하기에 부족하므로, Think MAX 품질 평가는 별도 30개 이상 프롬프트로 다시 보는 것이 맞습니다.

## 5. sidecar expert 활성화 확인

`DS4_BITLIFT_TRACE_HITS=1`을 추가해 실제 routed-MoE 경로에서 sidecar slot hit를 출력하도록 했습니다.

검증 로그:

```text
/Users/kch3dri4n/llm_provide/ds4/runs/20260519_followup/sidecar_plan/bitlift_trace_hits_smoke.log
```

요약:

| layer | routed rows | sidecar top-k hits | rows with sidecar hit | sidecar top-k share |
|---:|---:|---:|---:|---:|
| 37 | 28 | 107 | 26 | 0.6369 |
| 38 | 28 | 131 | 27 | 0.7798 |
| 40 | 28 | 111 | 28 | 0.6607 |
| 41 | 28 | 102 | 27 | 0.6071 |
| 42 | 28 | 113 | 27 | 0.6726 |

이 결과는 지목한 layer의 sidecar experts가 단순히 파일에 존재하는 것이 아니라 실제 라우팅 중 계산 경로에 들어갔음을 보여줍니다.

## 6. 현재 한계

1. prefill sidecar 경로는 correctness-first per-token 방식입니다. 긴 prompt에서는 sidecar 대상 layer에서 prefill throughput이 떨어질 수 있습니다.
2. decode도 현재는 sidecar layer마다 GPU 결과를 동기화하고 route를 CPU로 읽습니다. 이 때문에 sidecar를 붙이면 decode 속도가 base보다 낮아집니다.
3. 현재 sidecar는 5개 layer만 포함합니다. 기존 4bit layer까지 포함한 완전한 Layer10Q4 equivalent sidecar는 아닙니다.
4. sidecar writer는 Q4_K slice 복사에 초점을 맞춘 도구입니다. 향후 Q6_K/Q8_0 또는 다른 quant type 혼합 sidecar까지 확장하려면 tensor type별 expert slice copy 규칙을 더 넣어야 합니다.
5. 품질 평가는 아직 smoke 수준입니다. KMMLU 300, Think MAX 30, 장문 지시문 재설계 평가는 sidecar 고속화 전/후로 나누어 다시 실행하는 것이 좋습니다.

## 7. 다음 구현 우선순위

### P0: GPU-native route partition/remap

현재 CPU readback을 없애기 위해 router top-k 결과를 GPU에서 바로 base group과 sidecar group으로 나누는 kernel을 추가해야 합니다.

필요 산출물:

```text
base_selected/base_weights
side_selected/side_weights
base_count/side_count
```

### P1: sidecar-aware fused MoE

가능하면 base와 sidecar를 두 번 호출하고 add하는 구조가 아니라, routed-MoE kernel 내부에서 expert id별 tensor source를 선택하도록 만드는 것이 좋습니다.

### P2: full evaluation

고속화 후 아래 평가를 다시 실행해야 합니다.

```text
1. KMMLU 300
2. Think MAX 한국어 30
3. exact-copy 확대
4. 장문 지시문 v2/v3
5. 영어/중국어/control 퇴화 확인
```

## 8. 최종 판단

이번 단계는 “실사용 계산 경로가 전혀 없음”에서 “실제로 sidecar Q4 expert를 선택해 계산하고 결과를 합산하는 1차 런타임”으로 넘어간 상태입니다. 기능적으로는 의미 있는 성과가 있고, sidecar GGUF 포맷과 writer도 실제 파일 생성까지 검증되었습니다.

다만 성능 관점에서는 아직 완성형이 아닙니다. 다음 성과 지점은 CPU readback 없는 GPU-native route remap을 구현해서 sidecar decode 속도를 base 대비 0~3% 손실 수준으로 낮추는 것입니다.
