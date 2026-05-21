# DS4 한국어 Q4 후속 평가 및 Sidecar 설계 계획서

작성일: 2026-05-18  
작업 디렉터리: `/Users/kch3dri4n/llm_provide/ds4`  
평가 결과 디렉터리: `/tmp/ds4-ko-cal/structured_eval_layerq4`

## 0. 이번 턴에서 수행한 작업

이번 작업은 사용자의 지시에 따라 로컬 디스크를 먼저 정리한 뒤, 현행 layer 단위 Q4 모델인 `Worst5Q4`를 기준으로 품질과 속도를 다시 측정했습니다. 또한 `Mixed32`를 그대로 밀어붙이기보다, 현재 런타임에서 실제로 표현 가능한 방식과 expert-level sidecar 설계 사이의 경계를 분리했습니다.

진행한 단계는 다음과 같습니다.

1. 로컬 디스크 정리 및 평가 공간 확보
2. held-out 한국어 100개로 `base` vs `Worst5Q4` 비교
3. 영어/중국어/control 퇴화 확인
4. exact-copy와 장문 지시문 확대 평가
5. expert-level sidecar GGUF/runtime 설계
6. `KR-Mixed32` 실제 생성 가능성 및 layer-Q4 대체 경로 판단

## 1. 로컬 디스크 관리 결과

초기 상태에서는 로컬 Data 볼륨 여유 공간이 약 34GiB 수준이라, 새 GGUF 생성과 재평가를 동시에 진행하기 어려웠습니다. `DeepSeek-V4-Flash` 원본 HF cache는 재다운로드 가능한 중간 산출물로 판단해 제거했고, 그 결과 로컬 여유 공간은 현재 약 182GiB 수준입니다.

현재 주요 디스크 상태는 다음과 같습니다.

| 위치 | 상태 |
|---|---:|
| `/` | 약 182GiB free |
| `/Volumes/Back_UP` | 약 73GiB free |
| `gguf/` | 약 170GiB |
| HF hub cache | 약 89GiB |

중요한 점은 `/Volumes/Back_UP`에 있는 safetensors는 `JANGTQ-K` 변환본이며, 현재 `gguf-tools/deepseek4-quantize`가 기대하는 원본 `deepseek-ai/DeepSeek-V4-Flash` safetensors가 아닙니다. 즉 새 full GGUF를 다시 만들려면 원본 HF safetensors를 다시 받아야 합니다.

## 2. 평가 대상 모델

| 모델 | 경로 | 크기 |
|---|---|---:|
| base | `/Users/kch3dri4n/llm_provide/ds4/ds4flash.gguf` | 80.76GiB |
| Worst5Q4 | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf` | 89.20GiB |

`Worst5Q4`는 기존 2bit 계열 GGUF에서 routed expert tensor layer `L23/L25/L28/L34/L36`만 full layer Q4_K로 올린 모델입니다. expert 단위 sidecar나 Mixed32가 실제 적용된 모델은 아직 아닙니다.

## 3. 평가 인프라 변경

기존에는 프롬프트마다 `./ds4` 프로세스를 새로 띄워 모델 mmap과 초기화를 반복했습니다. 이번에는 `ds4_cli.c`에 평가용 batch generation 모드를 추가했습니다.

추가된 옵션은 다음과 같습니다.

```text
--batch-prompts-tsv FILE
--batch-output-jsonl FILE
```

TSV 입력은 `id<TAB>max_tokens<TAB>system<TAB>prompt` 형식이며, 출력은 prompt별 JSONL입니다. 이 변경으로 모델을 한 번 로드한 뒤 여러 평가 프롬프트를 순차 처리할 수 있게 됐습니다.

평가 스크립트는 다음 파일입니다.

```text
/Users/kch3dri4n/llm_provide/ds4/tools/eval_ds4_project.py
```

## 4. 전체 평가 요약

| Suite | Model | N | Pass | Pass Rate | Avg Prefill t/s | Avg Decode t/s |
|---|---|---:|---:|---:|---:|---:|
| korean100 | base | 100 | 88 | 88.00% | 122.41 | 31.81 |
| korean100 | Worst5Q4 | 100 | 80 | 80.00% | 121.12 | 31.63 |
| control60 | base | 60 | 60 | 100.00% | 59.22 | 32.26 |
| control60 | Worst5Q4 | 60 | 60 | 100.00% | 58.77 | 31.95 |
| exact_long_extra | base | 30 | 10 | 33.33% | 136.82 | 31.62 |
| exact_long_extra | Worst5Q4 | 30 | 10 | 33.33% | 134.34 | 31.40 |

## 5. 세부 태스크별 결과

| Suite | Kind | Base Pass | Worst5Q4 Pass | Base Prefill | Worst Prefill | Base Decode | Worst Decode |
|---|---|---:|---:|---:|---:|---:|---:|
| control60 | chinese | 20/20 | 20/20 | 59.27 | 58.72 | 32.22 | 31.88 |
| control60 | control_exact | 20/20 | 20/20 | 62.62 | 62.03 | 32.35 | 31.97 |
| control60 | english | 20/20 | 20/20 | 55.76 | 55.57 | 32.20 | 32.00 |
| exact_long_extra | exact | 0/20 | 0/20 | 138.83 | 136.07 | 31.66 | 31.40 |
| exact_long_extra | long | 10/10 | 10/10 | 132.79 | 130.88 | 31.53 | 31.41 |
| korean100 | daily | 16/20 | 20/20 | 112.22 | 111.22 | 31.91 | 31.72 |
| korean100 | exact | 20/20 | 20/20 | 112.63 | 110.40 | 31.97 | 31.82 |
| korean100 | long | 12/20 | 0/20 | 160.72 | 160.27 | 31.55 | 31.43 |
| korean100 | summary | 20/20 | 20/20 | 133.92 | 132.10 | 31.62 | 31.40 |
| korean100 | tech | 20/20 | 20/20 | 92.57 | 91.60 | 31.98 | 31.78 |

핵심 해석은 다음과 같습니다.

- `daily`, `summary`, `tech`, `korean exact-copy`는 Worst5Q4가 base와 동등하거나 더 안정적입니다.
- `control60`은 base와 Worst5Q4 모두 60/60으로 통과했습니다. 영어/중국어/control 퇴화는 이번 측정에서는 보이지 않습니다.
- 확대 exact-copy의 한글 자모 케이스는 base와 Worst5Q4 모두 0/20입니다. 이 실패는 Q4 layer 추가 때문이라기보다 모델/프롬프트/토큰화 난도 자체가 원인입니다.
- `korean100`의 장문 지시문은 처음에는 base 12/20, Worst5Q4 0/20으로 나왔지만, 모든 장문 응답이 max token에 걸렸습니다.

## 6. 장문 512 재평가

장문 지시문은 기존 220/260 token 제한에서는 답변이 잘려 형식 평가가 왜곡될 가능성이 컸습니다. 그래서 long 계열 30개만 `max_tokens=512`로 재측정했습니다.

| Model | N | Pass | Pass Rate | Avg Prefill t/s | Avg Decode t/s | Maxed |
|---|---:|---:|---:|---:|---:|---:|
| base | 30 | 30 | 100.00% | 152.36 | 31.55 | 0 |
| Worst5Q4 | 30 | 25 | 83.33% | 151.18 | 31.31 | 0 |

이 재평가가 가장 중요합니다. 토큰 예산을 늘리면 base는 30/30으로 회복했고, Worst5Q4도 25/30까지 회복했습니다. 따라서 Worst5Q4의 장문 문제는 대부분 토큰 예산 영향이지만, 예산을 충분히 줘도 base 대비 약 5개 케이스에서 형식 안정성 손실이 남습니다.

## 7. 속도 판단

속도는 매우 안정적입니다.

- 한국어 held-out 100개 기준 decode: base 31.81 tok/s, Worst5Q4 31.63 tok/s
- control60 기준 decode: base 32.26 tok/s, Worst5Q4 31.95 tok/s
- 장문 512 기준 decode: base 31.55 tok/s, Worst5Q4 31.31 tok/s

Worst5Q4는 Q4 layer 5개를 포함하지만 decode 속도 손실은 약 0.5~1.0% 수준입니다. Prefill도 평균적으로 큰 차이는 없습니다.

## 8. Expert-Level Sidecar GGUF/Runtime 설계

현재 GGUF/runtime은 layer tensor 전체 qtype만 표현합니다. 즉 다음 세 tensor가 layer 단위로 통째로 `q2_k/iq2_xxs` 또는 `q4_k`가 됩니다.

```text
blk.N.ffn_gate_exps.weight
blk.N.ffn_down_exps.weight
blk.N.ffn_up_exps.weight
```

Mixed32처럼 layer 내부 특정 expert만 Q4로 올리려면 별도 sidecar 표현이 필요합니다.

### 8.1 GGUF tensor 제안

각 layer별로 선택된 expert만 담는 sidecar tensor를 추가합니다.

```text
blk.N.ffn_gate_exps.bitlift_q4.weight
blk.N.ffn_up_exps.bitlift_q4.weight
blk.N.ffn_down_exps.bitlift_q4.weight
blk.N.ffn_exps.bitlift_q4.ids
```

`ids`는 해당 sidecar tensor의 expert 순서를 나타내는 `i32` 배열입니다. 예를 들어 `ids=[37, 184, ...]`이면 sidecar weight tensor의 expert slot 0은 원래 expert 37을 의미합니다.

### 8.2 Metadata 제안

```text
quantize.bitlift.version = 1
quantize.bitlift.mode = expert_sidecar
quantize.bitlift.base_qtype = q2_k/iq2_xxs
quantize.bitlift.sidecar_qtype = q4_k
quantize.bitlift.manifest_sha256 = ...
```

구버전 런타임은 알 수 없는 tensor를 무시하므로, 기존 모델 호환성을 깨지 않는 방향으로 넣을 수 있습니다. 단, 파일 크기는 sidecar만큼 증가합니다.

### 8.3 Runtime routing 흐름

1. Router가 token별 top-k expert id를 선택합니다.
2. Runtime은 layer별 `expert_id -> sidecar_slot` lookup table을 확인합니다.
3. 선택 expert가 sidecar에 있으면 Q4_K sidecar slice를 사용합니다.
4. 없으면 기존 base low-bit expert tensor slice를 사용합니다.
5. token별 routed output은 기존 router weight를 곱해 누산합니다.

### 8.4 Metal kernel 변경 포인트

- Decode path: selected expert마다 base/sidecar를 분기해 matvec을 호출해야 합니다.
- Prefill batch path: active expert set을 base group과 sidecar group으로 나누고, 결과를 같은 output buffer에 누산해야 합니다.
- 누산 순서가 바뀌면 미세한 수치 차이가 생길 수 있으므로, 가능하면 selected expert 순서를 유지하고 backend별 재현성 테스트를 넣어야 합니다.
- `ids` 중복, 범위 초과, sidecar tensor shape 불일치, qtype 불일치에 대한 로드 시점 검증이 필요합니다.

### 8.5 Quantizer 변경 포인트

현재 quantizer는 tensor 전체를 생성합니다. sidecar 방식은 manifest를 입력받아 expert별 원본 HF slice만 Q4_K로 다시 양자화해야 합니다.

필요한 옵션 예시는 다음과 같습니다.

```text
--bitlift-manifest bitlift_mixed_top32_skip_existing4.json
--bitlift-sidecar-qtype q4_k
--bitlift-base-template MODEL.gguf
```

결과적으로 full layer Q4보다 용량 효율이 훨씬 좋고, `mixed_top32` 같은 trace 기반 후보를 실제로 반영할 수 있습니다.

## 9. KR-Mixed32 실제 생성 판단

이번 턴에서는 `KR-Mixed32`를 실제 GGUF로 생성하지 않았습니다. 이유는 기술적/운영상 제약이 명확합니다.

1. 현재 runtime은 expert 단위 qtype 선택을 지원하지 않습니다.
2. 현재 GGUF는 layer tensor 단위 qtype만 지원합니다.
3. 원본 `deepseek-ai/DeepSeek-V4-Flash` safetensors는 디스크 정리 과정에서 제거했습니다.
4. 외장 디스크의 `JANGTQ-K` safetensors는 `tq_packed/tq_norms/tq_bits` 구조라 현재 quantizer의 원본 HF 입력으로 사용할 수 없습니다.
5. 로컬 free 182GiB만으로는 원본 HF 약 149GiB와 새 GGUF 약 95~100GiB를 동시에 보관하기 어렵습니다.

따라서 사용자가 말한 “Mixed 말고 지금처럼”을 반영하면, 다음 실제 생성 후보는 expert-level Mixed32가 아니라 layer 단위 Q4 확장 모델입니다.

## 10. Layer-Q4 대체 생성안

현 구조에서 만들 수 있는 다음 모델은 예를 들어 다음 형태입니다.

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-HotLate5Q4-chat-v2-imatrix.gguf
```

후보 layer는 stable core가 몰린 late routing layer입니다.

```text
L37, L38, L40, L41, L42
```

다만 이 모델은 기존 Worst5Q4의 `L23/L25/L28/L34/L36`에 추가로 hot late 5개 layer를 full Q4로 올리는 방식이라, 파일 크기가 대략 97~99GiB까지 늘 가능성이 있습니다. Worst5Q4가 장문 512에서 이미 base 대비 5개 케이스 손실을 보인 점을 고려하면, 무작정 layer를 더 올리기 전에 장문 안정성을 다시 검증해야 합니다.

실제 생성에 필요한 선행 조건은 다음과 같습니다.

```text
1. 원본 HF safetensors 재다운로드
2. peak disk 확보: 원본 HF 약 149GiB + 출력 GGUF 약 98GiB + template GGUF
3. base GGUF 또는 기타 대형 파일 삭제/외장 이동에 대한 명시적 결정
4. deepseek4-quantize layer override로 full layer Q4 생성
5. 생성 후 동일 평가 재실행
```

예상 명령 골격은 다음과 같습니다.

```sh
gguf-tools/deepseek4-quantize   --hf /path/to/deepseek-ai/DeepSeek-V4-Flash   --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf   --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-HotLate5Q4-chat-v2-imatrix.gguf   --imatrix /path/to/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat   --tensor-type blk.37.ffn_gate_exps.weight=q4_k   --tensor-type blk.37.ffn_down_exps.weight=q4_k   --tensor-type blk.37.ffn_up_exps.weight=q4_k   --tensor-type blk.38.ffn_gate_exps.weight=q4_k   --tensor-type blk.38.ffn_down_exps.weight=q4_k   --tensor-type blk.38.ffn_up_exps.weight=q4_k   --tensor-type blk.40.ffn_gate_exps.weight=q4_k   --tensor-type blk.40.ffn_down_exps.weight=q4_k   --tensor-type blk.40.ffn_up_exps.weight=q4_k   --tensor-type blk.41.ffn_gate_exps.weight=q4_k   --tensor-type blk.41.ffn_down_exps.weight=q4_k   --tensor-type blk.41.ffn_up_exps.weight=q4_k   --tensor-type blk.42.ffn_gate_exps.weight=q4_k   --tensor-type blk.42.ffn_down_exps.weight=q4_k   --tensor-type blk.42.ffn_up_exps.weight=q4_k   --threads 8
```

## 11. 한계점

현재 결과의 한계는 분명합니다.

- 평가 스코어는 휴리스틱 기반입니다. 실제 한국어 품질을 완전히 대표하지 않습니다.
- exact-copy는 지나치게 엄격하며, 특히 자모 문자열은 base도 실패하므로 Worst5Q4만의 문제라고 보기 어렵습니다.
- 장문 평가는 token budget에 민감합니다. 220/260에서는 비교가 왜곡되고, 512에서는 훨씬 정상화됩니다.
- held-out 100개는 직접 구성한 synthetic set입니다. KMMLU, HAE-RAE, 실제 사용자 로그 기반 평가는 아직 아닙니다.
- routing trace는 “자주 쓰인 expert”를 찾은 것이지, “한국어에만 특화된 expert”를 분리한 것은 아닙니다.
- Mixed32 manifest는 후보 목록일 뿐이며, 현재 GGUF/runtime이 expert 단위 sidecar를 실행하지 못합니다.
- 새 full GGUF 생성은 disk peak와 원본 HF source availability가 병목입니다.

## 12. 결론

Worst5Q4는 속도 손실이 거의 없고, 한국어 단문/요약/보안 설명/exact-copy/control 다국어에서는 안정적입니다. 문제는 장문 형식 안정성인데, token budget을 512로 올리면 대부분 회복되지만 base 대비 약간의 손실은 남습니다.

따라서 지금 바로 더 많은 layer를 full Q4로 올리는 것은 조심해야 합니다. 다음 우선순위는 `KR-HotLate5Q4` 생성보다, expert-level sidecar runtime을 구현해 `mixed_top32`를 실제로 표현하는 것입니다. 다만 사용자가 “Mixed 말고 지금처럼”을 계속 원한다면, 다음 실험은 `HotLate5Q4`를 만들되 base GGUF 삭제 또는 원본 HF 재다운로드 위치를 먼저 결정해야 합니다.
