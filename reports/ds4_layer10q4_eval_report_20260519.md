# DS4 KR-Layer10Q4 생성 및 1차 평가 보고서

작성일: 2026-05-19  
작업 경로: `/Users/kch3dri4n/llm_provide/ds4`  
대상 모델: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf`

## 1. 결론 요약

이번 단계에서는 기존 `Worst5Q4`의 early Q4 layer 5개에 late stable layer 5개를 더해 총 10개 routed expert layer를 Q4로 올린 `KR-Layer10Q4` GGUF를 생성하고, KMMLU 샘플 100개, 한국어 held-out 100개, 영어/중국어/control 60개, exact-copy/장문 확대 세트, Think MAX 벤치, expert usage trace를 확인했습니다.

최종 판단은 다음과 같습니다.

- `KR-Layer10Q4`는 KMMLU 샘플 100개에서 `71/100`으로 `Worst5Q4`와 동률이며 base 대비 `+2`입니다.
- Think MAX 소규모 벤치에서는 `6/10`으로 현재 비교군 중 가장 좋고, 한국어 Think MAX subset은 `4/4`를 통과했습니다.
- 다만 일반 `nothink` 한국어 held-out 100개에서는 `83/100`으로 base `88/100`, `LateStable5Q4` `84/100`보다 낮습니다.
- 특히 한국어 장문 지시문 subset에서 `4/20`만 통과해 base `12/20`, `LateStable5Q4` `8/20`보다 약합니다.
- 영어/중국어/control 퇴화는 이번 측정에서는 보이지 않았습니다. 모든 모델이 `60/60`을 통과했습니다.
- prefill/decode 속도는 실사용 가능한 범위입니다. 다만 외장 디스크 cold start에서는 Metal residency가 약 188초 걸렸으므로 배포 위치는 신중하게 잡아야 합니다.

따라서 `KR-Layer10Q4`는 "Think MAX 한국어 후보"로는 보관할 가치가 있지만, "일반 한국어 기본 모델"로 바로 승격하기에는 장문 지시문 회귀가 큽니다.

## 2. 디스크 정리 및 저장소 배치

내부 프로젝트 디렉터리의 GGUF 파일들이 커져서 외장 디스크 저장소를 먼저 정리했습니다.

- `/Volumes/Timemachine`: 약 `263GiB` 여유가 있었지만 Time Machine 보호로 쓰기 제한이 있어 작업 대상에서 제외했습니다.
- `/Volumes/Back_UP`: 기존에는 약 `56GiB` 여유였고, redownload 가능한 Hugging Face cache를 제거해 작업 공간을 확보했습니다.
- 제거한 cache: `/Volumes/Back_UP/hf-cache-offload/JANGQ-AI/DeepSeek-V4-Flash-JANGTQ-K`
- `KR-Layer10Q4` 생성 후 `/Volumes/Back_UP` 여유 공간은 약 `38GiB`입니다.

GGUF 본체는 외장 디스크에 두고, 프로젝트 내부에는 symlink만 둔 상태입니다.

```text
실제 파일:
/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf

프로젝트 symlink:
/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf
```

GGUF 크기:

```text
104,839,504,480 bytes
약 97.65 GiB
Finder 표시 약 98G
```

## 3. 생성 방식

이번 `KR-Layer10Q4`는 다음 구조입니다.

- 기존 `Worst5Q4` 유지: `L23`, `L25`, `L28`, `L34`, `L36` routed expert tensors Q4
- 추가 `LateStable5Q4`: `L37`, `L38`, `L40`, `L41`, `L42` routed expert tensors Q4
- 총 Q4 routed expert layer: 10개
- 각 target layer에서 `ffn_gate_exps.weight`, `ffn_down_exps.weight`, `ffn_up_exps.weight`를 Q4_K로 구성

생성은 `Worst5Q4` GGUF를 template으로 사용하고, late 5개 layer만 HF partial shard에서 다시 양자화하는 방식으로 수행했습니다. 이 방식은 이미 만들어진 early Q4 layer를 재생성하지 않고 복사하기 때문에 시간과 디스크 사용량을 줄입니다.

생성 명령의 핵심은 다음과 같습니다.

```bash
gguf-tools/deepseek4-quantize \
  --hf hf-partial/DeepSeek-V4-Flash-late5 \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Worst5Q4-chat-v2-imatrix.gguf \
  --out /Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-KR-Layer10Q4-chat-v2.gguf \
  --copy-unchanged \
  --tensor-type blk.37/38/40/41/42 ffn_{gate,down,up}_exps.weight=q4_k \
  --threads 8
```

## 4. Tensor Type 검증

간이 GGUF parser로 tensor type을 확인했습니다.

| layer | gate | down | up | 판정 |
|---:|---|---|---|---|
| L23 | q4_K | q4_K | q4_K | OK |
| L25 | q4_K | q4_K | q4_K | OK |
| L28 | q4_K | q4_K | q4_K | OK |
| L34 | q4_K | q4_K | q4_K | OK |
| L36 | q4_K | q4_K | q4_K | OK |
| L37 | q4_K | q4_K | q4_K | OK |
| L38 | q4_K | q4_K | q4_K | OK |
| L40 | q4_K | q4_K | q4_K | OK |
| L41 | q4_K | q4_K | q4_K | OK |
| L42 | q4_K | q4_K | q4_K | OK |

대조용 low-bit layer도 확인했습니다.

| layer | gate | down | up |
|---:|---|---|---|
| L24 | iq2_xxs | q2_K | iq2_xxs |
| L27 | iq2_xxs | q2_K | iq2_xxs |
| L39 | iq2_xxs | q2_K | iq2_xxs |

즉 의도한 10개 layer만 Q4로 올라갔고, 주변 layer가 실수로 바뀐 흔적은 없습니다.

## 5. 기본 구동 및 속도

Smoke test prompt:

```text
한국어로 한 문장으로 인사하세요.
```

출력:

```text
안녕하세요, 무엇을 도와드릴까요?
```

속도:

```text
prefill: 68.21 t/s
generation: 32.46 t/s
```

주의할 점은 cold start입니다. 외장 HFS 볼륨에서 처음 Metal residency를 만들 때 약 `188,162 ms`, 즉 약 `188초`가 걸렸습니다. 이후 같은 세션에서는 file cache 덕분에 로드 시간이 크게 줄었습니다. 따라서 이 GGUF를 실사용하려면 내부 SSD 배치 또는 warm cache 운용이 필요합니다.

## 6. KMMLU 샘플 100개 평가

평가 데이터는 `HAERAE-HUB/KMMLU`에서 seed `20260519`로 100개를 샘플링한 세트입니다.  
소스: https://huggingface.co/datasets/HAERAE-HUB/KMMLU

| model | correct | accuracy | invalid | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| base | 69/100 | 0.69 | 0 | 152.91 | 31.13 |
| Worst5Q4 | 71/100 | 0.71 | 0 | 149.86 | 30.84 |
| LateStable5Q4 | 69/100 | 0.69 | 0 | 151.30 | 31.00 |
| Layer10Q4 | 71/100 | 0.71 | 1 | 147.85 | 31.40 |

Pairwise 관점에서는 `Layer10Q4`가 base 대비 `+2` net입니다.

```text
Layer10Q4 vs base:
gains 8
losses 6
net +2
```

KMMLU만 보면 `Layer10Q4`는 `Worst5Q4`와 동률입니다. 다만 `Layer10Q4`에서 invalid prediction이 1개 나온 점은 추가 샘플에서 다시 확인해야 합니다.

## 7. 한국어 held-out 100개 및 control 평가

전체 suite 결과입니다.

| suite | model | pass | pass rate | avg prefill t/s | avg decode t/s |
|---|---|---:|---:|---:|---:|
| korean100 | base | 88/100 | 0.88 | 122.22 | 32.58 |
| korean100 | Worst5Q4 | 80/100 | 0.80 | 121.41 | 31.66 |
| korean100 | LateStable5Q4 | 84/100 | 0.84 | 123.33 | 31.41 |
| korean100 | Layer10Q4 | 83/100 | 0.83 | 119.81 | 31.88 |
| control60 | base | 60/60 | 1.00 | 58.86 | 32.15 |
| control60 | Worst5Q4 | 60/60 | 1.00 | 58.81 | 32.05 |
| control60 | LateStable5Q4 | 60/60 | 1.00 | 59.46 | 31.79 |
| control60 | Layer10Q4 | 60/60 | 1.00 | 58.17 | 32.36 |
| exact_long_extra | base | 10/30 | 0.33 | 135.98 | 31.64 |
| exact_long_extra | Worst5Q4 | 10/30 | 0.33 | 135.66 | 31.32 |
| exact_long_extra | LateStable5Q4 | 10/30 | 0.33 | 137.18 | 31.24 |
| exact_long_extra | Layer10Q4 | 10/30 | 0.33 | 132.56 | 31.84 |

`korean100`의 kind별 결과입니다.

| kind | base | Worst5Q4 | LateStable5Q4 | Layer10Q4 |
|---|---:|---:|---:|---:|
| daily | 16/20 | 20/20 | 16/20 | 20/20 |
| exact | 20/20 | 20/20 | 20/20 | 19/20 |
| long | 12/20 | 0/20 | 8/20 | 4/20 |
| summary | 20/20 | 20/20 | 20/20 | 20/20 |
| tech | 20/20 | 20/20 | 20/20 | 20/20 |

`control60`의 kind별 결과입니다.

| kind | base | Worst5Q4 | LateStable5Q4 | Layer10Q4 |
|---|---:|---:|---:|---:|
| chinese | 20/20 | 20/20 | 20/20 | 20/20 |
| control_exact | 20/20 | 20/20 | 20/20 | 20/20 |
| english | 20/20 | 20/20 | 20/20 | 20/20 |

`exact_long_extra`의 kind별 결과입니다.

| kind | base | Worst5Q4 | LateStable5Q4 | Layer10Q4 |
|---|---:|---:|---:|---:|
| exact | 0/20 | 0/20 | 0/20 | 0/20 |
| long | 10/10 | 10/10 | 10/10 | 10/10 |

해석:

- 영어/중국어/control 퇴화는 발견되지 않았습니다.
- `Layer10Q4`는 일상 대화형 한국어에서는 강합니다. `daily 20/20`입니다.
- 그러나 장문 지시문에서 약합니다. `long 4/20`이라 base와 LateStable보다 낮습니다.
- `Worst5Q4`가 `long 0/20`으로 가장 크게 망가졌고, `Layer10Q4`는 그보다는 낫지만 아직 충분하지 않습니다.
- 장문 지시문 회귀는 layer-level Q4 확대가 단순히 품질을 올리는 방향으로만 작동하지 않는다는 신호입니다.

## 8. Think MAX 벤치

`--ctx 393216` 조건에서 `Layer10Q4`만 새로 측정했고, 이전 Think MAX 결과와 비교했습니다.

| model | pass | pass rate | avg prefill t/s | avg decode t/s | avg gen tokens |
|---|---:|---:|---:|---:|---:|
| base | 4/10 | 0.40 | 기존 측정 | 기존 측정 | 기존 측정 |
| Worst5Q4 | 4/10 | 0.40 | 기존 측정 | 기존 측정 | 기존 측정 |
| LateStable5Q4 | 2/10 | 0.20 | 기존 측정 | 기존 측정 | 기존 측정 |
| Layer10Q4 | 6/10 | 0.60 | 131.51 | 32.03 | 268.10 |

`Layer10Q4` 세부 결과:

| suite | pass |
|---|---:|
| korean | 4/4 |
| control | 2/3 |
| exact | 0/2 |
| long | 0/1 |

해석:

- Think MAX 조건에서는 `Layer10Q4`가 현재 후보 중 가장 좋습니다.
- 특히 한국어 Think MAX subset은 `4/4`를 통과했습니다.
- 하지만 exact-copy와 long은 여전히 약합니다.
- Think MAX용 모델로는 추가 검증할 가치가 있지만, exact/long 회귀를 해결하지 못한 상태입니다.

## 9. Expert Usage Trace

`Layer10Q4`에 대해 nothink decode 64, think decode 64 조건으로 10개 prompt씩 expert usage trace를 기록했습니다.

Trace 요약:

```text
nothink:
prompts=10
prompt_tokens=15310
decode_tokens=242
routes=62436

think:
prompts=10
prompt_tokens=15539
decode_tokens=208
routes=53664
```

Q4 target layer별 활성 expert 수입니다.

| layer | nothink active | think active |
|---:|---:|---:|
| L23 | 176/256 | 172/256 |
| L25 | 172/256 | 175/256 |
| L28 | 151/256 | 167/256 |
| L34 | 150/256 | 177/256 |
| L36 | 147/256 | 168/256 |
| L37 | 167/256 | 174/256 |
| L38 | 167/256 | 174/256 |
| L40 | 139/256 | 155/256 |
| L41 | 166/256 | 190/256 |
| L42 | 170/256 | 182/256 |

각 target layer의 route 수:

```text
nothink: 각 target layer 1452 routes
think:   각 target layer 1248 routes
```

즉 "우리가 Q4로 올린 layer들이 실제로 사용되는가?"라는 질문에 대해서는 yes입니다. 10 prompt만으로도 각 Q4 layer에서 139개에서 190개의 expert가 활성화되었습니다.

다만 "이전에 안정 후보로 찍은 특정 expert가 항상 활성화되는가?"라는 질문에는 no입니다. 예를 들어 stable core였던 `L40:E037`, `L41:E184`, `L38:E021`, `L37:E025`, `L42:E032`는 이번 10 prompt trace에서 일부만 약하게 나타나거나 아예 나타나지 않는 경우가 있었습니다.

이것은 두 가지를 의미합니다.

- layer-level Q4는 넓은 route coverage를 제공하므로 runtime에서 확실히 사용됩니다.
- 하지만 한국어 품질을 정밀하게 올리려면 layer 전체 Q4보다 expert-level sidecar가 더 낫습니다. 특정 hot expert만 올리는 설계가 더 직접적입니다.

## 10. 속도 및 실사용성

속도 측면에서는 큰 문제는 없습니다.

- KMMLU decode 평균: `31.40 t/s`
- structured eval decode 평균: `31.88-32.36 t/s`
- Think MAX decode 평균: `32.03 t/s`
- prefill은 suite에 따라 다르지만, `Layer10Q4`가 base보다 약간 낮은 경향이 있습니다.

다만 외장 디스크 cold start는 실사용에서 눈에 띄는 문제입니다.

- 첫 Metal residency: 약 `188초`
- 이후 warm cache: 크게 완화됨
- 배포 권장: 내부 SSD 또는 외장 SSD warm-cache 운용

## 11. 한계점

이번 실험의 한계는 명확합니다.

1. KMMLU는 100개 샘플만 사용했습니다. `+2` net은 방향성 신호로 볼 수 있지만 통계적으로 확정하기에는 부족합니다.
2. Think MAX 벤치도 10 prompt뿐입니다. `Layer10Q4`가 좋아 보이지만 더 큰 세트로 확인해야 합니다.
3. `korean100`의 pass 기준은 자동 휴리스틱입니다. 실제 선호도, 자연스러움, 논리성 평가는 별도 judge 또는 사람 평가가 필요합니다.
4. exact-copy 확대 세트가 전 모델에서 `0/20`이라 모델 간 차이를 잘 분리하지 못했습니다. 난이도 구간을 나누어야 합니다.
5. layer-level Q4는 너무 넓은 처방입니다. 실제 hot expert만 올리는 설계보다 메모리 효율이 낮고, 장문 지시문 회귀를 일으킬 수 있습니다.
6. 이번에는 영어/중국어/control 60개만 봤습니다. 다국어 퇴화 검증은 최소 200개 이상으로 늘려야 합니다.
7. 외장 디스크 cold start가 커서, 벤치 속도와 실제 첫 실행 체감이 다릅니다.

## 12. 다음 권장 진행

현재 기준으로는 다음 순서가 가장 합리적입니다.

1. `Layer10Q4`는 Think MAX 후보로 보관합니다.
2. 일반 한국어 기본 후보는 `LateStable5Q4`와 base를 계속 기준선으로 둡니다.
3. `Layer10Q4`를 기본값으로 승격하지 않습니다. 장문 지시문 `4/20`이 너무 약합니다.
4. KMMLU를 `3 x 100` 또는 `1 x 300`으로 늘려서 `Worst5Q4`와 `Layer10Q4`의 동률이 재현되는지 확인합니다.
5. Think MAX 한국어 prompt를 30개 이상으로 늘려 `Layer10Q4`의 `4/4` 신호가 유지되는지 확인합니다.
6. exact-copy는 쉬움/중간/어려움으로 나누어 새 benchmark를 만듭니다.
7. 다음 큰 구현은 expert-level sidecar GGUF/runtime입니다. layer 전체 Q4보다 hot expert만 올리는 쪽이 메모리 대비 효과가 더 좋을 가능성이 큽니다.

## 13. 최종 판정

`KR-Layer10Q4`는 생성 자체는 성공했고, target Q4 layer도 모두 의도대로 반영되었습니다. Expert usage trace에서도 Q4 layer들이 실제 decode와 think 경로에서 폭넓게 활성화되는 것을 확인했습니다.

성능 면에서는 KMMLU와 Think MAX 한국어에서 좋은 신호가 있지만, 일반 한국어 장문 지시문에서 회귀가 큽니다. 따라서 지금 당장 기본 모델로는 `Layer10Q4`보다 더 섬세한 expert-level 접근이 필요합니다.

현재 추천 운영안:

```text
일반 nothink/chat: base 또는 LateStable5Q4 기준 유지
Think MAX 한국어 실험: Layer10Q4 후보 유지
다음 구현: expert-level sidecar GGUF/runtime
다음 평가: KMMLU 300개 + Think MAX 30개 + 장문 지시문 재설계
```

