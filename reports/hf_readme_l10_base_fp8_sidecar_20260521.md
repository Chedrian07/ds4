---
license: other
library_name: gguf
base_model: deepseek-ai/DeepSeek-V4-Flash-Base
tags:
- gguf
- deepseek-v4
- moe
- sidecar
- korean
- experimental
private: true
---

# DeepSeek-V4-Flash KR L10 Base-FP8 Q4 Sidecar GGUF

This private repository contains experimental Layer 10 routed-expert sidecar GGUF artifacts for local DS4 runtime experiments.

## Important Result

These artifacts are **not** the current recommended production variant.

The source-based sidecar pipeline worked: the files load, inspect correctly, route selected experts through sidecar tensors, and run structured/KMMLU/Think MAX/long-instruction evaluations without runtime failures. However, quality did not beat the existing operational choices.

Current recommendation from the local evaluation:

- General chat / nothink: keep `base` or `LateStable5Q4`.
- Think MAX / KMMLU Korean experiments: keep `Layer10Q4`.
- This Base-FP8 L10 sidecar set: keep as reproducible experimental artifacts and pipeline proof.

## Artifacts

| file | coverage | size | sha256 |
|---|---:|---:|---|
| `DeepSeek-V4-Flash-KR-ThinkTop64-L10-BaseFP8-Q4.sidecar.gguf` | L10 top64 routed experts | 864M | `fb3c755c658e39287424dfe85cc9bfe0fbcc8a4bfbaf775ed0263e4640de2f0e` |
| `DeepSeek-V4-Flash-KR-ThinkTop128-L10-BaseFP8-Q4.sidecar.gguf` | L10 top128 routed experts | 1.7G | `74bc248b0ff480f4a56066a73693b15e8dcfdb8e0d5608119ada730f58db888b` |
| `DeepSeek-V4-Flash-KR-Full256-L10-BaseFP8-Q4.sidecar.gguf` | L10 all 256 routed experts | 3.4G | `b22a14ab2bedf72ef31968168186f8c937c229c3e2bd38b9fb55346b903ee94a` |

## Source

The source tensors were read from `deepseek-ai/DeepSeek-V4-Flash-Base`, specifically Layer 10 expert tensors in `model-00012-of-00046.safetensors`.

The source format is official Base FP8:

- weight dtype: `F8_E4M3`
- block scale dtype: `F32`
- block shape: 128 x 128
- sidecar output quantization: `Q4_K`

This is not a full BF16-source sidecar.

## Local Evaluation Summary

| eval | base | Layer10Q4 | Top64 BaseFP8 | Top128 BaseFP8 | Full256 BaseFP8 |
|---|---:|---:|---:|---:|---:|
| Korean structured | 88/100 | not rerun here | 85/100 | 85/100 | 80/100 |
| Control structured | 60/60 | not rerun here | 60/60 | 60/60 | 60/60 |
| Exact/long extra | 10/30 | not rerun here | 10/30 | 10/30 | 10/30 |
| KMMLU 300 | 209/300 | 212/300 | 205/300 | 205/300 | 205/300 |
| Think MAX 30 | 10/30 | 17/30 | 14/30 | 12/30 | 10/30 |
| Long instruction v2 | 41/60 | 27/60 | 34/60 | 30/60 | 36/60 |

Decode speed stayed around 31 tok/s across the tested variants. The result is therefore a quality-selection issue, not a runtime-stability issue.

## How To Load

Use with a compatible local DS4 runtime that supports `--bitlift-sidecar`:

```bash
./ds4 --metal \
  -m ds4flash.gguf \
  --bitlift-sidecar DeepSeek-V4-Flash-KR-ThinkTop64-L10-BaseFP8-Q4.sidecar.gguf \
  --nothink -p "한국어로 짧게 답하세요."
```

For Think MAX:

```bash
./ds4 --metal \
  -m ds4flash.gguf \
  --bitlift-sidecar DeepSeek-V4-Flash-KR-ThinkTop64-L10-BaseFP8-Q4.sidecar.gguf \
  --think-max -c 393216 -p "CTF와 시스템 보안을 초보자에게 설명해 주세요."
```

## Included Reports

The repository also includes local reports and summary JSON files under `reports/` and `eval/` for reproducibility.
