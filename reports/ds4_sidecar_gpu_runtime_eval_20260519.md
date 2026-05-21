# DS4 Korean Bitlift Sidecar GPU Runtime Report

- Date: 2026-05-19 KST
- Workspace: `/Users/kch3dri4n/llm_provide/ds4`
- Runtime target: base DS4 GGUF + expert-level Q4 sidecar GGUF
- Main conclusion: sidecar runtime is now real GPU-resident dispatch, not just loader/planning. Quality is mixed: useful for long/Think MAX style Korean instruction following, not good as a default nothink KMMLU replacement.

## What Changed

Implemented the missing runtime piece: selected MoE routes can now be partitioned on Metal into base routes and sidecar Q4 routes without CPU readback.

- Added `kernel_dsv4_bitlift_partition_routes` in `metal/moe.metal`.
- Added `ds4_gpu_bitlift_partition_routes_tensor(...)` in `ds4_gpu.h` / `ds4_metal.m`.
- Wired decode and prefill routed-MoE helpers in `ds4.c` to call base MoE plus sidecar MoE and add both outputs.
- Kept `DS4_BITLIFT_TRACE_HITS=1` and added `DS4_BITLIFT_CPU_PARTITION=1` as diagnostic CPU/readback fallbacks.
- Extended local evaluation scripts with `thinktop32_sidecar` alias, which expands to `-m ds4flash.gguf --bitlift-sidecar gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf`.

## Sidecar Artifact

| item | value |
|---|---:|
| sidecar path | `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf` |
| symlink | `/Users/kch3dri4n/llm_provide/ds4/gguf/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf` |
| file size | 2.11 GiB |
| sidecar layers | L37, L38, L40, L41, L42 |
| expert slots | 160 total, 32 per layer |
| tensor triplets | 15 Q4 expert tensors + 5 id tensors |

Important limitation: this sidecar was built from the existing Layer10Q4 source model, so only late layers already present as Q4 in that source could be extracted: L37, L38, L40, L41, L42. It is not the full Mixed32 38-layer plan yet.

## Runtime Verification

- `make ds4 ds4-bench ds4-eval` completed successfully.
- `./ds4 --inspect` loads base + sidecar and reports `bitlift sidecar: layers=5 expert_slots=160 tensor_triplets=15 qtype=q4_k`.
- Nothink smoke with sidecar completed at roughly `prefill 63.94 t/s`, `generation 31.45 t/s`.
- Think MAX smoke with `--ctx 393216` completed at roughly `prefill 108.52 t/s`, `generation 33.65 t/s`.
- Diagnostic trace fallback confirms all five sidecar layers receive sidecar routes on a Korean plan prompt.

## Sidecar Route Activation Trace

`DS4_BITLIFT_TRACE_HITS=1` was run on a short Korean structured prompt. This forces the diagnostic CPU partition path, so it is not a speed benchmark; it is only route coverage instrumentation.

| layer | routed rows | rows with sidecar hit | sidecar routes | base routes | unique sidecar slots hit / 32 |
|---:|---:|---:|---:|---:|---:|
| L37 | 69 | 67 | 289 | 125 | 23 / 32 |
| L38 | 69 | 68 | 324 | 90 | 28 / 32 |
| L40 | 69 | 69 | 297 | 117 | 23 / 32 |
| L41 | 69 | 68 | 294 | 120 | 22 / 32 |
| L42 | 69 | 68 | 289 | 125 | 25 / 32 |

Interpretation: every sidecar layer is active, and the prompt exercised 22 to 28 of the 32 slots per layer. This proves the route split/remap path is live. It does not prove every one of the 160 experts is activated on every workload; broader coverage requires a larger trace set.

## KMMLU 300

Prompt mode: nothink, greedy, max 8 generated tokens. This is a regression signal, not a public benchmark number.

| model | correct / n | accuracy | invalid | avg prefill t/s | avg decode t/s |
|---|---:|---:|---:|---:|---:|
| `base` | 209 / 300 | 69.7% | 0 | 154.04 | 31.38 |
| `layer10q4` | 212 / 300 | 70.7% | 1 | 150.81 | 30.95 |
| `thinktop32_sidecar` | 203 / 300 | 67.7% | 6 | 151.42 | 30.34 |

Finding: `Layer10Q4` is best on this KMMLU sample. `thinktop32_sidecar` regresses by 2.0 points vs base and 3.0 points vs Layer10Q4, with more invalid one-token answers. Do not use this sidecar as the default nothink/KMMLU model.

## Think MAX 30

Prompt mode: `--think-max --ctx 393216`, expanded 30-prompt suite. This is closer to the sidecar candidate’s intended use.

| model | pass / n | pass rate | avg prefill t/s | avg decode t/s | avg generated tokens | Korean suite | long suite | control suite | exact suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 10 / 30 | 33.3% | 151.65 | 31.36 | 304.57 | 3/10 | 3/8 | 4/6 | 0/6 |
| `layer10q4` | 17 / 30 | 56.7% | 148.44 | 31.14 | 319.70 | 8/10 | 5/8 | 4/6 | 0/6 |
| `thinktop32_sidecar` | 17 / 30 | 56.7% | 146.50 | 30.77 | 311.10 | 7/10 | 6/8 | 4/6 | 0/6 |

Finding: sidecar ties Layer10Q4 overall at 17/30. It is slightly weaker on Korean short tasks, slightly stronger on long tasks, and about 1.2% slower in decode than Layer10Q4.

## Long Instruction v2

Prompt mode: nothink, 60 Korean long-format prompts. This suite stresses formatting, risk/validation sections, bullet counts, step labels, and polite Korean.

| model | pass / n | pass rate | avg score | avg prefill t/s | avg decode t/s | avg generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 42 / 60 | 70.0% | 0.852 | 166.64 | 31.53 | 348.73 |
| `layer10q4` | 27 / 60 | 45.0% | 0.820 | 164.97 | 31.28 | 369.45 |
| `thinktop32_sidecar` | 44 / 60 | 73.3% | 0.841 | 166.13 | 30.50 | 361.78 |

By kind:

| model | basic_plan | risk_plan | term_explain | validation_plan |
|---|---:|---:|---:|---:|
| `base` | 15/15 | 6/15 | 15/15 | 6/15 |
| `layer10q4` | 12/15 | 0/15 | 9/15 | 6/15 |
| `thinktop32_sidecar` | 13/15 | 6/15 | 13/15 | 12/15 |

Finding: sidecar is strongest here: 44/60 vs base 42/60 and Layer10Q4 27/60. The biggest useful signal is `validation_plan`: sidecar 12/15, base 6/15, Layer10Q4 6/15.

## Held-Out Korean 100 + Control 60 + Exact/Long Extra

Prompt mode: nothink, greedy, local synthetic regression suites.

| suite | model | pass / n | pass rate | avg prefill t/s | avg decode t/s |
|---|---|---:|---:|---:|---:|
| `korean100` | `base` | 88 / 100 | 88.0% | 124.51 | 31.64 |
| `korean100` | `layer10q4` | 84 / 100 | 84.0% | 122.80 | 31.32 |
| `korean100` | `thinktop32_sidecar` | 88 / 100 | 88.0% | 121.93 | 30.75 |
| `control60` | `base` | 60 / 60 | 100.0% | 59.76 | 32.02 |
| `control60` | `layer10q4` | 60 / 60 | 100.0% | 59.12 | 31.63 |
| `control60` | `thinktop32_sidecar` | 60 / 60 | 100.0% | 58.76 | 31.10 |
| `exact_long_extra` | `base` | 10 / 30 | 33.3% | 138.62 | 31.54 |
| `exact_long_extra` | `layer10q4` | 10 / 30 | 33.3% | 135.63 | 31.13 |
| `exact_long_extra` | `thinktop32_sidecar` | 10 / 30 | 33.3% | 135.45 | 30.60 |

Finding: sidecar matches base on Korean held-out 100 (88/100) and all models pass control60 (60/60), so no obvious English/Chinese/control degradation is visible here. Exact-copy remains weak and unchanged at 10/30 for all three in the expanded exact/long extra suite.

## Speed Summary

- Sidecar overhead is small but measurable. In the broad held-out/control suite, decode is `30.75 t/s` vs base `31.64 t/s` on Korean100, about 2.8% slower.
- In Think MAX 30, sidecar decode is `30.77 t/s` vs Layer10Q4 `31.14 t/s`, about 1.2% slower.
- In long instruction v2, sidecar decode is `30.50 t/s` vs base `31.53 t/s`, about 3.3% slower.
- Prefill stayed comparable because the GPU route partition avoids CPU readback in the default path.

## Decision

Recommended operating split:

- General chat / nothink: keep `base` or `LateStable5Q4` style baseline. Do not switch default to `thinktop32_sidecar`.
- KMMLU-like Korean multiple choice: `Layer10Q4` remains better in this local sample.
- Think MAX / long Korean planning: keep `thinktop32_sidecar` as a live experimental candidate; it ties Layer10Q4 on Think MAX 30 and beats base/Layer10Q4 on long instruction v2.
- Exact-copy: none of these variants solves the issue. This needs prompt/template work, decoding constraints, or targeted exact-copy calibration rather than this sidecar alone.

## Remaining Limits

- This is not full KR-Mixed32. The sidecar only contains five late layers because the source Q4 GGUF only had those Q4 expert slices available.
- The current GPU route partition computes both base and sidecar MoE for fixed six lanes with zero-weight inactive entries. That is simple and stable, but not the most efficient possible dispatch.
- Route activation trace is sampled. It proves the sidecar path is live, not exhaustive activation of every expert under all workloads.
- KMMLU and synthetic scoring are local regression signals. They are enough for engineering direction, not a final public quality claim.

## Files

- Runtime/eval output directory: `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_sidecar_gpu_runtime`
- This report: `/Users/kch3dri4n/llm_provide/ds4/reports/ds4_sidecar_gpu_runtime_eval_20260519.md`
- Patch snapshot: `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_sidecar_gpu_runtime/meta/sidecar_gpu_runtime.patch`
- Sidecar GGUF: `/Volumes/Back_UP/ds4-gguf-offload/DeepSeek-V4-Flash-KR-ThinkTop32-Layer10Q4.sidecar.gguf`
- KMMLU report: `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_sidecar_gpu_runtime/kmmlu300/REPORT.md`
- Long v2 report: `/Users/kch3dri4n/llm_provide/ds4/runs/20260519_sidecar_gpu_runtime/longv2/REPORT.md`

## Next Engineering Step

Build the real full Mixed32 sidecar from a source that actually has Q4 slices for all selected layers, or add an extraction path that creates Q4 sidecar tensors directly from base 2-bit experts. Then rerun the same four suites and compare against this five-layer ThinkTop32 sidecar baseline.
