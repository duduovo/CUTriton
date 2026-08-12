# CUTriton FP16 Transformer FFN report

| Implementation | p50 (ms) | p95 (ms) |
|:--|--:|--:|
| CUTriton fused | 0.033728 | 0.038912 |
| CUTriton unfused | 0.038848 | 0.044704 |
| ONNX Runtime CUDA | 0.113728 | 0.143360 |
| PyTorch eager | 0.030816 | 0.031712 |

ORT subgraph speedup: `3.372x`. Gate: `passed`.
