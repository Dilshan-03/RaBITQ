# Adaptive Bit-Width Vector Quantization (ABV-Quant) Benchmark Report

### Benchmark Configuration
- **Dataset**: `synthetic` (Dimension: 256)
- **Base Vectors**: 10,000 | **Query Vectors**: 500
- **Target Bits/Dimension**: 2.0 | **Importance Metric**: `variance`
- **Top-K**: 10

### Performance Results

| Method           |   Recall@1 |   Recall@10 | MSE        | RelError   |   Bytes/Vec | Compression   |   Latency(ms) |    QPS |   Build(s) |
|------------------|------------|-------------|------------|------------|-------------|---------------|---------------|--------|------------|
| Flat FP32        |      1     |      1      | 9.9317e-14 | 0.0000     |        1024 | 1.0x          |         0.805 | 1241.6 |       0.01 |
| Binary Sign      |      0.01  |      0.0318 | N/A        | N/A        |          32 | 32.0x         |         2.56  |  390.6 |       0.01 |
| RaBitQ 1-bit     |      0.248 |      0.3958 | 8.3806e-02 | 0.0280     |          40 | 25.6x         |         2.066 |  484   |       0.14 |
| Uniform 2-bit    |      0.428 |      0.5358 | 2.1707e-02 | 0.0135     |          80 | 12.8x         |         5.717 |  174.9 |       0.2  |
| ABV-Quant (Ours) |      0.436 |      0.551  | 2.1991e-02 | 0.0137     |          88 | 11.6x         |         8.039 |  124.4 |       0.21 |

