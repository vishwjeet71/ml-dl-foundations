# Overview

DeepSeek-v3 is a **671B-parameter** model, but only **37B parameters** are active for each token during inference.
Instead of using the full model every time, it activates only the parts needed for the current computation, which makes the model far more efficient.

## Core Engineering Components

### 1. Multi-head Latent Attention (MLA)

MLA is designed to reduce the memory cost of attention during inference.

In standard attention mechanisms, the model stores large **Key-Value (KV) caches**, which consume a significant amount of memory. MLA compresses this KV cache into a lower-dimensional latent representation, reducing memory usage while still preserving important information.

### 2. DeepSeekMoE

DeepSeekMoE is the main computation engine of the model.

It uses a **Mixture of Experts (MoE)** architecture, where different experts handle different tokens. The model dynamically routes tokens to the most relevant experts instead of activating the entire network.

It also introduces **Auxiliary-Loss-Free Load Balancing**, which improves expert utilization without adding extra balancing losses during training.

### 3. DualPipe (Training Infrastructure)

DualPipe improves training efficiency across multiple GPUs.

Communication between GPUs, especially across nodes, is often slower than computation. DualPipe overlaps communication and computation so both happen simultaneously.

While one batch is being processed on the GPU, the next batch is transferred in the background. This reduces idle time and improves overall training throughput.

### 4. Multi-token Prediction (MTP)

Traditional language models predict one token at a time.

DeepSeek-v3 uses **Multi-token Prediction (MTP)**, where the model predicts multiple future tokens in a single step. This improves training efficiency and can also help accelerate inference.

# Strategy for Training a Huge Model Efficiently

### Fine-Grained FP8 Quantization

Models like DeepSeek-V3 contain billions of parameters. Training such large models requires massive infrastructure and is very expensive. To reduce training cost while maintaining stability, DeepSeek-V3 uses a custom FP8 mixed-precision framework.

Only the most computationally expensive matrix multiplications (GEMMs) are performed in FP8. More sensitive parts of the model, such as embeddings, attention, and normalization layers, remain in higher precision (BF16 or FP32) to ensure stable training.

#### The Problem with FP8 Precision

FP8 can represent only a limited range of numbers. During training, some values can become too large or too small, causing accuracy issues.

#### Example

**The Problem:** Imagine multiplying 450 × 3, which equals 1350. But your calculator can display only 3 digits. It may round 1350 to 999 or truncate it in some way. If this happens repeatedly across thousands of calculations, the final result can become very inaccurate.

**The Solution:** The multiplications are done using the fast and efficient FP8 Tensor Cores. However, the results are immediately transferred to high-precision FP32 registers in the CUDA Cores, where they are accumulated accurately. This provides the speed benefits of FP8 while preserving numerical accuracy.



## Goal of this repository

The goal of this repository is to understand and implement the core architectural components of DeepSeek-v3.

# overview of architecture
```
Input Tokens
      │
      ▼
Token Embedding Layer
      │
      ▼
Transformer Blocks (Stacked)
 ┌─────────────────────────────┐
 │  RMSNorm                    │
 │      ↓                      │
 │  MLA Attention              │
 │      ↓                      │
 │  Residual Connection        │
 │      ↓                      │
 │  RMSNorm                    │
 │      ↓                      │
 │  DeepSeek MoE               │
 │      ↓                      │
 │  Residual Connection        │
 └─────────────────────────────┘
      │
      ▼
Final RMSNorm
      │
      ▼
Linear Output Head
      │
      ▼
Token Logits / Predictions
```

### MLA Attention (Internal Structure)
```
Input Hidden State
        │
        ├── Query Projection (Wq)
        │
        └── KV Compression (W_dkv)
                    │
                    ▼
             Latent Cache
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Key Reconstruction     Value Reconstruction
        (W_uk)                 (W_uv)
```
then 

```
Q × Kᵀ
   │
Softmax Attention
   │
Attention × V
   │
Output Projection
```

### DeepSeek MoE (Internal Structure)
```
Input Token
     │
     ├── Shared Expert
     │
     └── Router
            │
            ▼
      Select Top-K Experts
            │
            ▼
      Expert Processing
            │
            ▼
 Weighted Expert Combination
            │
            ▼
 Final MoE Output
```

### Transformer block: Output Head (According to code)
``` 
Final Hidden State
       │
   RMSNorm
       │
Linear Layer
       │
Vocabulary Logits
```
### MTP Module (Multi-Token Prediction)
```
Main Hidden State
        │
        ├──────┐
        │      │
        ▼      ▼
Hidden State + Next Token Embedding
                │
             Concatenate
                │
             Projection
                │
         Transformer Block
                │
           Output Head
                │
     Future Token Prediction
```
