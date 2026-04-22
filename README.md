# Adaptive k-Allocation for k-MIP Attention in Graph Transformers

A compute-constrained empirical study of adaptive sparse attention schedules in GraphGPS.

## Overview

Graph transformers are powerful for modelling long-range interactions, but dense self-attention scales poorly on large graphs. k-Maximum Inner Product (k-MIP) attention addresses this by restricting each query to its top-k keys, reducing the cost of global attention.

This project investigates whether **adapting k during training** can improve performance over a fixed-k baseline. In particular, it compares four k-allocation strategies within the **GraphGPS + k-MIP** framework:

- **Fixed k**
- **Global cosine annealing**
- **Layerwise cosine annealing**
- **Hybrid layerwise + entropy-adaptive allocation**

The core question is whether adaptive sparsification reduces supervision starvation and improves predictive performance, especially on larger or more structurally diverse graph tasks.

## Key Results

The main finding is that **adaptive k-allocation is not universally better than a fixed-k baseline**.

- On **Peptides-func** and **Peptides-struct**, **fixed k = 15** performed best.
- On **PascalVOC-SP**, the **hybrid adaptive method** achieved the strongest result:
  - **50.35 ± 5.66 F1** (hybrid adaptive)
  - **47.23 ± 7.53 F1** (fixed k)

This suggests that adaptive sparsification may be more helpful on **larger, more complex graph tasks** than on smaller graph-level benchmarks.

## Methods

### Compared strategies

1. **Fixed k = 15**
2. **Global cosine annealing**
   - Uniform allocation across layers
   - **45 → 5** on Peptides
   - **30 → 10** on PascalVOC-SP
3. **Layerwise cosine annealing**
   - Same temporal schedule, with depth-dependent scaling
4. **Hybrid entropy-adaptive**
   - Layerwise cosine schedule
   - Additional per-query adjustment based on attention entropy

### Intuition

The project decomposes k-allocation into three dimensions:

- **Temporal adaptation**: reduce the attention budget over training
- **Layerwise adaptation**: allocate larger budgets to earlier layers and tighter budgets to deeper layers
- **Querywise adaptation**: assign more keys to higher-entropy queries and fewer keys to already concentrated ones

The motivation is to allow broader exploration earlier in training while preserving the efficiency benefits of sparse attention later.

## Experimental Setup

- **Framework**: GraphGPS + k-MIP
- **Datasets**:
  - Peptides-func
  - Peptides-struct
  - PascalVOC-SP
- **Model configuration**:
  - 8 layers
  - 4 attention heads
  - Hidden dimension: 52
  - GatedGCN local message passing
  - Two-layer output MLP
- **Training**:
  - AdamW optimizer
  - Cosine learning-rate decay
- **Compute setting**:
  - 3 random seeds
  - 50 epochs on Peptides benchmarks
  - 60 epochs on PascalVOC-SP
  - Compute-constrained study

## Results Summary

| Method | Peptides-func (AP ↑) | Peptides-struct (MAE ↓) | PascalVOC-SP (F1 ↑) |
|--------|----------------------:|-------------------------:|--------------------:|
| Fixed k = 15 | 63.01 ± 0.67 | 0.2652 ± 0.0039 | 47.23 ± 7.53 |
| Cosine annealing | 61.14 ± 1.67 | 0.2784 ± 0.0056 | 47.99 ± 0.97 |
| Layerwise cosine | 60.77 ± 0.74 | 0.2812 ± 0.0085 | 49.63 ± 3.13 |
| Hybrid adaptive | 61.45 ± 0.48 | 0.2834 ± 0.0042 | **50.35 ± 5.66** |

## Main Takeaways

- **Fixed k remained strongest** on the smaller peptide datasets.
- **Adaptive schedules helped most on PascalVOC-SP**, where the hybrid method performed best.
- **Entropy curves and layerwise k profiles confirmed** that the schedules behaved as intended, even when performance gains were limited.
- **Runtime and memory trade-offs were dataset-dependent**:
  - On peptide tasks, adaptive methods often stopped earlier but used more peak memory.
  - On PascalVOC-SP, adaptive methods cost more overall but produced better F1.

## Repository Structure

```text
.
├── README.md
├── paper/
│   └── Adaptive_k_Allocation_for_k_MIP_Attention.pdf
├── notebooks/
│   └── kmips_extension.ipynb
├── figures/
│   ├── entropy_curves.png
│   ├── layerwise_k_profiles.png
│   ├── runtime_memory.png
│   └── results_summary.png
└── requirements.txt
