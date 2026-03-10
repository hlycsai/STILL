# STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs

<h5 align="center">
🚀 Welcome to the repo of STILL! 


This repo contains the official code for STILL.

If you like our works, please support us with your stars⭐!

[![arXiv](https://img.shields.io/badge/Arxiv-2602.02180-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.02180)
</h5>


## Introduction

### Motivation

Large Language Models (LLMs) suffer from quadratic computational complexity ($\mathcal{O}(N^2)$) in standard softmax attention, which limits scalability to long sequences. Existing intra-layer hybrid attention methods for LLM linearization face two critical challenges:

1. **Position-based token routing**: Sliding-window-based token selection fails to capture token-specific global importance, discarding crucial non-local tokens from high-fidelity softmax attention (SA).
2. **Norm distortion in linear attention**: Learnable feature maps in linear attention (LA) distort pretrained feature magnitudes, breaking the "norm-aware" property of pretrained LLMs and degrading performance.

To address these limitations, we propose STILL, an intra-layer hybrid linearization framework that achieves linear complexity while preserving the expressive power of pretrained LLMs.

### Method

STILL introduces three core innovations to enable efficient and effective LLM linearization:

#### 1. Self-Saliency Score for Content-Aware Token Selection

We design a **Self-Saliency Score** with strong local–global consistency to estimate token importance using only sliding-window computation. This score quantifies how much a token relies on its self-attention term, enabling reliable selection of salient tokens for SA (high-fidelity modeling) while routing the rest to LA (efficient summarization) — replacing heuristic position-based routing.

Formally, the Self-Saliency Score is computed by comparing sliding-window attention distributions with/without the diagonal (self-attention) term:

$$
Scoret_t = \sum_{j∈Wt} a^{diag}_{t,j} \log \frac{a^{diag}_{t,j} + ϵ}{a^{nodiag}_{t,j} + ϵ}
$$

where $W_t$ is the local window index set, $a^{diag}$ and $a^{nodiag}$ are attention distributions with/without self-attention term, and $ϵ$ is a small constant for numerical stability.

#### 2. Norm-Preserved Feature Map (NP-Map)

To preserve pretrained norm statistics, we propose **NP-Map** that decouples feature direction from magnitude and reinjects pretrained norms into linear attention feature maps:

$$
u = \frac{f(x)}{\|f(x)\|} \|x\|, \quad \phi_{NP}(x) = [\text{softmax}(u) ⊕ \text{softmax}(-u)]
$$

where $f(·)$ is the learnable MLP in feature maps. NP-Map ensures linear attention aligns with the pretrained model’s representational intensity while satisfying non-negativity constraints.

#### 3. Unified Training-Inference Architecture

We adopt **chunk-wise parallelization** and **delayed selection** to improve hardware efficiency:

- Delayed selection: Token routing is performed at chunk granularity (instead of per-token) during decoding, reducing overhead and improving parallelism.
- Chunk-wise parallel form: Sequences are split into chunks for parallel computation of saliency scores and hybrid attention, unifying training and inference logic while maintaining linear complexity.

<p align="center">
    <img src="figures/mainfig.png" width= "700">
    <br>
    <em>Figure 1: Overall architecture of the STILL framework</em>
</p>


### Results

STILL achieves state-of-the-art performance on both standard reasoning and long-context benchmarks, while delivering significant efficiency gains:

#### Key Performance Highlights

- **Reasoning Tasks**: Matches or surpasses the original pretrained LLM on commonsense/general reasoning tasks (PIQA, ARC, HellaSwag, Winogrande), with up to 10.5% gains on MMLU over prior linearized baselines.
- **Long-Context Tasks**: Recovers 86.2% of full-attention performance on long-context benchmarks (RULER, BABILong) — a regime where existing hybrid baselines largely fail.
- **Efficiency**: Cuts required training tokens from 1000+B (scratch training) to 0.04B; achieves 45% average memory reduction and 28% decoding speed-up for sequences over 8K tokens.

