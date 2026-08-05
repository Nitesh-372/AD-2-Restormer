# Implementation Plan: High-Performance Real-Time Vision Restormer Methodology

Upgrade the lightweight prototype in `AD2 reatformer` into the full high-performance Vision Transformer architecture specified in the RVITM methodology diagram for real-time weather-degraded image restoration.

## Architecture Overview & Methodology Mapping

The target architecture replaces the baseline `SimpleRestormer` with a 3-stage hierarchical U-Net Restormer equipped with:
1. **Shallow Feature Extractor**: $3 \times 3$ Conv converting RGB image ($H \times W \times 3$) to initial feature space ($H \times W \times C$).
2. **Learnable Degradation Token ($T_d$) & FiLM Conditioning**: Global degradation parameter $T_d \in \mathbb{R}^D$ that modulates feature channels across encoder stages via affine transformation $\text{FiLM}(F) = \gamma(T_d) \odot F + \beta(T_d)$.
3. **Hierarchical Restormer Encoder (Stages 1–3)**: Multi-scale Restormer blocks featuring **Multi-DConv Head Transposed Attention (MDTA)** and **Gated-DConv Feed-Forward Networks (GDFN)**.
4. **Degradation Consistency Regularization**: Feature consistency loss between encoder stage representations and $T_d$.
5. **Bottleneck Dual Attention**: Bottleneck combining Channel Attention and Spatial Attention.
6. **Hierarchical Restormer Decoder (Stages 1–3)**: PixelShuffle / Transposed Conv upsampling with skip connections from matching encoder stages.
7. **Global Residual Learning**: Predicts residual $\hat{R}$ such that $I_{\text{clean}} = I_{\text{degraded}} + \hat{R}$.
8. **Edge-Preserving Supervision**: Hybrid loss function: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + \lambda_{\text{edge}} \mathcal{L}_{\text{Sobel}} + \lambda_{\text{deg}} \mathcal{L}_{\text{consistency}}$.

---

## User Review Required

> [!IMPORTANT]
> - **Model Parameter Count & Real-Time Performance**: Adding 3-stage MDTA + GDFN Restormer blocks and FiLM layers will increase parameter count and GPU memory usage. We will tune channel dimensions ($C=48$) and number of blocks per stage (e.g., $[2, 3, 3, 4]$) to maintain high FPS for real-time performance.
> - **Multi-Task Supervision & Loss Weights**: Training requires edge supervision (Sobel gradient loss) and degradation consistency loss in addition to Charbonnier image loss.

---

## Proposed Code Changes

### Model Architecture Components

#### [MODIFY] [model.py](file:///d:/project/AD2%20reatformer/model.py)
Implement the complete methodology diagram:
- **`MDTA` (Multi-DConv Head Transposed Attention)**: Transposed self-attention across channel dimension $O(C^2)$ with depthwise convolutions.
- **`GDFN` (Gated-DConv Feed-Forward Network)**: Dual-gated depthwise feed-forward layer.
- **`RestormerBlock`**: Sequence of LayerNorm $\rightarrow$ MDTA $\rightarrow$ LayerNorm $\rightarrow$ GDFN with residual connections.
- **`FiLMConditioning`**: Predicts $\gamma(T_d)$ and $\beta(T_d)$ from learnable token $T_d$ to modulate feature maps at each encoder stage.
- **`DualAttentionBottleneck`**: Combines Channel Attention (Squeeze-and-Excitation / CAB) and Spatial Attention Module (SAM).
- **`DegradationConsistencyHead`**: Computes cosine similarity / MSE regularization loss between stage feature vectors and $T_d$.
- **`AD2Restormer`**: Assembles Shallow Conv, 3-Stage Encoder, Bottleneck, 3-Stage Decoder with Skip Connections, Residual Prediction, and Global Additive Skip Connection ($I + \hat{R}$).

### Data & Training Pipeline

#### [MODIFY] [dataset.py](file:///d:/project/AD2%20reatformer/dataset.py)
- Support augmentations (random horizontal flip, random rotation, random crop) for training robustness.
- Maintain support for deterministic evaluation sizing.

#### [MODIFY] [train.py](file:///d:/project/AD2%20reatformer/train.py)
- Implement **Edge-Preserving Loss** using Sobel operators to extract edge gradients from output and target.
- Implement **Charbonnier Loss** ($\sqrt{\|X - Y\|^2 + \epsilon^2}$) for smooth gradient optimization.
- Implement **Degradation Consistency Loss** to enforce $T_d$ alignment.
- Add cosine annealing learning rate scheduler and PSNR / SSIM evaluation metrics during validation.

#### [MODIFY] [test.py](file:///d:/project/AD2%20reatformer/test.py)
- Update inference script to load new model architecture, output clean restored images, and calculate FPS / latency for real-time benchmarking.

---

## Verification Plan

### Automated Verification & Benchmark Tests
- Run architecture unit test in `model.py` to check forward pass dimensions, parameter count, and tensor shapes.
- Verify real-time inference speed (FPS / Latency per $256 \times 256$ frame on available GPU/CPU).
- Train for test epoch on training set and verify loss convergence across Charbonnier, Edge, and Consistency losses.

### Performance & Metric Evaluation
- Measure Peak Signal-to-Noise Ratio (**PSNR**) and Structural Similarity Index (**SSIM**) on the validation dataset.
