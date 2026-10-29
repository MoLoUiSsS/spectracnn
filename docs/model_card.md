# Model card — SpectraCNN

## Overview

| Field | Value |
| --- | --- |
| Model | 1D convolutional neural network (PyTorch) |
| Task | Automatic modulation classification (AMC) from raw I/Q samples |
| Classes | BPSK, QPSK, 16-QAM, 64-QAM |
| Input | 2 × 1,024 unit-power I/Q frame (I and Q rows) |
| Owner | Hammadi Mohammed |
| Version | 0.1.0 |
| License | MIT |
| Last updated | 2026 |

## Intended use

- Portfolio and teaching demonstration of an end-to-end RF + deep-learning service.
- Benchmarking signal-processing pipelines, CNN architectures and per-SNR evaluation.
- A baseline for cognitive radio, spectrum monitoring and 5G/6G link-adaptation experiments.

## Out-of-scope use

- **Not** a production electronic-warfare, spectrum-enforcement or intercept system.
- **Not** validated on over-the-air captures, SDR hardware or real 5G waveforms.
- Must not be used to classify signals without authorization or for safety-critical
  decisions such as jamming detection or dynamic spectrum access enforcement.
- The synthetic channel omits multipath fading, IQ imbalance, sampling clock drift
  and hardware nonlinearities, so real-world accuracy will be lower.

## Data

`src/spectracnn/signals.py` generates balanced synthetic frames:

1. Uniform random bits are mapped to complex symbols for the chosen modulation.
2. Symbols are upsampled to 8 samples/symbol and filtered with a root-raised-cosine
   pulse (β = 0.35, span = 8 symbols).
3. Impairments are applied: carrier frequency offset (±1% of the sample rate),
   uniform phase rotation, integer timing offset (0–7 samples) and amplitude gain.
4. Circular complex AWGN is added at a uniformly sampled SNR in [−10, 20] dB.
5. Frames are cropped to 1,024 samples and normalized to unit average power.

Training uses 6,000 frames (1,500 per class), split 70/15/15 with stratification.
Evaluation uses fresh frames generated at each SNR from −10 to 20 dB in 2 dB steps,
100 frames per class per point. Generation is seeded and reproducible.

Real-world adaptation would require replacing the generator with IQ captures
(e.g. RadioML 2018.01A) and re-validating every claim in this card.

## Architecture

```
Input (2 × 1024)
  Conv1d(2 → 32, k=7) + BN + ReLU + MaxPool(2) + Dropout
  Conv1d(32 → 64, k=5) + BN + ReLU + MaxPool(2) + Dropout
  Conv1d(64 → 128, k=3) + BN + ReLU + MaxPool(2) + Dropout
  Conv1d(128 → 128, k=3) + BN + ReLU + MaxPool(2) + Dropout
  AdaptiveAvgPool1d(1)
  Linear(128 → 64) + ReLU + Dropout
  Linear(64 → 4)
```

The model has 93,636 trainable parameters. Adaptive average pooling makes the
classifier tolerant to small timing shifts and keeps the head independent of the
exact frame length used during tracing.

## Training

| Setting | Value |
| --- | --- |
| Loss | Cross-entropy with label smoothing 0.05 |
| Optimizer | AdamW, lr = 1e-3, weight decay = 1e-4 |
| Schedule | Cosine annealing over the epoch budget |
| Batch size | 128 |
| Epochs | 12 (early stopping, patience 4) |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | CPU |

## Evaluation

The held-out test split is evaluated once. The per-SNR curve is generated from
fresh frames so that no test frame is reused across SNR points. Metrics are
written to [`../reports/metrics.json`](../reports/metrics.json) and
[`../reports/per_snr.csv`](../reports/per_snr.csv).

| Metric | Value |
| --- | --- |
| Test accuracy (mixed SNR) | **69.89%** |
| Test macro F1 | **67.31%** |
| Accuracy at 10 dB | **79.50%** |
| Accuracy at 0 dB | **67.00%** |
| Parameters | 93,636 |

Accuracy at very low SNR is expected to approach the 25% random baseline. The
model is strongest above roughly 0 dB, where the constellation structure is
visible.

## Limitations

- **Synthetic channel only.** No multipath, shadowing, IQ imbalance, DC offset,
  phase noise, sampling clock drift or PA nonlinearity.
- **Four linear modulations.** No APSK, OFDM subcarrier modulation, FSK, analog
  signals or burst/preamble detection.
- **Fixed frame length.** The serving contract is exactly 1,024 I/Q pairs; other
  lengths are rejected rather than resampled.
- **SNR dependence.** Below about −4 dB the task is close to chance for higher-order
  QAM; reported aggregate accuracy hides this.
- **Class balance.** Training and evaluation are artificially balanced.
- **No calibration.** Softmax confidence is not calibrated; treat it as a ranking
  score, not a probability guarantee.
- **TorchScript.** Serving uses `torch.jit`, which is deprecated in newer PyTorch
  releases; the roadmap includes an ONNX export.

## Ethical and operational considerations

- Only classify signals you are authorized to receive and process.
- Predictions are advisory; a human RF engineer should review low-confidence or
  low-SNR outputs.
- The API logs no raw IQ data; extend logging deliberately and respect spectrum
  privacy regulation.
- If deployed, add authentication, TLS, rate limiting and monitoring for drift in
  noise floor, sample rate and hardware front-end.
- Do not use the model for automated jamming, interception or enforcement.

## Maintenance

Retrain when the modulation set, frame length, sample rate or channel model
changes. Track per-SNR accuracy and macro F1 rather than aggregate accuracy alone;
a model can improve its average while collapsing in the low-SNR regime. Keep the
TorchScript artifact, JSON metadata and README metrics in sync, and rerun the
model card review before any real-world claim.
