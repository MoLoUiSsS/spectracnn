"""Synthetic I/Q signal generation for modulation classification.

The chain is intentionally explicit so every impairment can be inspected and
tested: random bits -> complex symbols -> RRC pulse shaping -> carrier offset,
phase, timing offset, amplitude and AWGN. Frames are returned as unit-power
``(2, length)`` float32 arrays (I and Q rows).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MODULATIONS: tuple[str, ...] = ("BPSK", "QPSK", "16QAM", "64QAM")
MODULATION_INDEX: dict[str, int] = {name: index for index, name in enumerate(MODULATIONS)}

DEFAULT_LENGTH = 1024
SPS = 8  # samples per symbol
RRC_SPAN = 8  # symbols on each side of the filter
RRC_BETA = 0.35


def rrc_taps(
    beta: float = RRC_BETA,
    sps: int = SPS,
    span: int = RRC_SPAN,
) -> np.ndarray:
    """Root-raised-cosine filter taps (unit energy)."""

    t = np.arange(-span * sps, span * sps + 1, dtype=float) / sps
    taps = np.zeros_like(t)
    for index, value in enumerate(t):
        if abs(value) < 1e-12:
            taps[index] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(4.0 * beta * value) - 1.0) < 1e-9:
            taps[index] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * value * (1.0 - beta)) + 4.0 * beta * value * np.cos(
                np.pi * value * (1.0 + beta)
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def _qam_levels(order: int) -> np.ndarray:
    side = int(np.sqrt(order))
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    # Normalise the complex symbol, i.e. both I and Q rails.
    return levels / np.sqrt(2.0 * np.mean(levels**2))


def modulate(
    modulation: str,
    n_symbols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Map random bits to unit-average-power complex symbols."""

    if modulation not in MODULATION_INDEX:
        raise ValueError(f"unknown modulation {modulation!r}; expected one of {MODULATIONS}")

    if modulation == "BPSK":
        bits = rng.integers(0, 2, n_symbols)
        return (2.0 * bits - 1.0).astype(np.complex128)

    if modulation == "QPSK":
        bits = rng.integers(0, 2, size=(n_symbols, 2))
        symbols = (1.0 - 2.0 * bits[:, 0]) + 1j * (1.0 - 2.0 * bits[:, 1])
        return symbols / np.sqrt(2.0)

    order = 16 if modulation == "16QAM" else 64
    levels = _qam_levels(order)
    indices = rng.integers(0, len(levels), size=(n_symbols, 2))
    return levels[indices[:, 0]] + 1j * levels[indices[:, 1]]


def pulse_shape(
    symbols: np.ndarray,
    sps: int = SPS,
    taps: np.ndarray | None = None,
) -> np.ndarray:
    """Upsample and apply RRC pulse shaping."""

    coefficients = rrc_taps(sps=sps) if taps is None else taps
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    filtered = np.convolve(upsampled, coefficients, mode="full")
    delay = (len(coefficients) - 1) // 2
    return filtered[delay : delay + len(upsampled)]


def normalize_power(signal: np.ndarray) -> np.ndarray:
    """Scale a complex signal to unit average power."""

    power = float(np.mean(np.abs(signal) ** 2))
    if power <= 0:
        return signal.astype(np.complex128)
    return signal / np.sqrt(power)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add circular complex Gaussian noise at the requested SNR."""

    power = float(np.mean(np.abs(signal) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise


def apply_impairments(
    signal: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
    cfo: float | None = None,
    phase: float | None = None,
    timing: int | None = None,
    amplitude: float | None = None,
) -> np.ndarray:
    """Apply CFO, phase, timing, amplitude and AWGN, then restore unit power."""

    signal = normalize_power(signal)
    n_samples = len(signal)

    frequency_offset = rng.uniform(-0.01, 0.01) if cfo is None else cfo
    phase_offset = rng.uniform(0.0, 2.0 * np.pi) if phase is None else phase
    timing_offset = int(rng.integers(0, SPS)) if timing is None else timing
    gain = rng.uniform(0.7, 1.4) if amplitude is None else amplitude

    time = np.arange(n_samples)
    rotated = signal * np.exp(1j * (2.0 * np.pi * frequency_offset * time + phase_offset))
    rotated = np.roll(rotated, timing_offset) * gain
    noisy = add_awgn(rotated, snr_db, rng)
    return normalize_power(noisy)


def generate_sample(
    modulation: str,
    length: int = DEFAULT_LENGTH,
    snr_db: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one unit-power ``(2, length)`` I/Q frame."""

    if length < 64:
        raise ValueError("length must be at least 64 samples")
    generator = np.random.default_rng() if rng is None else rng

    n_symbols = int(np.ceil((length + 2 * SPS * RRC_SPAN) / SPS)) + 1
    symbols = modulate(modulation, n_symbols, generator)
    shaped = pulse_shape(symbols)
    if len(shaped) < length:
        shaped = np.pad(shaped, (0, length - len(shaped)))
    shaped = shaped[:length]

    impaired = apply_impairments(shaped, generator, snr_db)
    frame = np.stack([impaired.real, impaired.imag]).astype(np.float32)
    return frame


def generate_dataset(
    samples_per_class: int = 1_500,
    length: int = DEFAULT_LENGTH,
    snr_range: tuple[float, float] = (-10.0, 20.0),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced mixed-SNR dataset.

    Returns:
        X: ``(n_samples, 2, length)`` float32
        y: ``(n_samples,)`` int64 class indices following ``MODULATIONS``
    """

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")
    rng = np.random.default_rng(seed)
    total = samples_per_class * len(MODULATIONS)
    X = np.empty((total, 2, length), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)

    cursor = 0
    for class_index, modulation in enumerate(MODULATIONS):
        for _ in range(samples_per_class):
            snr_db = float(rng.uniform(*snr_range))
            X[cursor] = generate_sample(modulation, length, snr_db, rng)
            y[cursor] = class_index
            cursor += 1

    permutation = rng.permutation(total)
    return X[permutation], y[permutation]


def generate_eval_set(
    snr_db: float,
    per_class: int = 100,
    length: int = DEFAULT_LENGTH,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced evaluation set at one fixed SNR."""

    rng = np.random.default_rng(seed)
    total = per_class * len(MODULATIONS)
    X = np.empty((total, 2, length), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    cursor = 0
    for class_index, modulation in enumerate(MODULATIONS):
        for _ in range(per_class):
            X[cursor] = generate_sample(modulation, length, snr_db, rng)
            y[cursor] = class_index
            cursor += 1
    return X, y


def frame_to_pairs(frame: np.ndarray, limit: int | None = None) -> list[list[float]]:
    """Convert a ``(2, length)`` frame into JSON-friendly I/Q pairs."""

    count = frame.shape[1] if limit is None else min(limit, frame.shape[1])
    return [
        [round(float(frame[0, index]), 5), round(float(frame[1, index]), 5)]
        for index in range(count)
    ]


def snr_grid(
    low: float = -10.0,
    high: float = 20.0,
    step: float = 2.0,
) -> Sequence[float]:
    return np.arange(low, high + step / 2.0, step).tolist()
