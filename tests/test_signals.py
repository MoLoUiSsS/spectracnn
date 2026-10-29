from __future__ import annotations

import numpy as np
import pytest

from spectracnn.dataset import stratified_split
from spectracnn.signals import (
    MODULATIONS,
    add_awgn,
    generate_dataset,
    generate_eval_set,
    generate_sample,
    modulate,
    normalize_power,
    pulse_shape,
    rrc_taps,
)


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_generate_sample_contract(modulation):
    frame = generate_sample(modulation, length=512, snr_db=8.0, rng=np.random.default_rng(1))
    assert frame.shape == (2, 512)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    # Total complex power is 1, so the mean over I and Q is 0.5.
    assert 0.4 < float(np.mean(frame**2)) < 0.6


def test_generation_is_reproducible():
    first = generate_sample("16QAM", 256, 5.0, np.random.default_rng(7))
    second = generate_sample("16QAM", 256, 5.0, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)


def test_rrc_filter_is_unit_energy_and_symmetric():
    taps = rrc_taps()
    assert abs(float(np.sum(taps**2)) - 1.0) < 1e-9
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-10)


def test_modulation_power_and_shape():
    rng = np.random.default_rng(2)
    for modulation in MODULATIONS:
        symbols = modulate(modulation, 1_000, rng)
        assert len(symbols) == 1_000
        assert abs(float(np.mean(np.abs(symbols) ** 2)) - 1.0) < 0.15


def test_pulse_shaping_and_power_normalization():
    rng = np.random.default_rng(3)
    symbols = modulate("QPSK", 200, rng)
    shaped = pulse_shape(symbols)
    assert len(shaped) == len(symbols) * 8
    normalized = normalize_power(shaped)
    assert abs(float(np.mean(np.abs(normalized) ** 2)) - 1.0) < 1e-9


def test_awgn_reduces_snr():
    rng = np.random.default_rng(4)
    signal = np.exp(1j * np.linspace(0, 6, 4_096))
    clean_power = float(np.mean(np.abs(signal) ** 2))
    noisy = add_awgn(signal, snr_db=0.0, rng=rng)
    noise = noisy - signal
    estimated_snr = 10 * np.log10(clean_power / float(np.mean(np.abs(noise) ** 2)))
    assert -2.0 < estimated_snr < 2.0


def test_dataset_is_balanced_and_reproducible():
    X, y = generate_dataset(samples_per_class=20, length=128, seed=9)
    assert X.shape == (80, 2, 128)
    assert y.shape == (80,)
    assert sorted(np.bincount(y)) == [20, 20, 20, 20]
    X2, y2 = generate_dataset(samples_per_class=20, length=128, seed=9)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


def test_eval_set_is_balanced():
    X, y = generate_eval_set(10.0, per_class=12, length=128, seed=3)
    assert X.shape == (48, 2, 128)
    assert sorted(np.bincount(y)) == [12, 12, 12, 12]


def test_stratified_split_is_disjoint_and_complete():
    _, y = generate_dataset(samples_per_class=20, length=64, seed=2)
    train, val, test = stratified_split(y, 0.15, 0.15, seed=2)
    combined = np.concatenate([train, val, test])
    assert len(combined) == len(y)
    assert len(np.unique(combined)) == len(y)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    for split in (train, val, test):
        assert len(np.unique(y[split])) == 4


def test_invalid_inputs():
    with pytest.raises(ValueError):
        generate_sample("BPSK", length=32)
    with pytest.raises(ValueError):
        modulate("8PSK", 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        stratified_split(np.zeros(10, dtype=int), 0.5, 0.5)
