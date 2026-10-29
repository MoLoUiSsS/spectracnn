from __future__ import annotations

import numpy as np
import torch

from spectracnn.evaluate import classification_metrics
from spectracnn.model import (
    ModulationCNN,
    count_parameters,
    load_torchscript,
    predict_proba,
    save_torchscript,
)
from spectracnn.signals import MODULATIONS


def test_forward_shape_and_parameters():
    model = ModulationCNN()
    logits = model(torch.randn(5, 2, 1024))
    assert logits.shape == (5, len(MODULATIONS))
    assert count_parameters(model) > 80_000


def test_torchscript_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ModulationCNN().eval()
    path = save_torchscript(model, tmp_path / "model.pt", length=256)
    loaded = load_torchscript(path)

    x = torch.randn(3, 2, 256)
    with torch.inference_mode():
        expected = model(x)
        actual = loaded(x)
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)


def test_predict_proba_sums_to_one():
    model = ModulationCNN().eval()
    probabilities = predict_proba(model, torch.randn(7, 2, 256))
    assert probabilities.shape == (7, len(MODULATIONS))
    np.testing.assert_allclose(
        probabilities.sum(dim=1).numpy(),
        np.ones(7),
        atol=1e-5,
    )
    assert (probabilities >= 0).all()


def test_classification_metrics_perfect():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    metrics = classification_metrics(y, y, list(MODULATIONS))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["n_test"] == 8
    assert metrics["confusion_matrix"][0][0] == 2


def test_classification_metrics_errors():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 2])
    metrics = classification_metrics(y_true, y_pred, list(MODULATIONS))
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["BPSK"]["recall"] == 1.0
    assert metrics["per_class"]["QPSK"]["recall"] == 0.5
    assert metrics["n_test"] == 8


def test_classification_metrics_rejects_mismatch():
    try:
        classification_metrics(np.zeros(3), np.zeros(4), list(MODULATIONS))
    except ValueError as error:
        assert "same length" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
