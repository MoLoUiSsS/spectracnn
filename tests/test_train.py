from __future__ import annotations

from pathlib import Path

from spectracnn.train import _flat_metrics, build_parser


def test_flat_metrics_flattens_nested_scalars():
    metrics = {
        "test": {
            "accuracy": 0.9,
            "per_class": {"BPSK": {"f1": 0.8}},
        },
        "history": {"train_loss": [1.0, 0.5]},
        "epochs": 3,
        "flag": True,
    }

    flat = _flat_metrics(metrics)

    assert flat == {
        "test.accuracy": 0.9,
        "test.per_class.BPSK.f1": 0.8,
        "epochs": 3.0,
    }


def test_parser_defaults_match_readme():
    args = build_parser().parse_args([])

    assert args.samples_per_class == 1_500
    assert args.epochs == 12
    assert args.length == 1_024
    assert args.snr_min == -10.0
    assert args.snr_max == 20.0
    assert args.output == Path("models/spectracnn.pt")
    assert args.meta == Path("models/spectracnn.json")
