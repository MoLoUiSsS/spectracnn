"""Training CLI with per-SNR evaluation, TorchScript export and MLflow tracking."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import IQDataset, stratified_split
from .evaluate import (
    classification_metrics,
    save_confusion_matrix,
    save_per_snr_curve,
    save_training_curves,
)
from .model import ModulationCNN, count_parameters, predict_proba, save_torchscript
from .signals import DEFAULT_LENGTH, MODULATIONS, generate_dataset, generate_eval_set, snr_grid


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for X, y in loader:
            logits = model(X)
            total_loss += float(criterion(logits, y)) * len(y)
            predictions.append(logits.argmax(dim=1).numpy())
            labels.append(y.numpy())
            total += len(y)
    predicted = np.concatenate(predictions)
    truth = np.concatenate(labels)
    accuracy = float((predicted == truth).mean())
    return total_loss / total, accuracy, predicted, truth


def predict_numpy(
    module: nn.Module | torch.jit.ScriptModule,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size].astype(np.float32))
            probabilities.append(predict_proba(module, batch).numpy())
    return np.concatenate(probabilities)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    started = time.perf_counter()

    print(
        f"generating {args.samples_per_class * len(MODULATIONS):,} frames "
        f"({args.samples_per_class}/class, {args.snr_min:+.0f}..{args.snr_max:+.0f} dB) ..."
    )
    X, y = generate_dataset(
        samples_per_class=args.samples_per_class,
        length=args.length,
        snr_range=(args.snr_min, args.snr_max),
        seed=args.seed,
    )
    train_idx, val_idx, test_idx = stratified_split(
        y,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    train_loader = DataLoader(
        IQDataset(X[train_idx], y[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        IQDataset(X[val_idx], y[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        IQDataset(X[test_idx], y[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ModulationCNN(n_classes=len(MODULATIONS))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = 0.0
    best_epoch = 0
    stale_epochs = 0

    print(f"training {count_parameters(model):,} parameters on CPU ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach()) * len(batch_y)
            running_correct += int((logits.argmax(dim=1) == batch_y).sum())
            running_total += len(batch_y)

        train_loss = running_loss / running_total
        train_accuracy = running_correct / running_total
        val_loss, val_accuracy, _, _ = _evaluate_loader(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_accuracy"].append(round(train_accuracy, 5))
        history["val_accuracy"].append(round(val_accuracy, 5))

        marker = ""
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
            marker = " *"
        else:
            stale_epochs += 1

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}{marker}"
        )

        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    test_loss, test_accuracy, test_pred, test_true = _evaluate_loader(model, test_loader, criterion)
    test_metrics = classification_metrics(test_true, test_pred, list(MODULATIONS))
    test_metrics["loss"] = round(test_loss, 5)

    per_snr: list[dict[str, float]] = []
    print("\nper-SNR evaluation")
    for snr_db in snr_grid(args.eval_snr_min, args.eval_snr_max, args.eval_snr_step):
        X_eval, y_eval = generate_eval_set(
            snr_db,
            per_class=args.eval_per_class,
            length=args.length,
            seed=1_000 + int(round(snr_db * 10)),
        )
        probabilities = predict_numpy(model, X_eval)
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(y_eval, predictions, list(MODULATIONS))
        row = {
            "snr_db": float(snr_db),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        per_snr.append(row)
        print(
            f"  {snr_db:>5.1f} dB  accuracy={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}"
        )

    plot_snr = min(max(10.0, args.eval_snr_min), args.eval_snr_max)
    X_plot, y_plot = generate_eval_set(
        plot_snr,
        per_class=args.eval_per_class,
        length=args.length,
        seed=7_777,
    )
    plot_predictions = predict_numpy(model, X_plot).argmax(axis=1)
    plot_metrics = classification_metrics(y_plot, plot_predictions, list(MODULATIONS))

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "test": test_metrics,
        "per_snr": per_snr,
        "confusion_at_snr": plot_snr,
        "confusion_at_snr_metrics": plot_metrics,
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_accuracy, 5),
            "parameters": count_parameters(model),
            "train_seconds": round(time.perf_counter() - started, 2),
            "history": history,
            "config": {
                "samples_per_class": args.samples_per_class,
                "length": args.length,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "snr_min": args.snr_min,
                "snr_max": args.snr_max,
                "seed": args.seed,
            },
        },
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (reports_dir / "per_snr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snr_db", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(per_snr)

    save_training_curves(history, reports_dir / "training_curve.png")
    save_per_snr_curve(per_snr, reports_dir / "per_snr.png")
    save_confusion_matrix(
        plot_metrics["confusion_matrix"],
        list(MODULATIONS),
        reports_dir / "confusion_matrix.png",
        title=f"Confusion matrix at {plot_snr:.0f} dB",
    )

    output = save_torchscript(model, args.output, args.length)
    metadata = {
        "name": "SpectraCNN",
        "version": "0.1.0",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": list(MODULATIONS),
        "length": args.length,
        "input": "2 x length unit-power I/Q frame",
        "parameters": count_parameters(model),
        "training": metrics["training"],
        "test": test_metrics,
        "per_snr": per_snr,
    }
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTest set")
    print(f"  accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  macro F1 : {test_metrics['macro_f1']:.4f}")
    for label, values in test_metrics["per_class"].items():
        print(
            f"  {label:<6} precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f}"
        )
    print(f"\n  parameters : {count_parameters(model):,}")
    print(f"  best epoch : {best_epoch}")
    print(f"  duration   : {metrics['training']['train_seconds']:.1f}s")

    print(f"\nmodel   -> {output}")
    print(f"meta    -> {meta_path}")
    print(f"metrics -> {metrics_path}")

    if args.mlflow:
        _log_mlflow(args, metrics, output, meta_path, metrics_path, reports_dir)

    return metrics


def _log_mlflow(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    model_path: Path,
    meta_path: Path,
    metrics_path: Path,
    reports_dir: Path,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed; run: uv pip install -e '.[mlops]'")
        return

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"cnn-{args.samples_per_class}x{len(MODULATIONS)}"):
        mlflow.log_params(
            {
                "samples_per_class": args.samples_per_class,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "length": args.length,
                "snr_min": args.snr_min,
                "snr_max": args.snr_max,
                "seed": args.seed,
            }
        )
        mlflow.log_metrics(_flat_metrics(metrics))
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        for name in ("per_snr.csv", "per_snr.png", "confusion_matrix.png", "training_curve.png"):
            candidate = reports_dir / name
            if candidate.exists():
                mlflow.log_artifact(str(candidate), artifact_path="reports")
        print(f"mlflow run logged to {mlflow.get_tracking_uri()}")


def _flat_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "history":
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flat_metrics(value, f"{name}."))
        elif isinstance(value, list):
            continue
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SpectraCNN")
    parser.add_argument("--samples-per-class", type=int, default=1_500)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snr-min", type=float, default=-10.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-class", type=int, default=100)
    parser.add_argument("--eval-snr-min", type=float, default=-10.0)
    parser.add_argument("--eval-snr-max", type=float, default=20.0)
    parser.add_argument("--eval-snr-step", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/spectracnn.pt"))
    parser.add_argument("--meta", type=Path, default=Path("models/spectracnn.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--experiment", default="spectracnn")
    return parser


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
