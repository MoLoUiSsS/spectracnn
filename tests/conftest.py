from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spectracnn.api import create_app
from spectracnn.config import Settings
from spectracnn.model import ModulationCNN, count_parameters, save_torchscript
from spectracnn.signals import MODULATIONS


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("spectracnn")
    model = ModulationCNN()
    model_path = save_torchscript(model, directory / "spectracnn.pt", length=256)
    meta_path = directory / "spectracnn.json"
    meta_path.write_text(
        """
        {
          "name": "SpectraCNN",
          "version": "0.1.0",
          "trained_at": "2026-01-01T00:00:00+00:00",
          "classes": ["BPSK", "QPSK", "16QAM", "64QAM"],
          "length": 256,
          "parameters": 93636,
          "test": {"accuracy": 0.9, "macro_f1": 0.9}
        }
        """,
        encoding="utf-8",
    )
    return SimpleNamespace(
        model_path=model_path,
        meta_path=meta_path,
        parameters=count_parameters(model),
    )


@pytest.fixture()
def client(model_bundle, tmp_path):
    settings = Settings(
        model_path=model_bundle.model_path,
        meta_path=model_bundle.meta_path,
        web_dir=tmp_path / "web",
        api_key=None,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def secured_client(model_bundle, tmp_path):
    settings = Settings(
        model_path=model_bundle.model_path,
        meta_path=model_bundle.meta_path,
        web_dir=tmp_path / "web",
        api_key="secret-key",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def valid_iq(model_bundle):
    return [[0.1, -0.2] for _ in range(256)]


@pytest.fixture()
def class_names():
    return list(MODULATIONS)
