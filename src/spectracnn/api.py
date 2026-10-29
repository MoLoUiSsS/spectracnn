"""FastAPI application factory, TorchScript inference and demo endpoints."""

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings, get_settings
from .model import load_torchscript
from .schema import (
    DemoResponse,
    HealthResponse,
    IQRequest,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
)
from .signals import (
    DEFAULT_LENGTH,
    MODULATIONS,
    frame_to_pairs,
    generate_sample,
)

logger = logging.getLogger("spectracnn.api")


@dataclass
class LoadedModel:
    module: torch.jit.ScriptModule
    classes: list[str]
    length: int
    metrics: dict
    trained_at: str
    parameters: int

    def predict(self, iq: np.ndarray) -> tuple[str, float, dict[str, float]]:
        iq = np.asarray(iq, dtype=np.float32)
        if iq.ndim != 2 or iq.shape[1] != 2:
            raise ValueError("iq must have shape (n, 2)")
        if not np.isfinite(iq).all():
            raise ValueError("iq frame contains non-finite values")
        power = float(np.mean(iq**2))
        if power <= 1e-12:
            raise ValueError("iq frame has zero power")
        normalized = (iq / np.sqrt(power)).astype(np.float32)
        tensor = torch.from_numpy(normalized.T[None, :, :].copy())

        with torch.inference_mode():
            probabilities = torch.softmax(self.module(tensor), dim=1)[0].numpy()
        index = int(np.argmax(probabilities))
        return (
            self.classes[index],
            float(probabilities[index]),
            {label: float(probabilities[position]) for position, label in enumerate(self.classes)},
        )


def load_bundle(settings: Settings) -> LoadedModel:
    metadata: dict = {}
    if settings.meta_path.exists():
        metadata = json.loads(settings.meta_path.read_text(encoding="utf-8"))
    module = load_torchscript(settings.model_path)
    return LoadedModel(
        module=module,
        classes=list(metadata.get("classes", MODULATIONS)),
        length=int(metadata.get("length", DEFAULT_LENGTH)),
        metrics=dict(metadata.get("test", {})),
        trained_at=str(metadata.get("trained_at", "")),
        parameters=int(metadata.get("parameters", 0)),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, LoadedModel | None] = {"bundle": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        if settings.model_path.exists():
            state["bundle"] = load_bundle(settings)
            logger.info("loaded TorchScript model from %s", settings.model_path)
        else:
            logger.warning(
                "model not found at %s; run 'spectracnn-train' before serving",
                settings.model_path,
            )
        app.state.bundle = state["bundle"]
        yield

    app = FastAPI(
        title="SpectraCNN",
        version=settings.model_version,
        description=(
            "Automatic modulation classification (BPSK, QPSK, 16-QAM, 64-QAM) "
            "from raw I/Q samples with a PyTorch 1D CNN."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_bundle() -> LoadedModel:
        bundle = state["bundle"]
        if bundle is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Run 'spectracnn-train' first.",
            )
        return bundle

    def require_api_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    @app.get("/", include_in_schema=False)
    def index():
        page = settings.web_dir / "index.html"
        if page.exists():
            return FileResponse(page)
        return RootResponse(
            name="SpectraCNN",
            version=settings.model_version,
            docs="/docs",
            health="/health",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"description": "Model not loaded"}},
        tags=["meta"],
    )
    def health():
        loaded = state["bundle"] is not None
        payload = HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            version=settings.model_version,
        )
        return JSONResponse(
            status_code=200 if loaded else 503,
            content=payload.model_dump(),
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
    def model_info(bundle: Annotated[LoadedModel, Depends(get_bundle)]) -> ModelInfoResponse:
        return ModelInfoResponse(
            name="SpectraCNN",
            version=settings.model_version,
            trained_at=bundle.trained_at,
            classes=bundle.classes,
            length=bundle.length,
            parameters=bundle.parameters,
            metrics=bundle.metrics,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        dependencies=[Depends(require_api_key)],
        tags=["inference"],
    )
    def predict(
        payload: IQRequest,
        bundle: Annotated[LoadedModel, Depends(get_bundle)],
    ) -> PredictionResponse:
        if len(payload.iq) != bundle.length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"expected exactly {bundle.length} I/Q samples, got {len(payload.iq)}",
            )
        started = time.perf_counter()
        try:
            modulation, confidence, probabilities = bundle.predict(np.asarray(payload.iq))
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        latency_ms = (time.perf_counter() - started) * 1000
        return PredictionResponse(
            modulation=modulation,
            confidence=round(confidence, 6),
            probabilities={label: round(value, 6) for label, value in probabilities.items()},
            model_version=settings.model_version,
            latency_ms=round(latency_ms, 3),
        )

    @app.get("/demo", response_model=DemoResponse, tags=["demo"])
    def demo(
        bundle: Annotated[LoadedModel, Depends(get_bundle)],
        snr_db: Annotated[float, Query(ge=-20, le=30)] = 10.0,
        modulation: Annotated[str | None, Query()] = None,
    ) -> DemoResponse:
        rng = np.random.default_rng()
        if modulation is None:
            modulation = MODULATIONS[int(rng.integers(0, len(MODULATIONS)))]
        if modulation not in MODULATIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"modulation must be one of {MODULATIONS}",
            )

        frame = generate_sample(modulation, bundle.length, snr_db, rng)
        started = time.perf_counter()
        predicted, confidence, probabilities = bundle.predict(frame.T)
        latency_ms = (time.perf_counter() - started) * 1000
        return DemoResponse(
            modulation=predicted,
            confidence=round(confidence, 6),
            probabilities={label: round(value, 6) for label, value in probabilities.items()},
            true_modulation=modulation,
            snr_db=float(snr_db),
            constellation=frame_to_pairs(frame, limit=256),
            model_version=settings.model_version,
            latency_ms=round(latency_ms, 3),
        )

    return app
