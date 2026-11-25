"""Scoring API (deployment stage, step 2).

Local run, from the repo root:
    TRIAGE_API_KEY=<key> python -m uvicorn serving.api:app_from_env --factory --port 8000

Routes:
  GET  /health   liveness only, no key needed
  GET  /model    frozen configuration and artefact hashes, key needed
  POST /score    1 to 20 contracts, key needed (header X-API-Key)

Scoring uses serving.core exactly as the parity gates checked it. Nothing is fitted here.
Responses never contain history records. Request bodies are not logged.
"""
import hashlib
import hmac
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving import core
from serving.schema import MAX_BODY_BYTES, ContractScore, ScoreRequest, ScoreResponse, vocab_errors
from src.constants import METHOD_LEVEL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "serving_bundle.joblib"
PUBLIC = {"/health"}
MEMO_CAP = 20_000  # resolver cache of unseen supplier names; results do not depend on it
log = logging.getLogger("triage.api")


class StartupError(RuntimeError):
    pass


def load_bundle(path):
    """Loads the artefact only if its SHA-256 matches the metadata file next to it."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise StartupError(f"artefact or metadata missing: {path.name}, {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != meta.get("artifact_sha256"):
        raise StartupError("artefact hash does not match its metadata; refusing to start")
    bundle = joblib.load(path)
    if bundle.get("fitted_state_sha256") != core.fitted_state_hash(bundle["ib"]):
        raise StartupError("fitted state hash differs; refusing to start")
    return bundle, meta


def vocabulary(bundle):
    onehot = bundle["prep"].named_transformers_["category"].steps[-1][1]
    return {"methods": set(METHOD_LEVEL), "categories": {str(c) for c in onehot.categories_[0]}}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def request_frame(contracts):
    """Validated contracts as the frame the core scores. Values are passed on unchanged."""
    return pd.DataFrame([c.model_dump() for c in contracts])


def model_info(bundle, meta):
    """What /model shows: configuration and provenance, no records."""
    keep = ["deployed_model", "comparison_model", "threshold_rule", "history", "input_names",
            "artifact_sha256", "fitted_state_sha256", "history_sha256", "git_commit", "freeze_commit",
            "official_run", "parity", "versions"]
    info = {k: meta[k] for k in keep if k in meta}
    info["thresholds"] = bundle["thresholds"]
    info["note"] = bundle["note"]
    return info


def create_app(bundle, meta, api_key, docs=True):
    if not api_key or len(api_key) < 16:
        raise StartupError("an API key of at least 16 characters is required")
    app = FastAPI(title="Kenya procurement risk triage: scoring API", version="1.0",
                  docs_url="/docs" if docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if docs else None)
    vocab = vocabulary(bundle)
    resolver = bundle["ib"].resolver_
    memo0 = dict(resolver._memo) if resolver is not None else {}
    lock = threading.Lock()  # one scoring call at a time; also caps CPU use
    info = model_info(bundle, meta)
    key = api_key.encode()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path not in PUBLIC and not (docs and path in ("/docs", "/openapi.json")):
            given = request.headers.get("x-api-key", "").encode()
            if not hmac.compare_digest(given, key):
                return JSONResponse({"detail": "missing or invalid API key"}, status_code=401)
        if request.method == "POST":
            size = request.headers.get("content-length")
            if size is None:
                return JSONResponse({"detail": "Content-Length header required"}, status_code=411)
            if not size.isdigit() or int(size) > MAX_BODY_BYTES:
                return JSONResponse({"detail": f"request body above {MAX_BODY_BYTES} bytes"}, status_code=413)
        try:
            return await call_next(request)
        except Exception:  # never return a traceback
            log.exception("unhandled error on %s", path)
            return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        errs = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                 "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": "invalid request", "errors": errs}, status_code=422)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model")
    def model():
        return info

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        errs = vocab_errors(req, vocab)
        if errs:
            return JSONResponse({"detail": "invalid request", "errors": errs}, status_code=422)
        t0 = time.perf_counter()
        df = request_frame(req.contracts)
        with lock:
            try:
                res = core.score_rows(bundle, df)
            except ValueError as e:
                return JSONResponse({"detail": "invalid request",
                                     "errors": [{"field": "contracts", "message": str(e)}]}, status_code=422)
            finally:
                if resolver is not None and len(resolver._memo) > MEMO_CAP:
                    resolver._memo = dict(memo0)
        fields = ContractScore.model_fields
        rows = [ContractScore(**{k: _clean(r[k]) for k in fields}) for r in res.to_dict("records")]
        return ScoreResponse(deployed_model="eif", comparison_model="if", thresholds=bundle["thresholds"],
                             note=bundle["note"], latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                             results=rows)

    return app


def app_from_env():
    """Factory for uvicorn. Reads TRIAGE_API_KEY, TRIAGE_SERVING_BUNDLE and TRIAGE_DOCS."""
    bundle, meta = load_bundle(os.environ.get("TRIAGE_SERVING_BUNDLE", DEFAULT_BUNDLE))
    return create_app(bundle, meta, os.environ.get("TRIAGE_API_KEY", ""),
                      docs=os.environ.get("TRIAGE_DOCS", "1") == "1")
