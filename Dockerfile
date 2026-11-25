# Scoring API image. Serves the frozen EIF (and IF for comparison) through serving/api.py.
# The model artefact is copied in at build time from models/ (untracked). Nothing is fitted here.
#
#   docker build --platform linux/arm64 -t triage-api:arm64 .       # Apple Silicon
#   docker build --platform linux/amd64 -t triage-api:amd64 .       # x86_64, the cloud architecture
#
# Run with the key from the environment, never on the command line or in the image:
#   docker run --rm -p 127.0.0.1:8000:8000 -e TRIAGE_API_KEY triage-api:arm64

# Debian 13 (trixie) gives GCC 14. GCC 12.2 in bookworm crashed compiling isotree on arm64
# (internal compiler error in the vectoriser during LTO). Both stages use the same base.
ARG PYTHON=3.13-slim-trixie

# build stage: compile wheels (isotree has no wheel for this Python, so it builds from source
# with its own flags, including -O3 and -flto)
FROM python:${PYTHON} AS build
# isotree adds -march=native unless this is set; native code may not run on another CPU
ENV DONT_SET_MARCH=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY serving/requirements-serving.txt /tmp/requirements.txt
RUN pip wheel --wheel-dir /wheels -r /tmp/requirements.txt

# runtime stage: no compilers or raw CSVs; includes private serving artefact; runs as non-root
FROM python:${PYTHON}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRIAGE_SERVING_BUNDLE=/app/models/serving_bundle.joblib TRIAGE_DOCS=0
COPY --from=build /wheels /wheels
COPY serving/requirements-serving.txt /tmp/requirements.txt
# libgomp1: the OpenMP runtime the compiled isotree module links against
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-index --find-links /wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt \
    && useradd --create-home --uid 10001 app

WORKDIR /app
# only the modules the API and the pickled models need
COPY src/__init__.py src/constants.py src/entity_resolution.py src/features.py \
     src/pipeline.py src/transformers.py src/tuning.py /app/src/
COPY model_evaluation/eif_eval.py model_evaluation/harness.py /app/model_evaluation/
COPY serving/__init__.py serving/api.py serving/core.py serving/schema.py /app/serving/
COPY models/serving_bundle.joblib models/serving_bundle.json /app/models/
# build fails here if a shared library is missing, the artefact hash differs, or the models cannot score
RUN so=$(python -c "import isotree, glob, os; print(glob.glob(os.path.dirname(isotree.__file__) + '/_cpp_interface*.so')[0])") \
    && ldd "$so" && ! ldd "$so" | grep -q "not found" \
    && python -c "from serving.api import load_bundle; b, _ = load_bundle('/app/models/serving_bundle.joblib'); \
print('scores ok', [round(float(b['models'][m].score(b['background'][:3])[0]), 6) for m in ('eif', 'if')])"

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# the app checks the artefact's SHA-256 against serving_bundle.json at startup and exits on a mismatch
CMD ["python", "-m", "uvicorn", "serving.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--limit-concurrency", "10", "--no-server-header"]
