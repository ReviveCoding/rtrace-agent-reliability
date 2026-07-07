FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install . \
    && python -m pip check \
    && python -c "import lightgbm, matplotlib, rtrace; print(rtrace.__version__)"

COPY configs ./configs

RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "rtrace.cli"]
CMD ["run-all", "--output", "/app/artifacts/docker", "--seed", "17"]
