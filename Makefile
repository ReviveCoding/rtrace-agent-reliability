.PHONY: install test coverage lint format typecheck build validate run smoke verify incidents multiseed docker-smoke clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

coverage:
	python -m pytest --cov=rtrace --cov-report=term-missing --cov-report=xml

lint:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts

typecheck:
	python -m mypy src/rtrace

build:
	rm -rf build dist
	python -m build

validate:
	rtrace validate-data --output artifacts/validate --seed 17 --overwrite

run:
	rtrace run-all --output artifacts/demo --seed 17 --overwrite
	python scripts/verify_output.py --output artifacts/demo

smoke:
	rtrace run-all --output artifacts/smoke --seed 17 --config configs/ci_smoke.yaml --overwrite
	python scripts/verify_output.py --output artifacts/smoke

verify: test lint typecheck build smoke

incidents:
	rtrace replay-incidents --output artifacts/incidents --seed 17 --overwrite

multiseed:
	rtrace run-multiseed --output artifacts/multiseed --seeds 11,17,23,29,31

docker-smoke:
	@set -eu; \
	name=rtrace-agentic-evaluation-local; \
	trap 'docker rm -f $$name >/dev/null 2>&1 || true' EXIT; \
	rm -rf docker-artifacts; \
	docker build --tag rtrace-agentic-evaluation:local .; \
	docker run --name $$name rtrace-agentic-evaluation:local run-all --output /app/artifacts/docker --seed 17 --config /app/configs/ci_smoke.yaml; \
	mkdir -p docker-artifacts; \
	docker cp $$name:/app/artifacts/docker/. docker-artifacts; \
	python scripts/verify_output.py --output docker-artifacts

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('artifacts'), pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('wheel-artifacts'), pathlib.Path('docker-artifacts'), pathlib.Path('.wheel-venv'), pathlib.Path('src/rtrace_agentic_evaluation.egg-info'), pathlib.Path('htmlcov')]]; [p.unlink(missing_ok=True) for p in [pathlib.Path('.coverage'), pathlib.Path('coverage.xml'), pathlib.Path('coverage.json')]]"
