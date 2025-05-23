.PHONY: deps example sort-imports lint mypy test check build docs-serve docs-build docs-github docs-notebook

deps:
	uv sync

example:
	uv run python -m examples.basic
jupyter:
	uv run jupyter lab
mlflow:
	uv run mlflow ui --port 5000 --backend-store-uri examples/storage/mlruns
ray-up:
	uv run ray start --head --port 6379
ray-down:
	uv run ray stop

localstack:
	docker compose up localstack
localstack-list-objects:
	docker compose exec localstack awslocal s3api list-objects --bucket labtech-dev-bucket

sort-imports:
	uv run ruff check --select "I" --fix
lint:
	uv run ruff check
mypy:
	uv run mypy --show-error-codes labtech examples
test:
	uv run pytest \
		--cov="labtech" \
		--cov-report="html:tests/coverage" \
		--cov-report=term \
		tests/
check: lint mypy test

build: deps
	uv build

docs-serve:
	uv run mkdocs serve
docs-build:
	uv run mkdocs build
docs-github:
	uv run mkdocs gh-deploy

docs-notebooks:
	pandoc -o examples/cookbook.ipynb docs/cookbook.md
	pandoc -o examples/tutorial.ipynb docs/tutorial.md
