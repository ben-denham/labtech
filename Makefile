.PHONY: deps example lint mypy test check build docs-serve docs-build docs-github docs-notebook

deps:
	poetry install

example:
	poetry run python -m examples.basic
jupyter:
	poetry run jupyter lab
mlflow:
	poetry run mlflow ui --port 5000 --backend-store-uri examples/storage/mlruns

lint:
	poetry run flake8 labtech
mypy:
	poetry run mypy --show-error-codes labtech examples
test:
	poetry run pytest \
		--cov="labtech" \
		--cov-report="html:tests/coverage" \
		--cov-report=term
check: lint mypy test

build:
	poetry build

docs-serve:
	poetry run mkdocs serve
docs-build:
	poetry run mkdocs build
docs-github:
	poetry run mkdocs gh-deploy

docs-notebooks:
	pandoc -o examples/cookbook.ipynb docs/cookbook.md
	pandoc -o examples/tutorial.ipynb docs/tutorial.md
