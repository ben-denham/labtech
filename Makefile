.PHONY: deps example lint mypy test check docs-serve docs-build docs-github

deps:
	poetry install

example:
	poetry run python -m examples.basic
jupyter:
	poetry run jupyter lab

lint:
	poetry run flake8
mypy:
	poetry run mypy --show-error-codes --enable-recursive-aliases labtech examples
test:
	poetry run pytest \
		--cov="labtech" \
		--cov-report="html:tests/coverage" \
		--cov-report=term
check: lint mypy test

docs-serve:
	poetry run mkdocs serve
docs-build:
	poetry run mkdocs build
docs-github:
	poetry run mkdocs gh-deploy
