TEST_SCOPE?=tests/

.PHONY: build build-wheel build-sdist init-project init test lint_check type_check format format_check precommit

init-project:
	poetry install -E benchmarking -E matching -E experimental

init: init-project
	poetry run pre-commit install

test:
	poetry run pytest $(TEST_SCOPE)

lint_check:
	poetry run ruff check perception tests

type_check:
	poetry run mypy perception

format:
	poetry run black .

format_check:
	poetry run black --check . || (echo '\nUnexpected format.' && exit 1)

precommit:
	poetry check
	make lint_check
	make type_check
	make format_check
	make test

build-wheel:
	@poetry run pip install repairwheel
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@poetry build --format="wheel" --output="dist-tmp"
	@poetry run repairwheel -o dist dist-tmp/*.whl
	@rm -rf dist-tmp

build-sdist:
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@poetry build --format="wheel" --output="dist"

build: build-wheel build-sdist
