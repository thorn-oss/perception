TEST_SCOPE?=tests/

.PHONY: init-project init test lint_check type_check format format_check precommit

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
