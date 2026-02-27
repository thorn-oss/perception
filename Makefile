TEST_SCOPE?=tests/

.PHONY: build build-wheel build-sdist verify-version init-project init test lint_check type_check format format_check precommit

init-project:
	poetry install --all-extras

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

verify-version:
	@echo "Poetry: $$(poetry --version)"
	@echo "Poetry plugins:"
	poetry self show plugins
	@echo "Git describe: $$(git describe --tags --always)"
	@poetry self show plugins | grep -q "poetry-dynamic-versioning"

build-wheel:
	poetry run pip -q install repairwheel
	poetry self add "poetry-dynamic-versioning[plugin]"
	$(MAKE) verify-version
	poetry build --format="wheel" --output="dist-tmp"
	poetry run repairwheel -o dist dist-tmp/*.whl
	@find dist -name "*.whl" -type f | sed -n "s/\(.*\)\.linux.*\.whl$$/& \1.whl/p" | xargs -r -n 2 mv # Fix wheel name
	@rm -rf dist-tmp

build-sdist:
	poetry self add "poetry-dynamic-versioning[plugin]"
	$(MAKE) verify-version
	poetry build --format="sdist" --output="dist"

build: build-wheel build-sdist
