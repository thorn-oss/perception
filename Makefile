TEST_SCOPE?=tests/

.PHONY: build build-wheel build-sdist verify-version init-project init test lint_check type_check format format_check precommit clean-build

init-project:
	uv sync --all-extras

init: init-project
	uv run pre-commit install

test:
	uv run pytest $(TEST_SCOPE)

lint_check:
	uv run ruff check perception tests

type_check:
	uv run mypy perception

format:
	uv run black .

format_check:
	uv run black --check . || (echo '\nUnexpected format.' && exit 1)

precommit:
	uv lock --check
	uv run ruff check perception tests
	uv run mypy perception
	uv run black --check . || (echo '\nUnexpected format.' && exit 1)
	uv run pytest $(TEST_SCOPE)

verify-version:
	@echo "uv: $$(uv --version)"
	@echo "Python: $$(uv run python --version)"
	@echo "Git describe: $$(git describe --tags --always)"

clean-build:
	@rm -rf build Perception.egg-info

build-wheel:
	@rm -rf build Perception.egg-info
	@echo "uv: $$(uv --version)"
	@echo "Python: $$(uv run python --version)"
	@echo "Git describe: $$(git describe --tags --always)"
	uv build --wheel --out-dir="dist-tmp" --clear
	uv tool run --from repairwheel repairwheel -o dist dist-tmp/*.whl
	@find dist -name "*.whl" -type f | sed -n "s/\(.*\)\.linux.*\.whl$$/& \1.whl/p" | xargs -r -n 2 mv # Fix wheel name
	@rm -rf dist-tmp

build-sdist:
	@rm -rf build Perception.egg-info
	@echo "uv: $$(uv --version)"
	@echo "Python: $$(uv run python --version)"
	@echo "Git describe: $$(git describe --tags --always)"
	uv build --sdist --out-dir="dist"

build: build-wheel build-sdist
