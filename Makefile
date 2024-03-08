IMAGE_NAME = perception
DOCKER_ARGS = -v $(PWD):/usr/src --rm
IN_DOCKER = docker run $(DOCKER_ARGS)
NOTEBOOK_PORT = 5000
JUPYTER_OPTIONS := --ip=0.0.0.0 --port $(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
TEST_SCOPE?=tests/

.PHONY: build init-project init bash lab-server test lint_check type_check format format_check build_ext precommit precommit_docker publish

build:
	docker build --rm --force-rm -t $(IMAGE_NAME) .

init-project:
	poetry install -E benchmarking -E matching -E experimental

init: init-project
	poetry run pre-commit install

bash:
	$(IN_DOCKER) -it $(IMAGE_NAME) bash

lab-server:
	@-docker volume rm $(VOLUME_NAME)
	$(IN_DOCKER) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) poetry run jupyter lab $(JUPYTER_OPTIONS)

test:
	poetry run pytest $(TEST_SCOPE)

lint_check:
	poetry run ruff check perception tests

type_check:
	poetry run mypy perception

format:
	poetry run black .

format_check:
	poetry run black .
	poetry run black --check . || (echo '\nUnexpected format.' && exit 1)

build_ext:
	poetry run python setup.py build_ext --inplace

precommit:
	poetry check
	make build_ext
	make lint_check
	make type_check
	make format_check
	make test

precommit_docker:
	make build
	$(IN_DOCKER) $(IMAGE_NAME) make precommit

publish:
	pip install twine
	twine upload dist/*
