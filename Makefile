IMAGE_NAME = perception
VOLUME_NAME = $(IMAGE_NAME)_venv
DOCKER_ARGS = -v $(PWD):/usr/src -v $(VOLUME_NAME):/usr/src/.venv --rm
IN_DOCKER = docker run $(DOCKER_ARGS)
NOTEBOOK_PORT = 5000
DOCUMENTATION_PORT = 5001
JUPYTER_OPTIONS := --ip=0.0.0.0 --port $(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
TEST_SCOPE?=tests/

.PHONY: build
build:
	docker build --rm --force-rm -t $(IMAGE_NAME) .
	@-docker volume rm $(VOLUME_NAME)
init:
	# Blow away the venv to deal with pip caching issues with conflicting
	# versions of OpenCV.
	rm -rf .venv
	PIPENV_VENV_IN_PROJECT=true pipenv install --dev
	pipenv run pip install -r docs/requirements.txt
	pipenv run pip install cython
	pipenv run pip freeze | grep opencv | xargs -n 1 pipenv run pip uninstall -y
	pipenv run pip install -U --no-cache-dir opencv-contrib-python-headless
bash:
	$(IN_DOCKER) -it $(IMAGE_NAME) bash
lab-server:
	@-docker volume rm $(VOLUME_NAME)
	$(IN_DOCKER) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) pipenv run jupyter lab $(JUPYTER_OPTIONS)
documentation-server:
	$(IN_DOCKER) -p $(DOCUMENTATION_PORT):$(DOCUMENTATION_PORT) $(IMAGE_NAME) pipenv run sphinx-autobuild -b html "docs" "docs/_build/html" --host 0.0.0.0 --port $(DOCUMENTATION_PORT) $(O)
test:
	pipenv run pytest $(TEST_SCOPE)
lint_check:
	pipenv run pylint perception --rcfile=setup.cfg
lint_check_parallel:
	pipenv run pylint -j 0 perception --rcfile=setup.cfg
type_check:
	pipenv run mypy perception
format:
	pipenv run black .
format_check:
	pipenv run black --check . || (echo '\nUnexpected format.' && exit 1)
build_ext:
	pipenv run python setup.py build_ext --inplace
precommit:
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
