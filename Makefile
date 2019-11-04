IMAGE_NAME = perception
VOLUME_NAME = $(IMAGE_NAME)_venv
DOCKER_ARGS = -v $(PWD):/usr/src -v $(VOLUME_NAME):/usr/src/.venv --rm
IN_DOCKER = docker run $(DOCKER_ARGS) $(IMAGE_NAME) pipenv run
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
	PIPENV_VENV_IN_PROJECT=true pipenv install --dev --skip-lock
	pipenv run pip install -r docs/requirements.txt
	pipenv run pip freeze | grep opencv | xargs -n 1 pipenv run pip uninstall -y
	pipenv run pip install -U --no-cache-dir opencv-contrib-python-headless
bash:
	docker run -it $(DOCKER_ARGS)  --rm $(IMAGE_NAME) bash
lab-server:
	docker run -it $(DOCKER_ARGS)  --rm -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) pipenv run jupyter lab $(JUPYTER_OPTIONS)
documentation-server:
	docker run -it $(DOCKER_ARGS) -p $(DOCUMENTATION_PORT):$(DOCUMENTATION_PORT) $(IMAGE_NAME) pipenv run sphinx-autobuild -b html "docs" "docs/_build/html" --host 0.0.0.0 --port $(DOCUMENTATION_PORT) $(O)
test:
	$(IN_DOCKER) pytest $(TEST_SCOPE)
lint_check:
	$(IN_DOCKER) pylint -j 0 perception --rcfile=setup.cfg
type_check:
	$(IN_DOCKER) mypy perception
format:
	$(IN_DOCKER) yapf --recursive --in-place --exclude=perception/_version.py tests perception
format_check:
	$(IN_DOCKER) yapf --recursive --diff --exclude=perception/_version.py tests perception\
		|| (echo '\nUnexpected format.' && exit 1)
precommit:
	make build
	make lint_check
	make type_check
	make format_check
	make test