FROM python:3.9.18-bookworm

WORKDIR /usr/src
RUN apt-get update && apt-get install -y ffmpeg cmake git && rm -rf /var/lib/apt/lists/*
COPY Makefile pyproject.toml poetry.lock build.py ./
ENV PATH="/root/.local/bin:$PATH"
COPY ./.git ./.git
RUN curl -sSL https://install.python-poetry.org | python3 - &&\
    poetry self add "poetry-dynamic-versioning[plugin]" &&\
	poetry config virtualenvs.in-project true &&\
	poetry install -E benchmarking -E matching -E experimental --no-root
COPY ./README.md ./README.md
COPY ./perception ./perception
RUN poetry install -E benchmarking -E matching -E experimental --only-root
