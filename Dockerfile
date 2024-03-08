FROM python:3.9.18-bookworm

WORKDIR /usr/src
RUN apt-get update && apt-get install -y ffmpeg cmake && rm -rf /var/lib/apt/lists/*
COPY ./versioneer.py ./
COPY ./setup* ./
COPY ./docs/requirements.txt ./docs/requirements.txt
COPY Makefile pyproject.toml poetry.lock ./
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 - &&\
	poetry config virtualenvs.create false &&\
	poetry install -E benchmarking -E matching -E experimental --no-root
COPY ./README.md ./README.md
COPY ./perception ./perception
RUN poetry install -E benchmarking -E matching -E experimental --only-root
