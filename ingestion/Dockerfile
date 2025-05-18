# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS builder

RUN mkdir /usr/src/ingestion
# Set the working directory
WORKDIR /usr/src/ingestion

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY . .

RUN pip install uv
# Install all dependencies
# RUN uv add --no-cache-dir .

CMD ["/bin/sh"]