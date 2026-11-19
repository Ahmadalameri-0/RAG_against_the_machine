export UV_PROJECT_ENVIRONMENT ?= /goinfre/$(USER)/RAG
export VIRTUAL_ENV ?= $(UV_PROJECT_ENVIRONMENT)
export UV_CACHE_DIR ?= /goinfre/$(USER)/uv_cache
export HF_HOME ?= /goinfre/$(USER)/hf_cache
export UV_LINK_MODE = copy


PYTHON = python3


all: run

deps:
	uv add transformers torch accelerate


run:
	uv run $(PYTHON) -m src


install:
	uv sync


index:
	uv run $(PYTHON) -m src index --max_chunk_size 2000


debug:
	uv run $(PYTHON) -m pdb -m src


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .ruff_cache


lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src


.PHONY: all run install debug lint clean index deps
