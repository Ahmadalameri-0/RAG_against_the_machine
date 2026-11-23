export UV_PROJECT_ENVIRONMENT ?= /goinfre/$(USER)/RAG
export VIRTUAL_ENV ?= $(UV_PROJECT_ENVIRONMENT)
export UV_CACHE_DIR ?= /goinfre/$(USER)/uv_cache
export HF_HOME ?= /goinfre/$(USER)/hf_cache
export UV_LINK_MODE = copy

CYAN  := \033[1;36m
GREEN := \033[1;32m
RED   := \033[1;31m
RESET := \033[0m

PYTHON = python3


all: run

deps:
	uv add transformers torch


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


SRC = uv run $(PYTHON) -m src
QUERY = "How to configure the OpenAI server?"
DOCS_Q = data/datasets/UnansweredQuestions/dataset_docs_public.json
CODE_Q = data/datasets/UnansweredQuestions/dataset_code_public.json
DOCS_A = data/datasets/AnsweredQuestions/dataset_docs_public.json
CODE_A = data/datasets/AnsweredQuestions/dataset_code_public.json
OUT_S = data/output/search_results/UnansweredQuestions
OUT_A = data/output/search_results_and_answer/UnansweredQuestions


DIM := \033[2m

define step
	@printf "\n$(CYAN)━━ [$(1)] $(2) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@s=$$(python3 -c "import time; print(time.time())"); $(3); rc=$$?; \
	e=$$(python3 -c "import time; print(time.time())"); \
	t=$$(awk -v a=$$s -v b=$$e 'BEGIN{printf "%.2f", b-a}'); \
	if [ $$rc -eq 0 ]; then printf "$(GREEN)✔ PASS$(RESET) $(DIM)$(2) in $${t}s$(RESET)\n"; \
	else printf "$(RED)✖ FAIL$(RESET) $(DIM)$(2) after $${t}s$(RESET)\n"; exit $$rc; fi
endef


test:
	$(call step,1,index,$(SRC) index --max_chunk_size 2000)
	$(call step,2,search,$(SRC) search $(QUERY) --k 5)
	$(call step,3,search_dataset [docs],$(SRC) search_dataset --dataset_path $(DOCS_Q) --k 10 --save_directory $(OUT_S))
	$(call step,4,search_dataset [code],$(SRC) search_dataset --dataset_path $(CODE_Q) --k 10 --save_directory $(OUT_S))
	$(call step,5,answer,$(SRC) answer $(QUERY) --k 5)
	$(call step,6,answer_dataset [docs],$(SRC) answer_dataset --student_search_results_path $(OUT_S)/dataset_docs_public.json --save_directory $(OUT_A))
	$(call step,7,answer_dataset [code],$(SRC) answer_dataset --student_search_results_path $(OUT_S)/dataset_code_public.json --save_directory $(OUT_A))
	$(call step,8,evaluate [docs],$(SRC) evaluate --student_search_results_path $(OUT_S)/dataset_docs_public.json --dataset_path $(DOCS_A))
	$(call step,9,evaluate [code],$(SRC) evaluate --student_search_results_path $(OUT_S)/dataset_code_public.json --dataset_path $(CODE_A))
	@printf "\n$(GREEN)All checks finished.$(RESET)\n"


eval:
	@s=$$(python3 -c "import time; print(time.time())"); \
	$(SRC) search_dataset --dataset_path $(DOCS_Q) --k 10 --save_directory $(OUT_S) && \
	$(SRC) search_dataset --dataset_path $(CODE_Q) --k 10 --save_directory $(OUT_S); \
	rc=$$?; e=$$(python3 -c "import time; print(time.time())"); \
	t=$$(awk -v a=$$s -v b=$$e 'BEGIN{printf "%.2f", b-a}'); \
	printf "\n$(CYAN)━━ Retrieval Throughput (199 queries) ━━$(RESET)\n"; \
	if [ $$rc -eq 0 ]; then printf "$(GREEN)✔ PASS$(RESET) $${t}s <= 90.00s\n\n"; \
	else printf "$(RED)✖ FAIL$(RESET) after $${t}s\n"; exit $$rc; fi
	$(call step,1,evaluate [docs],$(SRC) evaluate --student_search_results_path $(OUT_S)/dataset_docs_public.json --dataset_path $(DOCS_A))
	$(call step,2,evaluate [code],$(SRC) evaluate --student_search_results_path $(OUT_S)/dataset_code_public.json --dataset_path $(CODE_A))
	
	@printf "\n$(GREEN)All evaluations finished.$(RESET)\n"


.PHONY: all run install debug lint clean index deps test eval
