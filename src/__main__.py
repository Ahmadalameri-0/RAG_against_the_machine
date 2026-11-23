import functools
import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import fire
from pydantic import BaseModel
from tqdm import tqdm
from . import CLI
from .indexer import Indexer
from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from .retriever import BM25Retriever


INDEX_PATH = "data/processed/index.json"
MAX_CHUNK_SIZE = 2000
RECALL_KS = (1, 3, 5, 10)
IOU_THRESHOLD = 0.05


def _safe(func: Callable[..., None]) -> Callable[..., None]:
    """Turn any exception of a command into a clean error message."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            CLI.warn("Interrupted.")
            sys.exit(130)
        except Exception as exc:
            CLI.error(str(exc))
            sys.exit(1)
    return wrapper


def _read(path: str) -> str:
    """Read a UTF-8 text file (raises on missing or unreadable file)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _save(model: BaseModel, directory: str, source: str, label: str) -> None:
    """Write a model as JSON into directory named like source."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, os.path.basename(source))
    with open(path, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))
    CLI.ok(f"Saved {label}", path)


def _check_k(k: Any) -> None:
    """Reject a k that is not a non-negative integer."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")


def _check_query(query: Any) -> str:
    """Return the query as a string, rejecting empty queries."""
    text = str(query).strip()
    if not text:
        raise ValueError("query must not be empty")
    return text


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character ranges of the same file."""
    if a.file_path != b.file_path:
        return 0.0
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    len_a = a.last_character_index - a.first_character_index
    len_b = b.last_character_index - b.first_character_index
    return inter / (len_a + len_b - inter)


class RAGCLI:
    """RAG pipeline: index, search, answer and evaluate."""

    @_safe
    def index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Ingest raw_dir and bCLIld the index under processed_dir."""
        if (
            isinstance(max_chunk_size, bool)
            or not isinstance(max_chunk_size, int)
            or not 0 < max_chunk_size <= MAX_CHUNK_SIZE
        ):
            raise ValueError(
                f"max_chunk_size must be between 1 and {MAX_CHUNK_SIZE}"
            )
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
        Indexer(raw_dir, processed_dir).index(max_chunk_size)
        CLI.ok("Ingestion complete", f"{processed_dir}/index.json")

    @_safe
    def search(
        self, query: str, k: int = 5, index_path: str = INDEX_PATH
    ) -> None:
        """Print the top-k sources of a single query."""
        text = _check_query(query)
        _check_k(k)
        result = BM25Retriever(index_path).search("query", text, k)
        CLI.search_result(text, result.search_results[0].retrieved_sources)

    @_safe
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_path: str = INDEX_PATH,
    ) -> None:
        """Search every question of a dataset and save the results."""
        _check_k(k)
        dataset = RagDataset.model_validate_json(_read(dataset_path))
        questions = dataset.rag_questions
        retriever = BM25Retriever(index_path)
        results = [
            retriever.search(q.question_id, q.question, k).search_results[0]
            for q in tqdm(
                questions, desc="Searching", bar_format=CLI.TQDM_FMT
            )
        ]
        _save(
            StudentSearchResults(search_results=results, k=k),
            save_directory,
            dataset_path,
            f"student_search_results ({len(results)} queries)",
        )

    @_safe
    def answer(
        self, query: str, k: int = 5, index_path: str = INDEX_PATH
    ) -> None:
        """Print the generated answer of a single query."""
        text = _check_query(query)
        _check_k(k)
        found = BM25Retriever(index_path).search("query", text, k)
        from .generator import QwenGenerator

        out = QwenGenerator().generate(found).search_results[0]
        CLI.answer_result(out.question, out.answer, out.retrieved_sources)

    @_safe
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for saved search results and save them."""
        found = StudentSearchResults.model_validate_json(
            _read(student_search_results_path)
        )
        from .generator import QwenGenerator

        answered = QwenGenerator().generate(found)
        _save(
            answered,
            save_directory,
            student_search_results_path,
            f"answers ({len(answered.search_results)} queries)",
        )

    @_safe
    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print a Recall@k report (own testing only)."""
        student = StudentSearchResults.model_validate_json(
            _read(student_search_results_path)
        )
        truth: Dict[str, List[MinimalSource]] = {
            q.question_id: q.sources
            for q in RagDataset.model_validate_json(
                _read(dataset_path)
            ).rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }
        rows = [r for r in student.search_results if r.question_id in truth]
        if not rows:
            raise ValueError("no question_id shared with the ground truth")

        ks = [n for n in RECALL_KS if n <= student.k] or [student.k]
        results: List[Tuple[int, float]] = []
        for n in ks:
            total = 0.0
            for row in rows:
                refs = truth[row.question_id]
                top = row.retrieved_sources[:n]
                total += sum(
                    any(_iou(s, ref) > IOU_THRESHOLD for s in top)
                    for ref in refs
                ) / len(refs)
            results.append((n, total / len(rows)))

        name = os.path.basename(dataset_path)
        CLI.recall_report(
            name,
            len(rows),
            sum(1 for r in rows if r.retrieved_sources),
            student.k,
            results,
            0.50 if "code" in name.lower() else 0.80,
            MAX_CHUNK_SIZE,
            IOU_THRESHOLD,
        )


if __name__ == "__main__":
    fire.Fire(RAGCLI)
