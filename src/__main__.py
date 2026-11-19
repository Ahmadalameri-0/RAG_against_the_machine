import fire
from .indexer import Indexer
from .retriever import BM25Retriever


class RAGCLI:
    """
    RAG-against-the-machine CLI interface for 42 School.
    Implements lexical indexing, BM25 retrieval, and generation pipelines.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """Reads raw files, chunks them, and builds the local JSON index."""
        indexer = Indexer(
            raw_dir="data/raw",
            processed_dir="data/processed",
            max_chunk_size=max_chunk_size
        )
        indexer.build_index()

    def search(self, question_id: str, query: str, top_k: int = 5) -> None:
        """Performs lexical BM25 search and prints
        the strict Pydantic JSON output."""
        retriever = BM25Retriever(
            index_path="data/processed/index.json"
        )
        result = retriever.search(
            question_id=question_id, query=query, top_k=top_k
        )
        print(result.model_dump_json(indent=4))

    def answer(self, question_id: str, query: str, top_k: int = 5) -> None:
        """Performs search AND generates an answer using Qwen"""


if __name__ == "__main__":
    fire.Fire(RAGCLI)
