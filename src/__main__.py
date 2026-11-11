import fire
from .Indexer import Indexer

class RAG_CLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(
            raw_dir="data/raw",
            processed_dir="data/processed",
            max_chunk_size=max_chunk_size
        )
        indexer.build_index()


if __name__ == "__main__":
    fire.Fire(RAG_CLI)
