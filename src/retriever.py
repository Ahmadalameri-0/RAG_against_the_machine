import json
import math
import heapq
from .indexer import Indexer
from typing import List, Dict, Any
from .models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults
)


class BM25Retriever:
    """Fast lexical BM25 retriever backed by a prebuilt inverted index."""
    def __init__(
            self,
            index_path: str = "data/processed/index.json",
            k1: float = 2,
            b: float = 0.75
    ) -> None:
        """Load inverted index and precompute IDF statistics."""
        self.index_path: str = index_path
        self.k1: float = k1
        self.b: float = b
        self.chunks: List[Dict[str, Any]] = []
        self.inverted_index: Dict[str, List[List[int]]] = {}
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}

        self._load_index()
        self._prepare_statistics()

    def _load_index(self) -> None:
        """Load chunks metadata and inverted index from JSON file."""
        with open(self.index_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        self.chunks = data.get("chunks", [])
        self.inverted_index = data.get("inverted_index", {})
        self.num_docs = int(data.get("num_docs", len(self.chunks)))
        self.avg_doc_len = float(
            data.get("avg_doc_len", 0.0)
        )

    def _prepare_statistics(self) -> None:
        """Precompute BM25 IDF values for every indexed term."""
        num = self.num_docs
        for term, postings in self.inverted_index.items():
            docs_frequency = len(postings)
            self.idf[term] = math.log(
                (num - docs_frequency + 0.5) / (docs_frequency + 0.5) + 1.0
            )

    def search(
            self, question_id: str, query: str, top_k: int = 5
    ) -> StudentSearchResults:
        """Rank chunks with BM25 over the inverted index and return top-k"""
        if (
            top_k <= 0 or self.num_docs == 0
            or self.avg_doc_len == 0.0
        ):
            return StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id=question_id,
                        question=query,
                        retrieved_sources=[]
                    )
                ],
                k=max(0, top_k),
            )

        tokens: List[str] = Indexer.tokenize(query)
        scores: Dict[int, float] = {}

        for term in tokens:
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_id, term_frequency in postings:
                doc_len = int(self.chunks[doc_id]["length"])
                length_penalty = (
                    1.0 - self.b + self.b * (
                        doc_len / self.avg_doc_len
                    )
                )
                denominator = term_frequency + self.k1 * length_penalty
                score = idf * (
                    (term_frequency * (self.k1 + 1.0)) / denominator
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        top_docs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        sources: List[MinimalSource] = [
            MinimalSource(
                file_path=str(self.chunks[doc_id]["file_path"]),
                first_character_index=int(
                    self.chunks[doc_id]["first_character_index"]
                ),
                last_character_index=int(
                    self.chunks[doc_id]["last_character_index"]
                ),
            )
            for doc_id, _ in top_docs
        ]

        return StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=question_id,
                    question=query,
                    retrieved_sources=sources
                )
            ],
            k=top_k
        )
