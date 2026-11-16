import json
import math
import re
from typing import List, Dict, Any
from pathlib import Path

# استيراد النماذج التي بنيناها سابقاً لضمان صحة المخرجات
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults

class BM25Retriever:
    """
    A Lexical Search Retriever implementing the Okapi BM25 algorithm.
    It scores and ranks documents based on term frequency and inverse document frequency,
    without relying on external Vector Databases or Embeddings.
    """

    def __init__(self, index_path: str = "data/processed/index.json"):
        """
        Initializes the retriever, loads the indexed chunks, and precomputes
        necessary BM25 statistics (IDF, document lengths, etc.).
        """
        self.k1 = 1.5  # Controls term frequency scaling (standard value: 1.2 to 2.0)
        self.b = 0.75  # Controls document length normalization (standard value: 0.75)
        
        self.chunks = self._load_index(index_path)
        self.total_documents = len(self.chunks)
        self.average_document_length = 0.0
        
        self.document_frequencies = {} # How many documents contain a specific word
        self.tokenized_corpus = []     # Pre-tokenized chunks for faster lookup
        
        self._prepare_statistics()

    def _load_index(self, path: str) -> List[Dict[str, Any]]:
        """Loads the indexed JSON file containing all text chunks."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _tokenize(self, text: str) -> List[str]:
        """
        Converts text to lowercase and extracts alphabetic words.
        Simple yet effective for lexical matching.
        """
        return re.findall(r'\b\w+\b', text.lower())

    def _prepare_statistics(self) -> None:
        """
        Precomputes the Document Frequency (DF) for each unique word
        and calculates the average document length across the entire corpus.
        """
        total_length = 0
        
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.get("text", ""))
            self.tokenized_corpus.append(tokens)
            total_length += len(tokens)
    
            # Count unique words in this specific chunk
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
                
        if self.total_documents > 0:
            self.average_document_length = total_length / self.total_documents

    def _calculate_idf(self, term: str) -> float:
        """
        Calculates the Inverse Document Frequency (IDF) for a specific term.
        Words that appear in many documents get a lower score, and rare words get a higher score.
        """
        doc_frequency = self.document_frequencies.get(term, 0)
        if doc_frequency == 0:
            return 0.0
        
        # Standard BM25 IDF formula
        numerator = self.total_documents - doc_frequency + 0.5
        denominator = doc_frequency + 0.5
        return math.log(1 + (numerator / denominator))

    def search(self, question_id: str, query: str, top_k: int = 5) -> StudentSearchResults:
        """
        Searches the corpus for the query and returns the top_k most relevant chunks,
        formatted strictly as the required Pydantic models.
        """
        query_tokens = self._tokenize(query)
        scores = []

        # 1. Calculate BM25 score for every chunk in the corpus
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            doc_score = 0.0
            doc_length = len(doc_tokens)
            
            for token in query_tokens:
                if token not in doc_tokens:
                    continue
                    
                # Term Frequency (TF): How many times the word appears in THIS chunk
                term_frequency = doc_tokens.count(token)
                
                # Inverse Document Frequency (IDF): The rarity of the word globally
                idf = self._calculate_idf(token)
                
                # Length Normalization: Penalize very long chunks
                length_penalty = 1 - self.b + (self.b * (doc_length / self.average_document_length))
                
                # Final BM25 Term Score Math
                tf_numerator = term_frequency * (self.k1 + 1)
                tf_denominator = term_frequency + (self.k1 * length_penalty)
                
                doc_score += idf * (tf_numerator / tf_denominator)
                
            scores.append((doc_score, self.chunks[idx]))

        # 2. Sort chunks by score (highest first) and take the Top K
        scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item[1] for item in scores[:top_k]]

        # 3. Format output using our strict Pydantic Models
        retrieved_sources = [
            MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"]
            )
            for chunk in top_chunks
        ]

        minimal_result = MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources
        )

        return StudentSearchResults(
            search_results=[minimal_result],
            k=top_k
        )
