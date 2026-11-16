*This activity has been created as part of the 42 curriculum by abani-am*

---
## System Architecture
```mermaid
graph TD
%%{init: {'flowchart': {'nodeSpacing': 15, 'rankSpacing': 15}, 'themeVariables': {'fontSize': '8px'}}}%%
    %% 1. (Indexing Phase)
    subgraph Indexing [1. Indexing Phase]
        A[📄 Raw Files / Corpus] -->|Parse & Split| B[✂️ Indexer / Chunker]
        B -->|Save| C[(🗂️ Local Index <br/> processed/index.json)]
    end

    %% 2.(Retrieval Phase)
    subgraph Retrieval [2. Retrieval Phase]
        Q[👤 User Query] -->|Search| D{🔍 Lexical Retriever <br/> BM25 / TF-IDF}
        C -.->|Match Words| D
        D -->|Extract Top-K| E[📑 Extracted Context / Chunks]
    end

    %% 3. (Generation Phase)
    subgraph Generation [3. Generation Phase]
        Q -->|Prompt| F[🧠 LLM <br/> Qwen / Qwen3-0.6B]
        E -->|Context| F
        F -->|Generate Answer| G[✅ Final Output <br/> JSON Answer]
    end

    %% Colors
    style A fill:#e1f5fe,stroke:#0288d1,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#e8f5e9,stroke:#388e3c,color:#000
    style Q fill:#e1f5fe,stroke:#0288d1,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style G fill:#ffebee,stroke:#d32f2f,color:#000
```


## Description

**RAG (Retrieval-Augmented Generation) lets a language model answer from an external source instead of relying only on its training data. This project indexes the vLLM repository, retrieves the most relevant snippets for a question (BM25), and feeds them to Qwen3-0.6B to generate a grounded answer.**

**How Does RAG work?**
*(Example: Customer Support)*
1. **Data Collection**
2. **Data chunking**
3. **Document embeddings & load data into vector database**
4. **Handling user queries:** Once the query is converted into an embedding, the system compares the query embedding with the document embeddings. It identifies and retrieves chunks whose embeddings are most similar to the query embedding, using measures such as cosine similarity

5. **Generating responses with an LLM:** The retrieved text chunks, along with the initial user query, are fed into a language model.

## Resources
- [NOTED] - (https://youtu.be/LIFyBPDkluI?si=-wSRP-grv_I1HQBc)
- [NOTED] - (https://youtu.be/sNLbKNMa6zI?si=QJkowz_fW-QmaxDO)
- [NOTED **Theoretical**] (https://youtu.be/rcovF6S1oFI?si=epbMj3qHqUZ0Bx4J)
- [NOTED **TF-IDF**] - (https://youtu.be/8Q0IdDI7yeE?si=EFxhLXiEeCWRTTqV)
- [fifth lesson **BM25**] - (https://youtu.be/YL-3G5-xVYU?si=Oxl_KMnYuDAdMoOU)

- [Full course on RAG](https://youtube.com/playlist?list=PLvLvlVqNQGHCUR2p0b8a0QpVjDUg50wQj&si=3JDbYtnrhWURKFTE)
