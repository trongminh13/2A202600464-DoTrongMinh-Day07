from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        results = self.store.search(question, top_k=top_k)
        
        # 2. Build a prompt with the chunks as context
        context_text = "\n\n".join([r["content"] for r in results])
        
        prompt = f"""Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context_text}

Question: {question}

Answer:"""
        
        # 3. Call the LLM to generate an answer
        return self.llm_fn(prompt)

