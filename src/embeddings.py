from __future__ import annotations

import hashlib
import math

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        try:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            if isinstance(text, str):
                return [float(value) for value in response.data[0].embedding]
            return [[float(value) for value in d.embedding] for d in response.data]
        except Exception as e:
            print(f"Lỗi OpenAI API: {e}")
            dim = 1536 # Default OpenAI embedding dim (small 3)
            return [0.0] * dim if isinstance(text, str) else [[0.0] * dim for _ in text]


class LMStudioEmbedder:
    """LM Studio local server API-backed embedder using OpenAI client."""

    def __init__(self, model_name: str = "jina-embeddings-v5-text-nano-retrieval") -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = f"LMStudio: {model_name}"
        # Trỏ base_url thẳng xuống LM Studio đang bật
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        try:
            # Truyền mảng batch input hoặc single input
            response = self.client.embeddings.create(model=self.model_name, input=text)
            if isinstance(text, str):
                return [float(value) for value in response.data[0].embedding]
            return [[float(value) for value in d.embedding] for d in response.data]
        except Exception as e:
            print(f"[!] Lỗi kết nối tới LM Studio Server: {e}")
            print("[!] Vui lòng bật LM Studio và khởi động Local Server tại port 1234.")
            dim = 512 # Thường dòng Nano của Jina hoặc Nomic mặc định chiều 512 hoặc 768
            return [0.0] * dim if isinstance(text, str) else [[0.0] * dim for _ in text]


_mock_embed = MockEmbedder()
