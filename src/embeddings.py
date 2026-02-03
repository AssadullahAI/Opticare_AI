import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import Config


class EmbeddingManager:
    """
    Handles sentence embeddings and FAISS index
    """

    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None

    def create_index(self, texts: list):
        if not texts:
            raise ValueError("No texts provided to create embeddings index.")

        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Convert to float32 (required by FAISS)
        embeddings_np = np.array(self.embeddings).astype("float32")

        # Create FAISS index after knowing dimension
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        # Add vectors to index
        self.index.add(embeddings_np)

    def search(self, query: str, k: int = Config.TOP_K_RESULTS):
        if self.index is None:
            raise ValueError("FAISS index not created. Run create_index(texts) first.")

        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k)
        return indices[0], distances[0]


class SemanticSearch:
    """
    Performs semantic search over embedded medical knowledge
    """

    def __init__(self, embedding_manager, texts, diseases):
        self.embedding_manager = embedding_manager
        self.texts = texts
        self.diseases = diseases

    def query(self, user_query, k=Config.TOP_K_RESULTS):
        idxs, distances = self.embedding_manager.search(user_query, k)
        results = []

        for i, d in zip(idxs, distances):
            results.append({
                "text": self.texts[i],
                "disease": self.diseases[i],
                "score": float(1 / (1 + d))
            })

        return results
