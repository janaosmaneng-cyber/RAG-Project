import chromadb
from semantic_router.encoders import HuggingFaceEncoder

from app.core.config import settings, load_pipeline_config
from app.utils.logging_config import logger


class RetrievalService:
    """Loads the embedding model and Chroma collection once at startup."""

    def __init__(self):
        self.pipeline_config = load_pipeline_config()

        logger.info(
            "Loading embedding model: %s",
            self.pipeline_config["embedding_model"],
        )

        # This is the same encoder used in the notebook.
        self.encoder = HuggingFaceEncoder(
            name=self.pipeline_config["embedding_model"]
        )

        logger.info(
            "Loading Chroma vector store from: %s",
            settings.vector_store_path,
        )

        client = chromadb.PersistentClient(
            path=settings.vector_store_path
        )

        self.collection = client.get_collection(
            name=self.pipeline_config["vector_store"]["collection_name"]
        )

        logger.info(
            "Retrieval service ready. Collection contains %d chunks.",
            self.collection.count(),
        )

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
    ):
        """Retrieve the most relevant chunks for a question."""

        # Generate an embedding for the user's question.
        query_embedding = self.encoder(
            docs=[query],
            normalize_embeddings=True,
        )

        # Search Chroma for the most similar chunks.
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        return results


retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global retrieval_service

    if retrieval_service is None:
        retrieval_service = RetrievalService()

    return retrieval_service