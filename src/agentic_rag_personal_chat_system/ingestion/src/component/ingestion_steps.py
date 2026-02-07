from typing import Any, Dict, List

from loguru import logger
from sentence_transformers import SentenceTransformer

from agentic_rag_personal_chat_system.ingestion.src.main_utils.utils import (
    get_splitter_object,
    process_page,
)


class IngestionSteps:
    """
    This class defines the steps for the ingestion process, including loading documents,
    splitting them into chunks, generating embeddings, and saving the results.
    """

    def __init__(self, config: Any, model: SentenceTransformer, db_client: Any) -> None:
        self.config = config
        self.model = model
        self.db_client = db_client

    def extract_text(
        self,
        docs: List[Dict[str, Any]],
        file_paths: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """Extract text content page by page from documents.

        Args:
            docs: List of document dictionaries from document conversion
            file_paths: List of original file paths

        Returns:
            List of lists containing extracted text and metadata for each document
        """
        extracted: List[List[Dict[str, Any]]] = []

        for doc in docs:
            content = doc.get("content") or doc.get("pages")
            metadata = doc.get("metadata", {})
            page_number = metadata.get("page_label", None)
            file_path = metadata.get("source", None)

            if content and file_path:
                extracted.append(
                    [
                        {
                            "content": content,
                            "metadata": {
                                "page_number": page_number,
                                "file_path": file_path,
                            },
                        }
                    ]
                )
                logger.info(f"Extracted {len(content)} pages with metadata {metadata}")
            else:
                extracted.append([])
                logger.warning(f"Missing content or file_path in document metadata: {metadata}")

        return extracted

    def chunk_text(self, pages_data: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Split extracted text into chunks for embedding.

        Args:
            pages_data: List of document pages with text and metadata

        Returns:
            List of document chunks, each chunk containing text and metadata
        """
        # Get splitter with configured parameters
        splitter = get_splitter_object(
            chunk_size=self.config.text.chunk_size,
            chunk_overlap=self.config.text.chunk_overlap,
        )

        all_chunks = []
        chunk_id_counter: int = 0

        for pages in pages_data:
            doc_chunks = []
            for page_data in pages:
                page_chunks = process_page(page_data, chunk_id_counter, splitter)
                doc_chunks.extend(page_chunks)
                chunk_id_counter += len(page_chunks)
            all_chunks.append(doc_chunks)

        return all_chunks

    def generate_embeddings(
        self,
        all_chunks: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Generate vector embeddings for each chunk.

        Args:
            all_chunks: List of document chunks to generate embeddings for

        Returns:
            Dictionary containing embeddings and metadata
        """
        batch_size = self.config.model.batch_size

        # Generate embeddings and metadata
        all_embeddings = []
        all_metadata = []

        for chunks in all_chunks:
            # Generate embeddings using configured model
            chunk_embeddings = self.model.encode(
                sentences=[chunk["text"] for chunk in chunks],
                show_progress_bar=False,
                batch_size=batch_size,
            )

            # Convert numpy arrays to nested lists for JSON serialization
            embeddings_list = [embedding.tolist() for embedding in chunk_embeddings]
            all_embeddings.append(embeddings_list)

            # Extract metadata
            chunk_metadata = [
                {
                    **chunk["metadata"],
                    "text": chunk["text"],
                }
                for chunk in chunks
            ]
            all_metadata.append(chunk_metadata)

        return {"embeddings": all_embeddings, "metadata": all_metadata}

    def create_collection(self, collection_name: str) -> None:
        """Creates Qdrant collection if not exists."""
        if not collection_name:
            raise ValueError("Collection name cannot be empty")

        # Use configured vector size or get from model
        vector_size = self.config.model.vector_size
        if not vector_size:
            vector_size = self.model.get_sentence_embedding_dimension()

        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError(f"Invalid vector size: {vector_size}")

        # Create collection with configured distance metric
        self.db_client.create_db_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=self.config.database.distance,
        )

    def insert_into_qdrant(
        self,
        collection_name: str,
        embeddings_metadata: Dict[str, Any],
    ) -> None:
        """Insert embeddings and metadata into Qdrant collection.

        Args:
            collection_name: Name of the Qdrant collection
            embeddings_metadata: Dictionary containing embeddings and metadata
        """
        embeddings = embeddings_metadata["embeddings"]
        metadata = embeddings_metadata["metadata"]

        if not embeddings:
            logger.info("No points to insert into Qdrant.")
            return

        # Process each document's chunks
        for doc_embeddings, doc_metadata in zip(embeddings, metadata, strict=True):
            self.db_client.insert_embeddings(
                collection_name=collection_name, embeddings=doc_embeddings, metadata=doc_metadata
            )

        total_chunks = sum(len(doc_embeddings) for doc_embeddings in embeddings)
        logger.info(
            f"Inserted {total_chunks} chunks from {len(embeddings)} documents into collection '{collection_name}'."
        )
