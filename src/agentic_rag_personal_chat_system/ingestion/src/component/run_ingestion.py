"""Document ingestion pipeline for processing PDFs and storing embeddings in Qdrant."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

# Set device to CPU as configured
device = torch.device("cpu")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DEVICE"] = "cpu"

from sentence_transformers import SentenceTransformer
from loguru import logger

from agentic_rag_personal_chat_system.ingestion.src.component.qdrant_db_client import (
    QdrantDBClient,
)
from agentic_rag_personal_chat_system.ingestion.src.main_utils.utils import (
    process_page,
    get_splitter_object,
)
from agentic_rag_personal_chat_system.ingestion.src.component.strategy import (
    LangChainStrategy,
    DoclingStrategy,
)

# Import the configuration system
from agentic_rag_personal_chat_system.configs.config_factory import ConfigFactory, AppConfig


class IngestionPipeline:
    """Main ingestion pipeline class that uses configuration from YAML."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or ConfigFactory.get_config()
        # self.logger = logger
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Setup pipeline components based on configuration."""
        # Initialize model
        self.model = SentenceTransformer(self.config.model.name, device=self.config.model.device)

        # Initialize database client
        self.db_client = QdrantDBClient()

        # Initialize strategy
        strategies = {"langchain": LangChainStrategy(), "docling": DoclingStrategy()}
        strategy_name = self.config.pipeline.loading_strategy.lower()
        self.strategy = strategies.get(strategy_name, LangChainStrategy())

        logger.info(f"Initialized pipeline with strategy: {strategy_name}")
        logger.info(f"Using model: {self.config.model.name}")
        logger.info(f"Data directory: {self.config.pipeline.data_dir}")

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

    def run_pipeline(
        self,
        file_paths: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Run the complete document ingestion pipeline.

        Args:
            file_paths: List of PDF file paths to process
            collection_name: Name of the Qdrant collection to use (uses config if None)

        Raises:
            Exception: If any pipeline step fails
        """
        if collection_name is None:
            collection_name = self.config.database.collection_name

        logger.info("Starting document ingestion pipeline...")

        try:
            if collection_name:
                # 1. Load documents using configured strategy
                docs = self.strategy.load_documents(file_paths)
                logger.debug(f"Loaded {len(docs)} documents")

                # 2. Process documents: extract text and create chunks
                pages_data = self.extract_text(docs, file_paths)
                chunks = self.chunk_text(pages_data)
                logger.info(
                    f"Created {sum(len(doc_chunks) for doc_chunks in chunks)} chunks from {len(pages_data)} documents"
                )

                # 3. Generate embeddings and store in database
                if self.config.database.recreate_collection:
                    self.db_client.remove_collection(collection_name)
                    logger.info(f"Recreated collection: {collection_name}")

                self.create_collection(collection_name)
                embeddings_and_metadata = self.generate_embeddings(chunks)
                self.insert_into_qdrant(collection_name, embeddings_and_metadata)

                logger.info("Document ingestion pipeline completed successfully")

        except ValueError as e:
            logger.error(f"Pipeline failed with validation error: {e}")
            raise
        except KeyError as e:
            logger.error(f"Pipeline failed with configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
            raise


def get_pdf_files(data_dir: str) -> List[str]:
    """Get list of PDF files from directory.

    Args:
        data_dir: Directory to scan for PDF files

    Returns:
        List of absolute paths to PDF files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {data_dir}")
        return []

    pdf_paths = [str(p) for p in data_path.rglob("*.pdf")]
    return pdf_paths


def validate_config(config: AppConfig) -> bool:
    """validate required configuration values"""
    if not config.database.url:
        logger.error("Database URL is not configured.")
        return False
    if not config.database.collection_name:
        logger.error("Database collection name is not configured.")
        return False
    if not config.model.name:
        logger.error("Model name is not configured.")
        return False
    logger.info("Configuration validated successfully.")
    return True


def main(config_path: Optional[Path] = None) -> List[str]:
    """Entry point for the ingestion pipeline."""

    # Initialize configuration
    ConfigFactory.initialize(config_path)
    config = ConfigFactory.get_config()

    # Setup logging
    # setup_logging(config.logging)
    if not validate_config(config):
        logger.error("Invalid configuration. Exiting.")
        sys.exit(1)

    # Scan for PDF files using configured data directory
    pdf_paths = get_pdf_files(config.pipeline.data_dir)

    if not pdf_paths:
        logger.warning(f"No PDF files found in {config.pipeline.data_dir}")
        return []

    logger.info(f"Found {len(pdf_paths)} PDF files for processing")
    return pdf_paths


def run_ingestion_pipeline(
    file_paths: List[str],
    collection_name: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> None:
    """Run the complete ingestion pipeline with configuration.

    Args:
        file_paths: List of PDF file paths to process
        collection_name: Optional collection name (uses config if None)
        config_path: Optional path to config file (uses default if None)
    """
    # Initialize configuration if not already done
    if config_path:
        ConfigFactory.initialize(config_path)

    config = ConfigFactory.get_config()

    # Create and run pipeline
    pipeline = IngestionPipeline(config)
    pipeline.run_pipeline(file_paths, collection_name)


def default_main() -> List[str]:
    """Default entry point that looks for config in standard location."""
    default_config_path = Path(__file__).parent.parent.parent.parent.parent.parent / "config.yaml"
    return main(default_config_path)


if __name__ == "__main__":
    # Get PDF files using configuration
    pdf_paths = default_main()

    if not pdf_paths:
        logger.error("No PDF files to process. Exiting.")
        sys.exit(0)

    # Run the pipeline with configured settings
    run_ingestion_pipeline(pdf_paths)
