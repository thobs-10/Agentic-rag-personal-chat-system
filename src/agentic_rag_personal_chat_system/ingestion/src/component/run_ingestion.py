"""Document ingestion pipeline for processing PDFs and storing embeddings in Qdrant."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

# Import the configuration system
from agentic_rag_personal_chat_system.configs.config_factory import (
    AppConfig,
    ConfigFactory,
)
from agentic_rag_personal_chat_system.ingestion.src.component.ingestion_steps import (
    IngestionSteps,
)
from agentic_rag_personal_chat_system.ingestion.src.component.qdrant_db_client import (
    QdrantDBClient,
)
from agentic_rag_personal_chat_system.ingestion.src.component.strategy import (
    DoclingStrategy,
    LangChainStrategy,
)

# Set device to CPU as configured
device = torch.device("cpu")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DEVICE"] = "cpu"


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
        self.model = SentenceTransformer(
            self.config.model.name, device=self.config.model.device
        )

        # Initialize database client
        self.db_client = QdrantDBClient()

        # Initialize strategy
        strategies = {"langchain": LangChainStrategy(), "docling": DoclingStrategy()}
        strategy_name = self.config.pipeline.loading_strategy.lower()
        self.strategy = strategies.get(strategy_name, LangChainStrategy())

        logger.info(f"Initialized pipeline with strategy: {strategy_name}")
        logger.info(f"Using model: {self.config.model.name}")
        self.ingestion_steps = IngestionSteps(
            config=self.config, model=self.model, db_client=self.db_client
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
            # If no specific collection, process all collections from config
            self._run_all_collections()
        else:
            # Process specific collection
            self._run_single_collection(collection_name, file_paths)

    def _run_all_collections(self) -> None:
        """Run ingestion pipeline for all configured collections."""
        logger.info("Starting ingestion pipeline for all collections...")

        for collection_config in self.config.database.collections:
            collection_name = collection_config.name
            data_dir = collection_config.data_dir

            logger.info(
                f"Processing collection: {collection_name} from directory: {data_dir}"
            )

            try:
                pdf_paths = get_pdf_files(data_dir)
                if not pdf_paths:
                    logger.warning(
                        f"No PDF files found in {data_dir} for collection {collection_name}"
                    )
                    continue

                logger.info(
                    f"Found {len(pdf_paths)} PDF files for collection {collection_name}"
                )
                self._run_single_collection(collection_name, pdf_paths)

            except Exception as e:
                logger.error(f"Failed to process collection {collection_name}: {e}")
                continue

        logger.info("Completed ingestion pipeline for all collections")

    def _process_documents(
        self, docs: Any, file_paths: List[str], collection_name: str
    ) -> List[List[Dict[str, Any]]]:
        """Process documents to extract text and create chunks.

        Args:
            docs: List of document dictionaries from document conversion
            file_paths: List of original file paths
        Returns:
            Processed document data
        """
        pages_data = self.ingestion_steps.extract_text(docs, file_paths)
        chunks = self.ingestion_steps.chunk_text(pages_data)
        logger.info(
            f"Created {sum(len(doc_chunks) for doc_chunks in chunks)} chunks from {len(pages_data)} documents for collection {collection_name}"
        )
        return chunks

    def _run_single_collection(
        self, collection_name: str, file_paths: List[str]
    ) -> None:
        """Run ingestion pipeline for a single collection.

        Args:
            collection_name: Name of the collection to process
            file_paths: List of PDF file paths to process
        """
        logger.info(
            f"Starting document ingestion pipeline for collection: {collection_name}..."
        )
        try:
            if collection_name and file_paths:
                # 1. Load documents using configured strategy
                docs = self.strategy.load_documents(file_paths)
                logger.debug(
                    f"Loaded {len(docs)} documents for collection {collection_name}"
                )

                # 2. Process documents: extract text and create chunks
                chunks = self._process_documents(docs, file_paths, collection_name)

                # 3. Generate embeddings and store in database
                if self.config.database.recreate_collection:
                    self.db_client.remove_collection(collection_name)
                    logger.info(f"Recreated collection: {collection_name}")

                self.ingestion_steps.create_collection(collection_name)
                embeddings_and_metadata = self.ingestion_steps.generate_embeddings(
                    chunks
                )
                self.ingestion_steps.insert_into_qdrant(
                    collection_name, embeddings_and_metadata
                )

                logger.info(
                    f"Document ingestion pipeline completed successfully for collection {collection_name}"
                )

        except ValueError as e:
            logger.error(
                f"Pipeline failed with validation error for collection {collection_name}: {e}"
            )
            raise
        except KeyError as e:
            logger.error(
                f"Pipeline failed with configuration error for collection {collection_name}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Pipeline failed with unexpected error for collection {collection_name}: {e}",
                exc_info=True,
            )
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
    if not config.database.collections:
        logger.error("No collections configured.")
        return False
    for collection in config.database.collections:
        if not collection.name:
            logger.error("Collection name is not configured.")
            return False
        if not collection.data_dir:
            logger.error(
                f"Data directory not configured for collection {collection.name}"
            )
            return False
    if not config.model.name:
        logger.error("Model name is not configured.")
        return False
    logger.info("Configuration validated successfully.")
    return True


def main(config_path: Optional[Path] = None) -> None:
    """Entry point for the ingestion pipeline."""

    ConfigFactory.initialize(config_path)
    config = ConfigFactory.get_config()

    if not validate_config(config):
        logger.error("Invalid configuration. Exiting.")
        sys.exit(1)

    logger.info("Starting ingestion pipeline for all configured collections")
    pipeline = IngestionPipeline(config)
    pipeline.run_pipeline([])

    logger.info("Ingestion pipeline completed for all collections")


if __name__ == "__main__":
    default_config_path = (
        Path(__file__).parent.parent.parent.parent.parent.parent / "config.yaml"
    )
    main(default_config_path)
