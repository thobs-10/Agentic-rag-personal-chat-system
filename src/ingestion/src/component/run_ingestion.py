"""Document ingestion pipeline for processing PDFs and storing embeddings in Qdrant."""

import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.src.component.qdrant_db_client import QdrantDBClient
from src.ingestion.src.config import QdrantDBConfig
from zenml.pipelines import pipeline
from zenml.steps import step


def _extract_text_from_attribute(page: Any, attr_name: str) -> Optional[str]:
    """Extract text using a specific attribute method.

    Args:
        page: Document page object
        attr_name: Name of the attribute to try

    Returns:
        Extracted text if successful, None otherwise
    """
    if not hasattr(page, attr_name):
        return None

    try:
        attr = getattr(page, attr_name)
        result = attr() if callable(attr) else attr

        if isinstance(result, str):
            text = result.strip()
            return text if text and not text.startswith("<") else None
        elif isinstance(result, (list, tuple)):
            text = "\n".join(str(item) for item in result if item)
            return text.strip() if text else None
        elif result is not None:
            return str(result).strip()
    except Exception as e:
        logger.debug(f"Attribute {attr_name} extraction failed: {str(e)}")
    return None


def extract_page_text(page: Any) -> Optional[str]:
    """Try different methods to extract text from a page.

    Args:
        page: Document page object

    Returns:
        Extracted text if successful, None otherwise
    """
    logger.debug(f"Extracting text from page of type: {type(page)}")

    for method_name in ["text", "export_to_text", "content", "texts", "get_text"]:
        if text := _extract_text_from_attribute(page, method_name):
            return text
    return None


def convert_documents(file_paths: List[str]) -> List[Any]:
    """Convert input PDF files to Docling documents.

    Args:
        file_paths: List of PDF file paths to convert

    Returns:
        List of converted Docling documents
    """
    docling_input_paths = [os.path.abspath(f) for f in file_paths]

    pdf_options = PdfPipelineOptions(generate_page_images=False)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options,
                backend=DoclingParseV4DocumentBackend,
            )
        }
    )

    return list(converter.convert_all(docling_input_paths, raises_on_error=False))


@step
def extract_text(docling_results: List[Any], file_paths: List[str]) -> List[List[Dict[str, Any]]]:
    """Extract text content page by page from Docling documents.

    Args:
        docling_results: List of converted Docling documents
        file_paths: List of original file paths

    Returns:
        List of document contents, where each document is a list of pages with text and metadata
    """
    extracted: List[List[Dict[str, Any]]] = []

    for result, file_path in zip(docling_results, file_paths, strict=True):
        if not result:
            logger.error(f"Document conversion failed for {file_path}")
            extracted.append([])
            continue

        pages = _get_document_pages(result)
        if not pages:
            logger.error(f"No pages found in document {file_path}")
            extracted.append([])
            continue

        pages_data = _process_document_pages(pages, file_path)
        extracted.append(pages_data)

        logger.info(f"Extracted {len(pages_data)} pages from {file_path}")

    _log_extraction_summary(extracted)
    return extracted

    def extract_text(
        self, docling_results: List[Any], file_paths: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """Step 2: Extracts text content page by page from Docling documents."""
        extracted = []
        for result, file_path in zip(docling_results, file_paths, strict=True):
            if hasattr(result, "document") and result.document:
                pages = self._extract_text_with_docling(result.document, file_path)
                extracted.append(pages)
            else:
                logger.warning(
                    f"Docling conversion succeeded but no document object found for {file_path}."
                )
                extracted.append([])
        return extracted

    def chunk_text(self, pages_data: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Step 3: Splits extracted text into chunks."""
        return [self._chunk_document_multi_stage(pages) for pages in pages_data]

    def generate_embeddings(self, all_chunks: List[List[Dict[str, Any]]]) -> List[np.ndarray]:
        """Step 4: Generates vector embeddings for each chunk."""
        return [
            self._generate_embeddings([chunk["text"] for chunk in chunks]) for chunks in all_chunks
        ]

    def prepare_qdrant_points(
        self, all_chunks: List[List[Dict[str, Any]]], all_embeddings: List[np.ndarray]
    ) -> List[models.PointStruct]:
        """Step 5: Prepares data points for Qdrant insertion."""
        points = []
        for chunks, embeddings in zip(all_chunks, all_embeddings, strict=True):
            points.extend(
                [
                    models.PointStruct(
                        id=chunk["metadata"]["global_chunk_id"],
                        vector=embedding.tolist(),
                        payload=chunk["metadata"],
                    )
                    for embedding, chunk in zip(embeddings, chunks, strict=True)
                ]
            )
        return points

    def insert_into_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """Step 6: Inserts points into Qdrant collection."""
        if points:
            # Extract vectors and convert to numpy array for compatibility
            import numpy as np

            vectors = np.array([p.vector for p in points])
            self.qdrant_db_client.insert_embbedings(collection_name, vectors, points)
            logger.info(
                f"Inserted {len(points)} chunks into Qdrant collection '{collection_name}'."
            )
        else:
            logger.info("No points to insert into Qdrant.")

    def create_collection(self, collection_name: str) -> None:
        """Creates Qdrant collection if not exists."""
        if not collection_name:
            raise ValueError("The passed in collection name is not valid")
        vector_size = self.encoding_model.get_sentence_embedding_dimension()
        self._create_collection(collection_name, vector_size)

    def run_ingestion_pipeline(self, file_paths: List[str], collection_name: str) -> None:
        """Orchestrates the modular ingestion pipeline steps."""
        logger.info("Starting ingestion pipeline...")
        self.create_collection(collection_name)
        docling_results = self.convert_documents(file_paths)
        pages_data = self.extract_text(docling_results, file_paths)
        all_chunks = self.chunk_text(pages_data)
        all_embeddings = self.generate_embeddings(all_chunks)
        points = self.prepare_qdrant_points(all_chunks, all_embeddings)
        self.insert_into_qdrant(collection_name, points)
        logger.info("Ingestion pipeline completed.")


if __name__ == "__main__":
    # Ensure QDRANT_URL environment variable is set
    # Example: os.environ["QDRANT_URL"] = "http://localhost"
    if not os.getenv("QDRANT_URL"):
        logger.error("Please set the QDRANT_URL environment variable.")
        exit(1)

    data_directory = "data/documents"  # Make sure this directory exists and contains PDF files

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        logger.info(
            f"Created directory: {data_directory}. Please add some PDF books here to ingest."
        )
        exit(0)

    # Instantiate the Qdrant client manager
    qdrant_client_instance = QdrantDBClient()

    # Instantiate the DocumentIngestor, passing the Qdrant client instance
    ingestor = DocumentIngestor(qdrant_client_instance, DoclingParseV4DocumentBackend)

    # Collect all PDF file paths for batch processing
    pdf_file_paths: List[str] = []
    for file_name in os.listdir(data_directory):
        full_path = os.path.join(data_directory, file_name)
        if os.path.isfile(full_path) and full_path.lower().endswith(".pdf"):
            pdf_file_paths.append(full_path)
        else:
            logger.info(f"Skipping non-PDF file or directory: {full_path}")

    if not pdf_file_paths:
        logger.warning(f"No PDF files found in {data_directory}. Nothing to ingest.")
        exit(0)

    # Run modular ingestion pipeline
    ingestor.run_ingestion_pipeline(pdf_file_paths, "Data-science")
