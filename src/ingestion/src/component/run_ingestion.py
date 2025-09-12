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


def _get_document_pages(result: Any) -> Union[List[Any], Any]:
    """Extract pages from a Docling document result.

    Args:
        result: Docling document result

    Returns:
        List of pages if found, None otherwise
    """
    if hasattr(result, "pages"):
        return result.pages
    if hasattr(result, "document") and result.document and hasattr(result.document, "pages"):
        return result.document.pages
    return None


def _process_document_pages(pages: List[Any], file_path: str) -> List[Dict[str, Any]]:
    """Process all pages in a document.

    Args:
        pages: List of document pages
        file_path: Source file path for metadata

    Returns:
        List of processed pages with text and metadata
    """
    pages_data: List[Dict[str, Any]] = []

    try:
        for page_num, page in enumerate(pages, start=1):
            if text := extract_page_text(page):
                pages_data.append(
                    {
                        "content": text,
                        "metadata": {
                            "source": file_path,
                            "page_number": page_num,
                        },
                    }
                )
                logger.debug(f"Extracted text from page {page_num} (length: {len(text)})")
            else:
                logger.warning(f"No text extracted from page {page_num}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)

    return pages_data


def _log_extraction_summary(extracted: List[List[Dict[str, Any]]]) -> None:
    """Log summary of text extraction results.

    Args:
        extracted: List of extracted document contents
    """
    total_docs = len(extracted)
    docs_with_content = sum(1 for doc in extracted if doc)
    logger.info(f"Documents processed: {total_docs}, with content: {docs_with_content}")


@step
def chunk_text(pages_data: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """Split extracted text into chunks for embedding.

    Args:
        pages_data: List of document pages with text and metadata

    Returns:
        List of document chunks, each chunk containing text and metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    def process_page(page_data: Dict[str, Any], chunk_id_counter: int) -> List[Dict[str, Any]]:
        """Process a single page into chunks.

        Args:
            page_data: Page text and metadata
            chunk_id_counter: Starting chunk ID for this page

        Returns:
            List of chunks with updated metadata
        """
        chunks = []
        text_chunks = splitter.split_text(page_data["content"])

        for i, text in enumerate(text_chunks, start=1):
            if not text.strip():
                continue

            metadata = page_data["metadata"].copy()
            metadata.update(
                {
                    "chunk_id_on_page": i,
                    "global_chunk_id": chunk_id_counter + i,
                }
            )

            chunks.append(
                {
                    "text": text.strip(),
                    "metadata": metadata,
                }
            )

        return chunks

    all_chunks = []
    chunk_id_counter = 0

    for pages in pages_data:
        doc_chunks = []
        for page_data in pages:
            page_chunks = process_page(page_data, chunk_id_counter)
            doc_chunks.extend(page_chunks)
            chunk_id_counter += len(page_chunks)
        all_chunks.append(doc_chunks)

    return all_chunks


# @step
def generate_embeddings(
    all_chunks: List[List[Dict[str, Any]]],
) -> tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
    """Step 4: Generates vector embeddings for each chunk."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    batch_size = QdrantDBConfig().batch_size

    # Generate embeddings and metadata
    embeddings = []
    metadata = []

    for chunks in all_chunks:
        chunk_embeddings = model.encode(
            sentences=[chunk["text"] for chunk in chunks],
            show_progress_bar=False,
            batch_size=batch_size,
        )
        embeddings.append(chunk_embeddings)

        # Extract metadata for each chunk
        chunk_metadata = [
            {
                **chunk["metadata"],
                "text": chunk["text"],  # Include the text in metadata
            }
            for chunk in chunks
        ]
        metadata.append(chunk_metadata)

    return embeddings, metadata


# @step
def insert_into_qdrant(
    collection_name: str,
    embeddings_and_metadata: tuple[List[np.ndarray], List[List[Dict[str, Any]]]],
) -> None:
    """Step 6: Inserts points into Qdrant collection."""
    embeddings, metadata = embeddings_and_metadata

    if not embeddings:
        logger.info("No points to insert into Qdrant.")
        return

    # Process each document's chunks
    client = QdrantDBClient()
    for doc_embeddings, doc_metadata in zip(embeddings, metadata, strict=True):
        # Convert single document embeddings to a list
        embeddings_list = [embedding.reshape(-1) for embedding in doc_embeddings]
        client.insert_embeddings(
            collection_name=collection_name, embeddings=embeddings_list, metadata=doc_metadata
        )

    total_chunks = sum(len(doc_embeddings) for doc_embeddings in embeddings)
    logger.info(
        f"Inserted {total_chunks} chunks from {len(embeddings)} documents into collection '{collection_name}'."
    )


# @step
def create_collection(collection_name: str) -> None:
    """Creates Qdrant collection if not exists."""
    if not collection_name:
        raise ValueError("Collection name cannot be empty")

    # Initialize embedding model and get vector size
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_size = model.get_sentence_embedding_dimension()
    if not isinstance(vector_size, int) or vector_size <= 0:
        raise ValueError(f"Invalid vector size: {vector_size}")

    # Create collection
    client = QdrantDBClient()
    client.create_db_collection(collection_name=collection_name, vector_size=vector_size)


@pipeline
def run_ingestion_pipeline(file_paths: List[str], collection_name: str) -> None:
    """Run the complete document ingestion pipeline.

    Args:
        file_paths: List of PDF file paths to process
        collection_name: Name of the Qdrant collection to use

    Raises:
        Exception: If any pipeline step fails
    """
    logger.info("Starting document ingestion pipeline...")

    try:
        # Initialize vector storage
        create_collection(collection_name)

        # Process documents
        docling_results = convert_documents(file_paths)
        pages_data = extract_text(docling_results, file_paths)
        chunks = chunk_text(pages_data)

        # Generate and store embeddings
        embeddings_and_metadata = generate_embeddings(chunks)
        insert_into_qdrant(collection_name, embeddings_and_metadata)

        logger.info("Document ingestion pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _get_pdf_files(data_dir: str) -> List[str]:
    """Get list of PDF files from directory.

    Args:
        data_dir: Directory to scan for PDF files

    Returns:
        List of absolute paths to PDF files

    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created directory: {data_dir}")
        return []

    pdf_paths = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath) and filepath.lower().endswith(".pdf"):
            pdf_paths.append(filepath)
        else:
            logger.debug(f"Skipping non-PDF file: {filepath}")

    return pdf_paths


def main() -> None:
    """Entry point for the ingestion pipeline."""
    # Set up environment
    os.environ["ZENML_SERVER"] = "http://localhost:8080"

    if not os.getenv("QDRANT_URL"):
        logger.error("QDRANT_URL environment variable not set")
        sys.exit(1)

    # Scan for PDF files
    data_dir = "data/documents"
    pdf_paths = _get_pdf_files(data_dir)

    if not pdf_paths:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    # Run pipeline
    run_ingestion_pipeline(file_paths=pdf_paths, collection_name="lease_documents")


if __name__ == "__main__":
    main()
