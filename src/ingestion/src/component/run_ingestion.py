import os
from typing import Any, Dict, List

import numpy as np
from config import QdrantDBConfig
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from qdrant_client.http import models
from qdrant_db_client import QdrantDBClient
from sentence_transformers import SentenceTransformer


class DocumentIngestor:
    """
    Handles the ingestion process of documents:
    1. Reads document content using Docling.
    2. Chunks the content.
    3. Generates embeddings for the chunks.
    4. Inserts embeddings and metadata into Qdrant via a QdrantKnowledgeBaseClient.
    """

    def __init__(self, db_client: QdrantDBClient, docling_backend=None) -> None:
        self.qdrant_db_client = db_client
        self.encoding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.batch_size = QdrantDBConfig().batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Optimal size for your embedding model and LLM context window
            chunk_overlap=100,  # To maintain context across sub-chunks
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        # Configure Docling's DocumentConverter for PDF batch processing
        pdf_pipeline_options = PdfPipelineOptions()
        # Set to False to avoid generating image files if not needed, improving performance
        pdf_pipeline_options.generate_page_images = False
        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                    backend=docling_backend
                    or DoclingParseV4DocumentBackend,  # Use the specified backend
                )
            }
        )

    def _extract_text_with_docling(
        self,
        docling_doc: Any,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Extracts text content from a Docling document object page by page.

        Args:
            docling_doc: Parsed document object from Docling
            file_path: Source file path for metadata tracking

        Returns:
            List of dictionaries containing:
                - content: Extracted text from page
                - metadata: Source file and page number info

        Note:
            Returns empty list if extraction fails or document contains no valid text
        """
        pages_data: List[Dict[str, Any]] = []
        try:
            for page_num, page in enumerate(docling_doc.pages, start=1):
                page_content = page.text()
                if page_content.strip():
                    pages_data.append(
                        {
                            "content": page_content,
                            "metadata": {
                                "source": file_path,
                                "page_number": page_num,
                            },
                        }
                    )
                else:
                    logger.warning(f"Page {page_num} of {file_path} extracted as empty by Docling.")

            if not pages_data:
                logger.warning(f"Docling extracted no content from any page in file: {file_path}")

            logger.info(f"Extracted {len(pages_data)} pages from {file_path}.")
            return pages_data

        except Exception as e:
            logger.error(
                f"Error extracting page-by-page text from DoclingDocument for {file_path}: {e}"
            )
            return []

    def _chunk_document_multi_stage(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits document pages into smaller text chunks using recursive splitting.

        Args:
            pages_data: List of page data dictionaries from _extract_text_with_docling

        Returns:
            List of dictionaries containing:
                - text: The text chunk content
                - metadata: Original metadata plus chunk position information

        Note:
            Maintains global chunk IDs across entire document for consistent referencing
        """
        final_chunks: List[Dict[str, Any]] = []
        global_chunk_id = 0
        for page_data in pages_data:
            page_content = page_data["content"]
            page_metadata = page_data["metadata"]

            texts_on_page = self.text_splitter.split_text(page_content)
            for i, text_chunk in enumerate(texts_on_page):
                if text_chunk.strip():
                    global_chunk_id += 1
                    chunk_metadata = page_metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_id_on_page": i + 1,
                            "global_chunk_id": global_chunk_id,
                        }
                    )
                    final_chunks.append(
                        {
                            "text": text_chunk.strip(),
                            "metadata": chunk_metadata,
                        }
                    )

        return final_chunks

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates vector embeddings for a list of text chunks.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings where each row corresponds to input texts

        Note:
            Uses batch processing for efficiency based on configured batch_size
        """
        embeddings = self.encoding_model.encode(
            sentences=texts, show_progress_bar=False, batch_size=self.batch_size
        )
        return embeddings

    def _create_collection(self, collection_name: str, vector_size: Any) -> None:
        """Creates a new Qdrant collection with specified parameters.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimensionality of vectors to be stored

        Raises:
            Exception: If collection creation fails
        """
        try:
            vector_size = self.encoding_model.get_sentence_embedding_dimension()
            self.qdrant_db_client.create_db_collection(collection_name, vector_size)
        except Exception as e:
            raise e

    def convert_documents(self, file_paths: List[str]) -> List[Any]:
        """Step 1: Converts input files to Docling documents."""
        docling_input_paths = [os.path.abspath(f) for f in file_paths]
        # Ensure we return a list, not an iterator
        return list(
            self.docling_converter.convert_all(
                docling_input_paths,
                raises_on_error=False,
            )
        )

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
