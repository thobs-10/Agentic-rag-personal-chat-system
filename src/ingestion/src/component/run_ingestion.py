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
        """
        This function generates embeddings for the given texts.

        Args:
            texts (List[str]): A list of texts to generate embeddings for.

        Returns:
            np.ndarray: A numpy array of embeddings where each row corresponds to a text.
        """
        embeddings = self.encoding_model.encode(
            sentences=texts, show_progress_bar=False, batch_size=self.batch_size
        )
        return embeddings

    def _create_collection(self, collection_name: str, vector_size: Any) -> None:
        """
        This function creates a collection in Qdrant.

        Args:
            collection_name (str): The name of the collection to be created.
        """
        try:
            vector_size = self.encoding_model.get_sentence_embedding_dimension()
            self.qdrant_db_client.create_db_collection(collection_name, vector_size)
        except Exception as e:
            raise e

    def ingest(self, file_path: str, collection_name: str) -> None:
        docling_input_paths = [os.path.abspath(f) for f in file_path]

        logger.info("Converting documents with Docling...")
        conv_results = self.docling_converter.convert_all(
            docling_input_paths,
            raises_on_error=False,  # Continue even if some documents fail
        )
        logger.info("Docling batch conversion complete.")

        all_points_for_qdrant: List[models.PointStruct] = []
        total_chunks_ingested = 0

        # Define vector size once for the collection
        vector_size = self.encoding_model.get_sentence_embedding_dimension()
        if not collection_name:
            raise ValueError("The passed in collection name is not valid")
        self._create_collection(collection_name, vector_size)

        for result in conv_results:
            if not result.document.export_to_text().strip():
                logger.error("Docling conversion failed ")
                continue

            # Ensure a DoclingDocument object is present
            if not result.document:
                logger.warning(
                    f"Docling conversion succeeded but no document object found for {file_path}."
                )
                continue

            logger.info(f"Processing document: {file_path}")
            try:
                # 1.5. Extract pages from the DoclingDocument object
                pages_data = self._extract_text_with_docling(result.document, file_path)

                if not pages_data:
                    logger.info(f"No pages extracted or valid from {file_path}. Skipping.")
                    continue

                # 2. Text-Level Chunking (Recursive Character Splitter on each page)
                final_chunks = self._chunk_document_multi_stage(pages_data)

                if not final_chunks:
                    logger.info(
                        f"No chunks generated from {file_path}. Skipping embedding and insertion."
                    )
                    continue

                # 3. Embeddings
                texts_to_embed = [chunk["text"] for chunk in final_chunks]
                embeddings = self._generate_embeddings(texts_to_embed)
                # Prepare points for insertion
                # Note: We need to adjust IDs for batch insertion or ensure uniqueness across all docs.
                # A simple approach is to manage a global ID counter within this batch ingestion.
                # For now, let's append to a single list of points and upsert all at once.
                # Qdrant IDs should be globally unique within a collection.
                # Using the `global_chunk_id` generated in `_chunk_document_multi_stage` is good.

                points_for_current_doc = [
                    models.PointStruct(
                        id=chunk["metadata"]["global_chunk_id"],  # Unique ID from chunking
                        vector=embedding.tolist(),
                        payload=chunk["metadata"],
                    )
                    for embedding, chunk in zip(embeddings, final_chunks, strict=True)
                ]
                all_points_for_qdrant.extend(points_for_current_doc)
                total_chunks_ingested += len(points_for_current_doc)

            except Exception as e:
                logger.error(
                    f"Error during processing of {file_path} after Docling conversion: {e}",
                    exc_info=True,
                )

        if all_points_for_qdrant:
            logger.info(
                f"Inserting a total of {len(all_points_for_qdrant)} chunks into Qdrant for this batch."
            )
            self.qdrant_db_client.insert_embbedings(
                collection_name, embeddings, all_points_for_qdrant
            )
            logger.info(
                f"Batch ingestion completed. Total chunks processed: {total_chunks_ingested}"
            )
        else:
            logger.info(
                "No documents were successfully processed or generated chunks in this batch."
            )


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
    pdf_file_paths = List[str]
    for file_name in os.listdir(data_directory):
        full_path = os.path.join(data_directory, file_name)
        if os.path.isfile(full_path) and full_path.lower().endswith(".pdf"):
            pdf_file_paths.append(full_path)
        else:
            logger.info(f"Skipping non-PDF file or directory: {full_path}")

    if not pdf_file_paths:
        logger.warning(f"No PDF files found in {data_directory}. Nothing to ingest.")
        exit(0)

    # Perform batch ingestion
    ingestor.ingest(pdf_file_paths, "Data-science")
