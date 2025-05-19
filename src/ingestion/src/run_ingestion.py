import os
from typing import Any, Dict, List

import numpy as np
from megaparse import Megaparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer


class DocumentIngestor:
    def __init__(self) -> None:
        self.encoding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            port=4443,
            prefer_grpc=True,
        )
        self.supported_file_extensions = {
            ".txt": self._ingest_txt,
            ".pdf": self._ingest_pdf,
            ".docx": self._ingest_docx,
        }
        self.batch_size = 32  # batch size for embedding generation

    def _ingest_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        This function reads the content of the file and chunks it into smaller pieces.

        Args:
            file_path (str): Path to the file to be ingested.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chunked content and metadata.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return self._chunk_document(content, file_path)

    def _ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        This function reads the content of the PDF file and chunks it into smaller pieces.

        Args:
            file_path (str): Path to the file to be ingested.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chunked content and metadata.
        """
        parser = Megaparse()
        with open(file_path, "rb") as file:
            content = parser.load(file)
            if not content:
                raise ValueError(f"Failed to parse PDF file: {file_path}")
        return self._chunk_document(content, file_path)

    def _ingest_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        This function reads the content of the DOCX file and chunks it into smaller pieces.

        Args:
            file_path (str): Path to the file to be ingested.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chunked content and metadata.
        """
        parser = Megaparse()
        with open(file_path, "rb") as file:
            content = parser.load(file)
            if not content:
                raise ValueError(f"Failed to parse DOCX file: {file_path}")
        return self._chunk_document(content, file_path)

    def _chunk_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        This function chunks the document content into smaller pieces.

        Args:
            content (str): The content of the document.
            file_path (str): Path to the file to be ingested.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chunked content and metadata.
        """
        chunks: List[Dict[str, Any]] = []
        if not content.strip():
            raise ValueError(f"Empty content in file: {source}")

        paragraphs = [p for p in content.split("\n") if p.strip()]
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > 1000:  # target chunk size
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk,
                            "metadata": {
                                "source": source,
                                "chunk": len(chunks) + 1,
                            },
                        }
                    )
                current_chunk = paragraph
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk,
                    "metadata": {
                        "source": source,
                        "chunk": len(chunks) + 1,
                    },
                }
            )
        return chunks

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        This function generates embeddings for the given texts.

        Args:
            texts (List[str]): A list of texts to generate embeddings for.

        Returns:
            np.ndarray: A numpy array of embeddings where each row corresponds to a text.
        """
        return self.encoding_model.encode(
            sentences=texts, show_progress_bar=True, batch_size=self.batch_size
        )

    def _create_collection(self, collection_name: str) -> None:
        """
        This function creates a collection in Qdrant.

        Args:
            collection_name (str): The name of the collection to be created.
        """
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "vector_params": VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            },
        )

    def _insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> None: ...
