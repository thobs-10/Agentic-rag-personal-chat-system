import os
from typing import Any, Dict, List

import numpy as np
from docx import Document
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models
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
        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file type for PDF ingestion: {file_path}")
        # use a pdf library to read the PDF content
        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        if not content.strip():
            raise ValueError(f"Empty content in file: {file_path}")
        return self._chunk_document(content, file_path)

    def _ingest_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        This function reads the content of the DOCX file and chunks it into smaller pieces.

        Args:
            file_path (str): Path to the file to be ingested.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chunked content and metadata.
        """
        if not file_path.lower().endswith(".docx"):
            raise ValueError(f"Unsupported file type for DOCX ingestion: {file_path}")

        doc = Document(file_path)
        content = ""
        for para in doc.paragraphs:
            content += para.text + "\n"
        if not content.strip():
            raise ValueError(f"Empty content in file: {file_path}")
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

    def _insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> None:
        """
        This function inserts documents into the Qdrant collection.

        Args:
            collection_name (str): The name of the collection to insert documents into.
            documents (List[Dict[str, Any]]): A list of dictionaries containing the documents to be inserted.
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self._generate_embeddings(texts)
        points = [
            models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=doc["metadata"],
            )
            for i, (embedding, doc) in enumerate(zip(embeddings, documents, strict=True))
        ]

        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def ingest(self, file_path: str) -> None:
        """
        This function ingests a document from the given file path.
        Args:
            file_path (str): Path to the file to be ingested.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.supported_file_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = self.supported_file_extensions[file_extension](file_path)
        collection_name = "documents"

        if not self.qdrant_client.get_collection(collection_name):
            self._create_collection(collection_name)

        self._insert_documents(collection_name, documents)
        print(f"Ingested {len(documents)} chunks from {file_path} into Qdrant.")


if __name__ == "__main__":
    # paths to the data files to be ingested
    file_path: str = "data/documents/"
    ingestor = DocumentIngestor()
    for file_name in os.listdir(file_path):
        full_path = os.path.join(file_path, file_name)
        if os.path.isfile(full_path):
            try:
                ingestor.ingest(full_path)
            except Exception as e:
                print(f"Failed to ingest {full_path}: {e}")
