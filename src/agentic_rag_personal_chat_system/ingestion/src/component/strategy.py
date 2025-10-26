import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.document_loaders import PyPDFLoader
from loguru import logger


class LoadingStrategy(ABC):
    @abstractmethod
    def load_documents(
        self,
        file_paths: List[str],
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def convert_to_absolute_path(
        self,
        file_path: str,
    ) -> Union[Tuple[List[Any], str], Tuple[Any, str, Path]]:
        """Convert a file path to absolute path and load the document.

        Args:
            file_path: Path to the document file

        Returns:
            A tuple containing:
            - Loaded document(s)
            - Absolute path to the file
            - (Optional) Path object for the file
        """
        pass


class LangChainStrategy(LoadingStrategy):
    def convert_to_absolute_path(
        self,
        file_path: str,
    ) -> Tuple[List[Any], str]:
        """Convert a file path to absolute path and load using LangChain.

        Args:
            file_path: Path to the document file

        Returns:
            A tuple containing:
            - List of loaded document objects
            - Absolute path to the file
        """
        # Convert to absolute path and load
        abs_path = os.path.abspath(file_path)
        loader = PyPDFLoader(Path(abs_path))
        docs = loader.load()
        return docs, abs_path

    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from a list of file paths using LangChain.

        Args:
            file_paths: List of paths to document files

        Returns:
            List of dictionaries containing document content and metadata

        Raises:
            ValueError: If no files are provided or document conversion fails
        """
        if not file_paths:
            raise ValueError("No PDF files provided for processing.")

        converted_docs: List[Dict[str, Any]] = []
        for file_path in file_paths:
            try:
                # Convert to absolute path and load documents
                docs, abs_path = self.convert_to_absolute_path(file_path)

                # Process each page
                for doc in docs:
                    if not hasattr(doc, "metadata") or not hasattr(doc, "page_content"):
                        raise ValueError(f"Invalid document format: {doc}")

                    # Add file metadata
                    doc.metadata.update(
                        {
                            "source": abs_path,
                            "file_name": Path(file_path).name,
                            "file_type": Path(file_path).suffix.lower(),
                        }
                    )

                    converted_docs.append({"content": doc.page_content, "metadata": doc.metadata})

            except Exception as e:
                logger.error(f"Failed to process '{file_path}': {str(e)}")
                raise ValueError(f"Document conversion failed: {file_path}") from e

        return converted_docs


class DoclingStrategy(LoadingStrategy):
    def get_document_converter(self) -> DocumentConverter:
        """Create and return a new DocumentConverter instance."""
        return DocumentConverter()

    def convert_to_absolute_path(
        self,
        file_path: str,
    ) -> Tuple[Any, str, Path]:
        """Convert a file path to absolute path and load using Docling.

        Args:
            file_path: Path to the document file

        Returns:
            A tuple containing:
            - Loaded document object
            - Absolute path to the file
            - Path object for the file
        """
        # Convert to absolute path and create Path object
        abs_path = os.path.abspath(file_path)
        doc_path = Path(abs_path)

        # Get a converter and convert the document
        converter = self.get_document_converter()
        result = converter.convert(doc_path, raises_on_error=False)
        doc = getattr(result, "document", None)

        return doc, abs_path, doc_path

    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from a list of file paths using Docling.

        Args:
            file_paths: List of paths to document files

        Returns:
            List of dictionaries containing document content and metadata

        Raises:
            ValueError: If no files are provided or document conversion fails
        """
        if not file_paths:
            raise ValueError("No PDF files provided for processing.")

        converted_docs = []
        for file_path in file_paths:
            try:
                # Convert file to absolute path and process with Docling
                doc, abs_path, doc_path = self.convert_to_absolute_path(file_path)

                if doc is None:
                    raise ValueError(f"Docling conversion failed for: {file_path}")

                # Convert to dict format (support both Pydantic v1 and v2)
                if hasattr(doc, "model_dump"):
                    doc_dict = doc.model_dump()
                elif hasattr(doc, "dict"):
                    doc_dict = doc.dict()
                else:
                    doc_dict = doc.__dict__

                # Add file metadata
                doc_dict["metadata"] = {
                    "source": abs_path,
                    "file_name": doc_path.name,
                    "file_type": doc_path.suffix.lower(),
                }

                converted_docs.append(doc_dict)

            except Exception as e:
                logger.error(f"Failed to process '{file_path}': {str(e)}")
                raise ValueError(f"Document conversion failed: {file_path}") from e

        return converted_docs
