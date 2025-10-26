from typing import Any, List, Optional, Dict, Union

from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_attribute(page: Any, attr_name: str) -> Optional[str]:
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


def get_document_pages(result: Any) -> Union[List[Any], Any]:
    """Extract pages from a Docling document result.

    Args:
        result: Docling document result

    Returns:
        List of pages if found, None otherwise
    """
    # If result is a dict (from serialization), handle accordingly
    if isinstance(result, dict):
        pages = result.get("content") or result.get("pages")
        if isinstance(pages, dict):
            # Docling sometimes returns pages as a dict of page_num: page_obj
            return list(pages.values())
        if isinstance(pages, list):
            return pages
        if isinstance(pages, str):
            return [pages]
        return None
    # If result is an object
    if hasattr(result, "pages"):
        return result.pages
    if hasattr(result, "document") and result.document and hasattr(result.document, "pages"):
        return result.document.pages
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
        if text := extract_text_from_attribute(page, method_name):
            return text
    return None


def process_document_pages(pages: List[Any], file_path: str) -> List[Dict[str, Any]]:
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


def log_extraction_summary(extracted: List[List[Dict[str, Any]]]) -> None:
    """Log summary of text extraction results.

    Args:
        extracted: List of extracted document contents
    """
    total_docs = len(extracted)
    docs_with_content = sum(1 for doc in extracted if doc)
    logger.info(f"Documents processed: {total_docs}, with content: {docs_with_content}")


def process_page(
    page_data: Dict[str, Any],
    chunk_id_counter: int,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Dict[str, Any]]:
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


def get_splitter_object(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> RecursiveCharacterTextSplitter:
    """Create and return a RecursiveCharacterTextSplitter instance.

    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
