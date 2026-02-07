"""Test cases for the utils functions that are used in the ingestion pipeline."""

from typing import Any, Dict, List, Tuple

from agentic_rag_personal_chat_system.ingestion.src.main_utils.utils import (
    extract_page_text,
    get_document_pages,
)


def test_extract_text_from_attribute_success() -> None:
    """Test extracting text from different page structures."""

    class PageWithText:
        def __init__(self, text: str) -> None:
            self.text = text

    class PageWithExportMethod:
        def export_to_text(self) -> str:
            return "Exported text"

    class PageWithContent:
        def __init__(self, content: str) -> None:
            self.content = content

    # Test with direct text attribute
    page1 = PageWithText("Direct text")
    assert extract_page_text(page1) == "Direct text"

    # Test with export_to_text method
    page2 = PageWithExportMethod()
    assert extract_page_text(page2) == "Exported text"

    # Test with content attribute
    page3 = PageWithContent("Content text")
    assert extract_page_text(page3) == "Content text"

    # Test with no valid text extraction method
    class EmptyPage:
        pass

    page4 = EmptyPage()
    assert extract_page_text(page4) is None


def test_extract_text_from_attribute_failure() -> None:
    """Test extracting text when attribute access fails."""

    class PageWithError:
        def text(self) -> str:
            raise Exception("Failed to get text")

    page = PageWithError()
    assert extract_page_text(page) is None


def test_extract_text_from_attribute_non_string() -> None:
    """Test extracting text when attribute returns non-string."""

    class PageWithNonStringText:
        def text(self) -> int:
            return 12345  # Non-string text

    page = PageWithNonStringText()
    assert extract_page_text(page) is None


def test_extract_text_from_attribute_empty_string() -> None:
    """Test extracting text when attribute returns empty string."""

    class PageWithEmptyText:
        def text(self) -> str:
            return "   "  # Empty string with whitespace

    page = PageWithEmptyText()
    assert extract_page_text(page) is None


def test_extract_text_from_attribute_tuple() -> None:
    """Test extracting text when attribute returns a tuple."""

    class PageWithTupleText:
        def text(self) -> Tuple[str, ...]:
            return ("Line 1", "Line 2", "Line 3")

    page = PageWithTupleText()
    assert extract_page_text(page) == "Line 1\nLine 2\nLine 3"


def test_extract_text_from_attribute_exception() -> None:
    """Test extracting text when attribute access raises an exception."""

    class PageWithException:
        def text(self) -> str:
            raise Exception("Error accessing text")

    page = PageWithException()
    assert extract_page_text(page) is None


def test_get_document_pages_success() -> None:
    """Test extracting pages from different document structures."""

    class DocumentWithPages:
        def __init__(self, pages: List[str]) -> None:
            self.pages = pages

    class DocumentWithDocumentPages:
        def __init__(self, document: Any) -> None:
            self.document = document

    # Test with direct pages attribute
    doc1 = DocumentWithPages(["Page 1", "Page 2"])
    assert get_document_pages(doc1) == ["Page 1", "Page 2"]

    # Test with document.pages attribute
    class InnerDocument:
        def __init__(self, pages: List[str]) -> None:
            self.pages = pages

    doc2 = DocumentWithDocumentPages(InnerDocument(["Page A", "Page B"]))
    assert get_document_pages(doc2) == ["Page A", "Page B"]

    # Test with dict structure (from serialization)
    doc3 = {"content": ["Serialized Page 1", "Serialized Page 2"]}
    assert get_document_pages(doc3) == ["Serialized Page 1", "Serialized Page 2"]


def test_get_document_pages_failure() -> None:
    """Test extracting pages when no valid page structure is found."""

    class EmptyDocument:
        pass

    doc = EmptyDocument()
    assert get_document_pages(doc) is None

    # Test with dict that has no content or pages
    doc2: Dict[str, Any] = {"no_pages": []}
    assert get_document_pages(doc2) is None


def test_get_document_pages_pages_as_dict() -> None:
    """Test extracting pages when pages are returned as a dict."""

    # Test with dict structure where pages are a dict of page_num: page_obj
    doc: Dict[str, Any] = {"content": {"1": "Page 1", "2": "Page 2"}}
    assert get_document_pages(doc) == ["Page 1", "Page 2"]


def test_get_document_pages_pages_as_string() -> None:
    """Test extracting pages when content is a single string."""

    # Test with dict structure where content is a single string
    doc: Dict[str, Any] = {"content": "Single page content"}
    assert get_document_pages(doc) == ["Single page content"]


def test_get_document_pages_pages_as_list() -> None:
    """Test extracting pages when content is a list of strings."""

    # Test with dict structure where content is a list of strings
    doc: Dict[str, Any] = {"content": ["Page 1", "Page 2", "Page 3"]}
    assert get_document_pages(doc) == ["Page 1", "Page 2", "Page 3"]


def test_get_document_pages_pages_as_invalid_type() -> None:
    """Test extracting pages when content is of an invalid type."""

    # Test with dict structure where content is an invalid type (e.g., int)
    doc: Dict[str, Any] = {"content": 12345}
    assert get_document_pages(doc) is None


def test_get_document_pages_content_as_object_with_pages_attribute() -> None:
    """Test extracting pages when content is an object with a pages attribute."""

    class PagesObject:
        def __init__(self, pages: List[str]) -> None:
            self.pages = pages

    # Test with dict where content is an object that has a pages attribute
    doc: Dict[str, Any] = {"content": PagesObject(["Page 1", "Page 2"])}
    assert get_document_pages(doc) == ["Page 1", "Page 2"]
