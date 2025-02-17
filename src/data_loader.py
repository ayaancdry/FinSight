"""Data ingestion and preprocessing module for PDF documents."""

import re
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import config


class DocumentLoader:
    """
    Handles loading, cleaning, and chunking of PDF documents.

    This class provides methods to:
    - Load PDF files and extract text content
    - Clean extracted text by removing noise
    - Split documents into chunks for embedding
    """

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap
    ):
        """
        Initialize the DocumentLoader.

        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract its content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects containing the extracted text.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {file_path}")

        loader = PyPDFLoader(str(path))
        documents = loader.load()
        return documents

    def load_pdf_from_bytes(self, file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
        """
        Load a PDF from bytes (useful for Streamlit file uploads).

        Args:
            file_bytes: PDF content as bytes.
            filename: Original filename for metadata.

        Returns:
            List of Document objects containing the extracted text.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            documents = self.load_pdf(tmp_path)
            # Update metadata with original filename
            for doc in documents:
                doc.metadata["source"] = filename
            return documents
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing noise, headers, footers, and artifacts.

        This function performs the following cleaning operations:
        - Removes page numbers and headers/footers patterns
        - Normalizes whitespace
        - Removes excessive line breaks
        - Removes common PDF artifacts

        Args:
            text: Raw text extracted from PDF.

        Returns:
            Cleaned text string.
        """
        # Remove page numbers (various formats)
        text = re.sub(r"(?i)page\s*\d+\s*(of\s*\d+)?", "", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Remove common header/footer patterns
        text = re.sub(r"(?i)(table of contents|confidential|draft)", "", text)

        # Remove URLs and email addresses
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def chunk_documents(
        self,
        documents: List[Document],
        clean: bool = True
    ) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.

        Args:
            documents: List of Document objects to chunk.
            clean: Whether to clean text before chunking.

        Returns:
            List of chunked Document objects.
        """
        if clean:
            for doc in documents:
                doc.page_content = self.clean_text(doc.page_content)

        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_pdf(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        filename: str = "document.pdf"
    ) -> List[Document]:
        """
        Complete pipeline: load, clean, and chunk a PDF document.

        Args:
            file_path: Path to PDF file (mutually exclusive with file_bytes).
            file_bytes: PDF content as bytes.
            filename: Original filename for metadata.

        Returns:
            List of processed and chunked Document objects.
        """
        if file_path:
            documents = self.load_pdf(file_path)
        elif file_bytes:
            documents = self.load_pdf_from_bytes(file_bytes, filename)
        else:
            raise ValueError("Either file_path or file_bytes must be provided")

        chunks = self.chunk_documents(documents)
        return chunks
