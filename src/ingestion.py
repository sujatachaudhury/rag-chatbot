"""PDF loading and chunking."""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load a single PDF, tagging each page with its source file."""
    pdf_path = Path(pdf_path)
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source_file"] = pdf_path.name
        page.metadata["file_type"] = "pdf"
    return pages


def load_pdfs(pdf_directory: Path) -> List[Document]:
    """Load every PDF under `pdf_directory`, tagging each page with its source file."""
    documents: List[Document] = []
    for pdf_file in sorted(Path(pdf_directory).glob("**/*.pdf")):
        documents.extend(load_pdf(pdf_file))
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
