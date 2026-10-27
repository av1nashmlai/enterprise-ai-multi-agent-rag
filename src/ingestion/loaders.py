"""
Task 3: Document Ingestion Module.
Supports uploading and parsing enterprise documents in PDF, TXT, CSV, and Excel formats
with automatic encoding fallback and malformed-line resiliency.
"""
import os
from typing import List
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document


class EnterpriseDocumentLoader:
    """Unified multi-format document loader for PDF, TXT, CSV, and Excel files."""

    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".pdf":
            return EnterpriseDocumentLoader._load_pdf(file_path, filename)
        elif ext == ".txt":
            return EnterpriseDocumentLoader._load_txt(file_path, filename)
        elif ext == ".csv":
            return EnterpriseDocumentLoader._load_csv(file_path, filename)
        elif ext in [".xlsx", ".xls"]:
            return EnterpriseDocumentLoader._load_excel(file_path, filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _load_pdf(file_path: str, filename: str) -> List[Document]:
        reader = PdfReader(file_path)
        docs = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": filename,
                            "type": "PDF",
                            "location": f"Page {idx + 1}",
                        },
                    )
                )
        return docs

    @staticmethod
    def _load_txt(file_path: str, filename: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return (
            [
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": filename,
                        "type": "TXT",
                        "location": "Full Text",
                    },
                )
            ]
            if text.strip()
            else []
        )

    @staticmethod
    def _load_csv(file_path: str, filename: str) -> List[Document]:
        try:
            df = pd.read_csv(
                file_path, encoding="utf-8", on_bad_lines="skip"
            ).fillna("N/A")
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path, encoding="latin-1", on_bad_lines="skip"
            ).fillna("N/A")
        docs = []
        # Convert tabular rows into rich semantic text records
        for idx, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(
                Document(
                    page_content=f"CSV Record from {filename} (Row {idx + 1}): {row_str}",
                    metadata={
                        "source": filename,
                        "type": "CSV",
                        "location": f"Row {idx + 1}",
                    },
                )
            )
        return docs

    @staticmethod
    def _load_excel(file_path: str, filename: str) -> List[Document]:
        docs = []
        with pd.ExcelFile(file_path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name).fillna("N/A")
                for idx, row in df.iterrows():
                    row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    docs.append(
                        Document(
                            page_content=(
                                f"Excel Record from {filename} [Sheet: {sheet_name}, Row {idx + 1}]: "
                                f"{row_str}"
                            ),
                            metadata={
                                "source": filename,
                                "type": "Excel",
                                "location": f"Sheet '{sheet_name}', Row {idx + 1}",
                            },
                        )
                    )
        return docs
