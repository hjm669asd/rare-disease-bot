"""文件解析器 - 支持TXT/PDF/Word/Excel格式解析"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


class FileParser:
    """文件解析器"""

    SUPPORTED_FORMATS = [".txt", ".pdf", ".docx", ".xlsx", ".md"]

    def __init__(self):
        pass

    def parse(self, file_path: str, metadata: Optional[dict] = None) -> list[Document]:
        """
        解析文件并返回Document列表

        Args:
            file_path: 文件路径
            metadata: 额外的元数据

        Returns:
            list[Document]: 解析后的Document列表
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的文件格式: {suffix}，支持的格式: {self.SUPPORTED_FORMATS}")

        # 基础元数据
        base_metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": suffix,
            "file_size": os.path.getsize(file_path),
        }
        if metadata:
            base_metadata.update(metadata)

        # 根据文件类型解析
        if suffix == ".txt" or suffix == ".md":
            return self._parse_txt(file_path, base_metadata)
        elif suffix == ".pdf":
            return self._parse_pdf(file_path, base_metadata)
        elif suffix == ".docx":
            return self._parse_docx(file_path, base_metadata)
        elif suffix == ".xlsx":
            return self._parse_xlsx(file_path, base_metadata)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _parse_txt(self, file_path: Path, metadata: dict) -> list[Document]:
        """解析TXT文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()

        if not content.strip():
            return []

        return [Document(
            page_content=content,
            metadata={**metadata, "doc_type": "text_file"},
        )]

    def _parse_pdf(self, file_path: Path, metadata: dict) -> list[Document]:
        """解析PDF文件"""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["doc_type"] = "pdf"

            return documents
        except ImportError:
            raise ImportError("请安装PDF解析库: pip install pypdf")
        except Exception as e:
            raise Exception(f"PDF解析失败: {str(e)}")

    def _parse_docx(self, file_path: Path, metadata: dict) -> list[Document]:
        """解析Word文件"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["doc_type"] = "word"

            return documents
        except ImportError:
            raise ImportError("请安装Word解析库: pip install docx2txt")
        except Exception as e:
            raise Exception(f"Word解析失败: {str(e)}")

    def _parse_xlsx(self, file_path: Path, metadata: dict) -> list[Document]:
        """解析Excel文件"""
        try:
            import pandas as pd

            df = pd.read_excel(file_path).fillna("")

            documents = []
            for index, row in df.iterrows():
                # 将每行转换为文本
                content = "\n".join([f"{col}: {val}" for col, val in row.items() if val])

                if content.strip():
                    doc_metadata = {
                        **metadata,
                        "doc_type": "excel",
                        "row_index": index + 1,
                    }
                    documents.append(Document(
                        page_content=content,
                        metadata=doc_metadata,
                    ))

            return documents
        except Exception as e:
            raise Exception(f"Excel解析失败: {str(e)}")

    def get_supported_formats(self) -> list[str]:
        """获取支持的文件格式"""
        return self.SUPPORTED_FORMATS.copy()
