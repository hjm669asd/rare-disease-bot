"""文件上传管理器 - 处理文件上传、解析、去重、存储"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from file_parser import FileParser


class FileUploader:
    """文件上传管理器"""

    def __init__(
        self,
        storage_dir: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        """
        初始化文件上传管理器

        Args:
            storage_dir: 存储目录
            chunk_size: 文档切分大小
            chunk_overlap: 切分重叠大小
        """
        # 设置存储目录
        if storage_dir is None:
            base_dir = Path(__file__).parent.parent
            storage_dir = base_dir / "storage"
        else:
            storage_dir = Path(storage_dir)

        self.storage_dir = storage_dir
        self.upload_dir = storage_dir / "uploads"
        self.metadata_file = storage_dir / "metadata.json"

        # 创建目录
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # 初始化文件解析器
        self.parser = FileParser()

        # 初始化文档切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )

        # 加载元数据
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"files": {}, "file_hashes": {}}
        return {"files": {}, "file_hashes": {}}

    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 保存元数据失败: {e}")

    def _calculate_file_hash(self, file_content: bytes) -> str:
        """计算文件哈希值（用于去重）"""
        return hashlib.md5(file_content).hexdigest()

    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        category: str = "用户上传",
    ) -> dict:
        """
        上传文件

        Args:
            file_content: 文件内容（字节）
            filename: 文件名
            category: 文件分类

        Returns:
            dict: 上传结果
        """
        try:
            # 1. 验证文件类型
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.parser.SUPPORTED_FORMATS:
                return {
                    "success": False,
                    "error": f"不支持的文件格式: {file_ext}，支持的格式: {self.parser.SUPPORTED_FORMATS}",
                }

            # 2. 计算文件哈希（去重）
            file_hash = self._calculate_file_hash(file_content)

            # 3. 检查是否已存在
            if file_hash in self.metadata.get("file_hashes", {}):
                existing_file = self.metadata["file_hashes"][file_hash]
                return {
                    "success": False,
                    "error": f"文件已存在: {existing_file['filename']}（上传于 {existing_file['upload_time']}）",
                    "duplicate": True,
                    "existing_file": existing_file,
                }

            # 4. 生成唯一文件名
            file_id = str(uuid.uuid4())[:8]
            safe_filename = f"{file_id}_{filename}"
            file_path = self.upload_dir / safe_filename

            # 5. 保存文件
            with open(file_path, "wb") as f:
                f.write(file_content)

            # 6. 解析文件
            documents = self.parser.parse(
                str(file_path),
                metadata={"category": category, "upload_time": datetime.now().isoformat()},
            )

            if not documents:
                return {
                    "success": False,
                    "error": "文件解析结果为空",
                }

            # 7. 切分文档
            chunks = self.splitter.split_documents(documents)

            # 8. 保存元数据
            file_info = {
                "file_id": file_id,
                "filename": filename,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "file_type": file_ext,
                "file_size": len(file_content),
                "category": category,
                "upload_time": datetime.now().isoformat(),
                "doc_count": len(documents),
                "chunk_count": len(chunks),
            }

            self.metadata["files"][file_id] = file_info
            self.metadata["file_hashes"][file_hash] = file_info
            self._save_metadata()

            return {
                "success": True,
                "file_id": file_id,
                "filename": filename,
                "doc_count": len(documents),
                "chunk_count": len(chunks),
                "message": f"文件上传成功，解析为 {len(documents)} 个文档，切分为 {len(chunks)} 个片段",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"上传失败: {str(e)}",
            }

    def get_uploaded_files(self) -> list[dict]:
        """获取已上传的文件列表"""
        files = []
        for file_id, file_info in self.metadata.get("files", {}).items():
            files.append({
                "file_id": file_id,
                "filename": file_info["filename"],
                "file_type": file_info["file_type"],
                "file_size": file_info["file_size"],
                "category": file_info.get("category", ""),
                "upload_time": file_info["upload_time"],
                "doc_count": file_info.get("doc_count", 0),
                "chunk_count": file_info.get("chunk_count", 0),
            })

        # 按上传时间倒序排序
        files.sort(key=lambda x: x["upload_time"], reverse=True)
        return files

    def delete_file(self, file_id: str) -> dict:
        """
        删除已上传的文件

        Args:
            file_id: 文件ID

        Returns:
            dict: 删除结果
        """
        if file_id not in self.metadata.get("files", {}):
            return {
                "success": False,
                "error": f"文件不存在: {file_id}",
            }

        file_info = self.metadata["files"][file_id]

        # 删除物理文件
        file_path = Path(file_info["file_path"])
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"[WARN] 删除物理文件失败: {e}")

        # 删除元数据
        file_hash = file_info.get("file_hash", "")
        if file_hash in self.metadata.get("file_hashes", {}):
            del self.metadata["file_hashes"][file_hash]

        del self.metadata["files"][file_id]
        self._save_metadata()

        return {
            "success": True,
            "message": f"文件 {file_info['filename']} 已删除",
        }

    def get_file_chunks(self, file_id: str) -> list[Document]:
        """
        获取指定文件的所有文档片段

        Args:
            file_id: 文件ID

        Returns:
            list[Document]: 文档片段列表
        """
        if file_id not in self.metadata.get("files", {}):
            return []

        file_info = self.metadata["files"][file_id]
        file_path = Path(file_info["file_path"])

        if not file_path.exists():
            return []

        try:
            # 解析文件
            documents = self.parser.parse(
                str(file_path),
                metadata={"category": file_info.get("category", ""), "file_id": file_id},
            )

            # 切分文档
            chunks = self.splitter.split_documents(documents)
            return chunks
        except Exception:
            return []

    def get_all_chunks(self) -> list[Document]:
        """
        获取所有已上传文件的文档片段

        Returns:
            list[Document]: 所有文档片段列表
        """
        all_chunks = []

        for file_id in self.metadata.get("files", {}):
            chunks = self.get_file_chunks(file_id)
            all_chunks.extend(chunks)

        return all_chunks
