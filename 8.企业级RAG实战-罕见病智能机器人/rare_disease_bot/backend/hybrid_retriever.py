"""混合检索器 - 结合精确匹配和向量检索"""

from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from structured_query import StructuredQuery


class HybridRetriever:
    """混合检索器 - 结合精确匹配、字段过滤和向量检索"""

    def __init__(
        self,
        primary_vectorstore: FAISS,
        secondary_vectorstore: Optional[FAISS] = None,
        primary_docs: list[Document] = None,
        secondary_docs: list[Document] = None,
    ):
        """
        初始化混合检索器

        Args:
            primary_vectorstore: 主索引向量库（完整疾病信息）
            secondary_vectorstore: 辅助索引向量库（按字段拆分）
            primary_docs: 主索引文档列表（用于精确匹配）
            secondary_docs: 辅助索引文档列表（用于精确匹配）
        """
        self.primary_vectorstore = primary_vectorstore
        self.secondary_vectorstore = secondary_vectorstore
        self.primary_docs = primary_docs or []
        self.secondary_docs = secondary_docs or []

        # 构建疾病名称索引（用于精确匹配）
        self._build_disease_index()

    def _build_disease_index(self):
        """构建疾病名称索引"""
        self.disease_index = {}  # disease_name -> Document

        for doc in self.primary_docs:
            disease_name = doc.metadata.get("disease_name_zh", "")
            if disease_name:
                self.disease_index[disease_name] = doc

    def retrieve(
        self,
        structured_query: StructuredQuery,
        default_top_k: int = 4,
    ) -> list[Document]:
        """
        根据结构化查询检索文档

        Args:
            structured_query: 结构化查询对象
            default_top_k: 默认返回文档数

        Returns:
            list[Document]: 检索到的文档列表
        """
        results = []
        seen_content = set()  # 用于去重

        # 1. 疾病名精确匹配（如果有）
        if structured_query.disease_name:
            exact_match = self._exact_match_disease(structured_query.disease_name)
            if exact_match:
                content_key = exact_match.page_content[:100]
                if content_key not in seen_content:
                    results.append(exact_match)
                    seen_content.add(content_key)

        # 2. 辅助索引检索（如果查询类型明确）
        if self.secondary_vectorstore and structured_query.query_type:
            secondary_results = self._retrieve_by_query_type(
                structured_query,
                top_k=min(2, default_top_k),
            )
            for doc in secondary_results:
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    results.append(doc)
                    seen_content.add(content_key)

        # 3. 主索引向量检索
        search_query = self._build_search_query(structured_query)
        primary_results = self._vector_search(
            self.primary_vectorstore,
            search_query,
            top_k=default_top_k,
        )
        for doc in primary_results:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                results.append(doc)
                seen_content.add(content_key)

        # 4. 如果结果不够，从辅助索引补充
        if len(results) < default_top_k and self.secondary_vectorstore:
            supplementary_results = self._vector_search(
                self.secondary_vectorstore,
                search_query,
                top_k=default_top_k - len(results),
            )
            for doc in supplementary_results:
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    results.append(doc)
                    seen_content.add(content_key)

        return results[:default_top_k]

    def _exact_match_disease(self, disease_name: str) -> Optional[Document]:
        """精确匹配疾病名称"""
        # 直接匹配
        if disease_name in self.disease_index:
            return self.disease_index[disease_name]

        # 模糊匹配（包含关系）
        for name, doc in self.disease_index.items():
            if disease_name in name or name in disease_name:
                return doc

        return None

    def _retrieve_by_query_type(
        self,
        structured_query: StructuredQuery,
        top_k: int = 2,
    ) -> list[Document]:
        """根据查询类型从辅助索引检索"""
        if not self.secondary_vectorstore or not structured_query.query_type:
            return []

        # 构建查询：疾病名 + 查询类型
        search_parts = []
        if structured_query.disease_name:
            search_parts.append(structured_query.disease_name)
        search_parts.append(structured_query.query_type)

        search_query = " ".join(search_parts)

        # 向量检索
        results = self._vector_search(
            self.secondary_vectorstore,
            search_query,
            top_k=top_k * 2,  # 多检索一些，后面过滤
        )

        # 过滤：只保留匹配查询类型的文档
        filtered_results = []
        for doc in results:
            doc_query_types = doc.metadata.get("query_types", [])
            if structured_query.query_type in doc_query_types:
                filtered_results.append(doc)
                if len(filtered_results) >= top_k:
                    break

        return filtered_results

    def _vector_search(
        self,
        vectorstore: FAISS,
        query: str,
        top_k: int = 4,
    ) -> list[Document]:
        """向量检索"""
        try:
            return vectorstore.similarity_search(query, k=top_k)
        except Exception:
            return []

    def _build_search_query(self, structured_query: StructuredQuery) -> str:
        """构建搜索查询字符串"""
        parts = []

        if structured_query.disease_name:
            parts.append(structured_query.disease_name)

        if structured_query.query_type:
            parts.append(structured_query.query_type)

        # 添加关键词
        for keyword in structured_query.keywords:
            if keyword not in parts:
                parts.append(keyword)

        return " ".join(parts) if parts else structured_query.original_question
