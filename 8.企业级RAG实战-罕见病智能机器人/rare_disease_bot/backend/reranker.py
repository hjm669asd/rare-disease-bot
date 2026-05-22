"""Reranker重排序模块 - 使用gte-rerank-v2对检索结果进行二次排序"""

from typing import Optional

from langchain_core.documents import Document


class Reranker:
    """Reranker重排序器 - 使用DashScope gte-rerank-v2模型"""

    def __init__(self, llm=None, top_k: int = 3):
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        if not documents:
            return []

        k = top_k or self.top_k

        if len(documents) <= k:
            return documents

        try:
            return self._model_rerank(query, documents, k)
        except Exception as e:
            print(f"[WARN] gte-rerank-v2 failed, using fallback: {e}")
            return self._fallback_rerank(query, documents, k)

    def _model_rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        from dashscope import TextReRank

        doc_texts = [doc.page_content[:500] for doc in documents]

        response = TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=doc_texts,
            top_k=top_k,
            return_documents=False,
        )

        if response.status_code != 200:
            raise Exception(f"Rerank API error: {response.code} - {response.message}")

        results = response.output.results
        ranked_docs = [documents[r.index] for r in results]
        return ranked_docs

    def _fallback_rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """回退：基于关键词匹配的简单排序"""
        query_lower = query.lower()
        scored_docs = []

        for doc in documents:
            score = 0
            content_lower = doc.page_content.lower()
            disease_name = doc.metadata.get("disease_name_zh", "").lower()

            if disease_name and disease_name in query_lower:
                score += 10

            for word in query_lower.split():
                if len(word) >= 2 and word in content_lower:
                    score += 2

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


class SimpleReranker:
    """简单重排序器（不需要模型，基于规则）"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        k = top_k or self.top_k

        if len(documents) <= k:
            return documents

        query_lower = query.lower()
        scored_docs = []

        for doc in documents:
            score = 0
            content_lower = doc.page_content.lower()
            disease_name = doc.metadata.get("disease_name_zh", "").lower()

            if disease_name and disease_name in query_lower:
                score += 20
            elif disease_name:
                for char in disease_name:
                    if char in query_lower:
                        score += 2

            for word in query_lower.split():
                if len(word) >= 2:
                    if word in content_lower:
                        score += 3
                    if word in disease_name:
                        score += 5

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
