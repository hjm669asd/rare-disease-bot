from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="demo-session", min_length=1)
    question: str = Field(..., min_length=1)
    debug: bool = False


class SourceItem(BaseModel):
    source: str
    disease_name_zh: str = ""
    disease_name_en: str = ""
    category: str = ""
    row_index: int | None = None


class RetrievedDocumentItem(BaseModel):
    content_preview: str
    metadata: dict[str, Any]


class StructuredQueryInfo(BaseModel):
    """结构化查询信息"""
    disease_name: str | None = None
    query_type: str | None = None
    keywords: list[str] = []


class ModelsUsedInfo(BaseModel):
    """模型使用信息"""
    intent_recognition: str = ""    # 意图识别模型
    rerank: str = ""                # Rerank模型
    rag_generation: str = ""        # RAG生成模型


class DebugInfo(BaseModel):
    intention: str = ""                    # 意图类型
    intention_label: str = ""              # 意图标签
    rag_called: bool = True                # 是否调用了RAG
    knowledge_source: str = ""             # 知识来源
    models_used: ModelsUsedInfo | None = None  # 模型使用信息
    structured_query: StructuredQueryInfo | None = None  # 结构化查询信息
    optimal_top_k: int = 0                 # 最优top_k值
    top_k: int = 0                         # 兼容原版
    initial_retrieved: int = 0             # 初始检索数量
    reranked_to: int = 0                   # Rerank后数量
    retrieved_count: int = 0               # 兼容原版
    retrieved_documents: list[RetrievedDocumentItem] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceItem]
    debug_info: DebugInfo | None = None


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]
