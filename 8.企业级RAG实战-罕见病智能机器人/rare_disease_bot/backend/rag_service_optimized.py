"""优化版RAG服务 - 集成意图识别、结构化查询和混合检索"""

import os
from pathlib import Path
from operator import itemgetter
from typing import Any

from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_dashscope import DashScopeEmbeddings

from config import (
    CHAT_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DASHSCOPE_API_KEY,
    DATASET_PATH,
    EMBEDDING_MODEL_NAME,
    LIGHTWEIGHT_MODEL_NAME,
    MEDICAL_DISCLAIMER,
    RETRIEVAL_TOP_K,
)
from intent_router import IntentionRouter, IntentionType
from structured_query import StructuredQueryExtractor
from hybrid_chunker import HybridChunker
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from file_uploader import FileUploader


class OptimizedRAGService:
    """优化版RAG服务"""

    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("请先在项目根目录 .env 文件中配置 DASHSCOPE_API_KEY")
        if not Path(DATASET_PATH).exists():
            raise FileNotFoundError(f"未找到罕见病数据集：{DATASET_PATH}")

        os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

        # 初始化强模型（用于RAG生成）
        self.llm = ChatTongyi(model=CHAT_MODEL_NAME, temperature=0)

        # 初始化弱模型（用于意图识别、rerank等轻量任务）
        self.lightweight_llm = ChatTongyi(model=LIGHTWEIGHT_MODEL_NAME, temperature=0)

        # 初始化意图路由器（使用弱模型）
        self.intent_router = IntentionRouter(self.lightweight_llm)

        # 初始化混合切分器
        self.chunker = HybridChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # 加载数据并构建索引
        self.disease_names = self.chunker.get_disease_names(Path(DATASET_PATH))
        self.structured_query_extractor = StructuredQueryExtractor(
            llm=self.lightweight_llm,  # 使用弱模型
            disease_names=self.disease_names,
        )

        # 构建向量库
        chunks = self.chunker.chunk_excel(Path(DATASET_PATH))
        self.primary_docs = chunks["primary"]
        self.secondary_docs = chunks["secondary"]

        # FAISS 持久化目录（使用相对路径，避免中文绝对路径问题）
        self.faiss_dir = Path("storage") / "faiss_index"
        primary_faiss_path = self.faiss_dir / "primary"
        secondary_faiss_path = self.faiss_dir / "secondary"
        primary_faiss_path.mkdir(parents=True, exist_ok=True)
        secondary_faiss_path.mkdir(parents=True, exist_ok=True)

        # 尝试从磁盘加载主索引
        self.primary_vectorstore = None
        if primary_faiss_path.exists():
            try:
                embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
                self.primary_vectorstore = FAISS.load_local(
                    str(primary_faiss_path), embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[OK] 从磁盘加载主索引向量库 ({len(self.primary_docs)} 文档)")
            except Exception as e:
                print(f"[WARN] 加载主索引缓存失败，将重新构建: {e}")

        # 磁盘缓存未命中，构建并保存
        if self.primary_vectorstore is None:
            try:
                self.primary_vectorstore = self._build_vectorstore(self.primary_docs)
                self.primary_vectorstore.save_local(str(primary_faiss_path))
                print(f"[OK] 主索引构建完成并已缓存到磁盘")
            except Exception as e:
                print(f"[WARN] 主索引构建失败（可能embedding额度不足）: {e}")
                self.primary_vectorstore = None

        # 尝试从磁盘加载辅助索引
        self.secondary_vectorstore = None
        if self.secondary_docs and self.primary_vectorstore is not None:
            if secondary_faiss_path.exists():
                try:
                    embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
                    self.secondary_vectorstore = FAISS.load_local(
                        str(secondary_faiss_path), embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    print(f"[OK] 从磁盘加载辅助索引向量库 ({len(self.secondary_docs)} 文档)")
                except Exception as e:
                    print(f"[WARN] 加载辅助索引缓存失败，将重新构建: {e}")

            if self.secondary_vectorstore is None:
                try:
                    self.secondary_vectorstore = self._build_vectorstore(self.secondary_docs)
                    self.secondary_vectorstore.save_local(str(secondary_faiss_path))
                    print(f"[OK] 辅助索引构建完成并已缓存到磁盘")
                except Exception as e:
                    print(f"[WARN] 辅助索引构建失败，将仅使用主索引: {e}")

        # 初始化混合检索器
        self.retriever = HybridRetriever(
            primary_vectorstore=self.primary_vectorstore,
            secondary_vectorstore=self.secondary_vectorstore,
            primary_docs=self.primary_docs,
            secondary_docs=self.secondary_docs,
        )

        # 初始化Reranker（使用弱模型）
        self.reranker = Reranker(top_k=3)

        # 初始化文件上传管理器
        self.file_uploader = FileUploader(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # 构建Prompt和RAG链
        self.prompt = self._build_prompt()
        self.common_disease_prompt = self._build_common_disease_prompt()
        self.chain = self._build_chain()

        print(f"[OK] RAG服务初始化完成")
        print(f"   - 强模型: {CHAT_MODEL_NAME} (用于RAG生成)")
        print(f"   - 弱模型: {LIGHTWEIGHT_MODEL_NAME} (用于意图识别/rerank)")
        print(f"   - 主索引文档数: {len(self.primary_docs)}")
        print(f"   - 辅助索引文档数: {len(self.secondary_docs)}")
        print(f"   - 疾病名称数: {len(self.disease_names)}")

    def add_documents_to_vectorstore(self, documents: list[Document]):
        """
        动态添加文档到向量库

        Args:
            documents: 要添加的文档列表
        """
        if not documents:
            return

        embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
        batch_size = 10  # embedding-v3 限制

        # 分批添加到向量库
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            if batch:
                try:
                    self.primary_vectorstore.add_documents(batch)
                except Exception as e:
                    print(f"[ERROR] 添加文档到向量库失败: {e}")

        # 持久化到磁盘
        try:
            self.primary_vectorstore.save_local(str(Path("storage") / "faiss_index" / "primary"))
        except Exception as e:
            print(f"[WARN] 保存向量库到磁盘失败: {e}")

        print(f"[OK] 已添加 {len(documents)} 个文档到向量库")

    def upload_and_add_to_knowledge_base(
        self,
        file_content: bytes,
        filename: str,
        category: str = "用户上传",
    ) -> dict:
        """
        上传文件并添加到知识库

        Args:
            file_content: 文件内容
            filename: 文件名
            category: 文件分类

        Returns:
            dict: 上传结果
        """
        # 1. 上传文件并解析
        upload_result = self.file_uploader.upload_file(file_content, filename, category)

        if not upload_result["success"]:
            return upload_result

        # 2. 获取解析后的文档片段
        file_id = upload_result["file_id"]
        chunks = self.file_uploader.get_file_chunks(file_id)

        if not chunks:
            return {
                "success": False,
                "error": "文件解析结果为空",
            }

        # 3. 添加到向量库
        self.add_documents_to_vectorstore(chunks)

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "doc_count": upload_result.get("doc_count", 0),
            "chunk_count": len(chunks),
            "message": f"文件上传成功并已添加到知识库，共 {len(chunks)} 个片段",
        }

    def get_uploaded_files(self) -> list[dict]:
        """获取已上传的文件列表"""
        return self.file_uploader.get_uploaded_files()

    def delete_uploaded_file(self, file_id: str) -> dict:
        """删除已上传的文件"""
        return self.file_uploader.delete_file(file_id)

    def answer(self, question: str, history: str, debug: bool = False) -> dict[str, Any]:
        """
        处理用户问题

        Args:
            question: 用户问题
            history: 对话历史
            debug: 是否返回调试信息

        Returns:
            dict: 包含answer, sources, debug_info的结果
        """
        # 1. 意图识别
        intention = self.intent_router.classify(question)

        # 2. 如果是非RAG意图，处理非罕见病问题
        if intention != IntentionType.MEDICAL_RAG:
            # 常见病：调用模型自身知识回答
            if intention == IntentionType.COMMON_DISEASE:
                return self._handle_common_disease(question, intention, debug)

            # 其他非RAG意图：返回预设回复
            non_rag_response = self.intent_router.get_response_for_non_rag(intention, question)
            if non_rag_response:
                return {
                    "answer": non_rag_response,
                    "sources": [],
                    "debug_info": {
                        "intention": intention.value,
                        "intention_label": self._get_intention_label(intention),
                        "rag_called": False,
                        "knowledge_source": "模型自身知识",
                    } if debug else None,
                }

        # 3. 结构化查询提取
        structured_query = self.structured_query_extractor.extract(question)

        # 4. 混合检索
        optimal_top_k = self._get_optimal_top_k(structured_query)
        initial_documents = self.retriever.retrieve(
            structured_query,
            default_top_k=optimal_top_k * 2,  # 多检索一些，给rerank用
        )

        # 5. Rerank重排序（使用弱模型）
        retrieved_documents = self.reranker.rerank(
            query=question,
            documents=initial_documents,
            top_k=optimal_top_k,
        )

        # 6. 格式化上下文
        context = self._format_documents(retrieved_documents)

        # 7. 调用强模型生成回答
        answer = self.chain.invoke({
            "context": context,
            "history": history,
            "question": question,
        })

        # 7. 添加医疗免责声明
        if MEDICAL_DISCLAIMER not in answer:
            answer = f"{answer.rstrip()}\n\n{MEDICAL_DISCLAIMER}"

        # 8. 构建结果
        result = {
            "answer": answer,
            "sources": self._build_sources(retrieved_documents),
            "debug_info": None,
        }

        if debug:
            result["debug_info"] = {
                "intention": intention.value,
                "intention_label": self._get_intention_label(intention),
                "rag_called": True,
                "models_used": {
                    "intent_recognition": LIGHTWEIGHT_MODEL_NAME,
                    "rerank": "gte-rerank-v2",
                    "rag_generation": CHAT_MODEL_NAME,
                },
                "structured_query": {
                    "disease_name": structured_query.disease_name,
                    "query_type": structured_query.query_type,
                    "keywords": structured_query.keywords,
                },
                "optimal_top_k": optimal_top_k,
                "initial_retrieved": len(initial_documents),
                "reranked_to": len(retrieved_documents),
                "retrieved_documents": [
                    {
                        "content_preview": doc.page_content[:220],
                        "metadata": doc.metadata,
                    }
                    for doc in retrieved_documents
                ],
            }

        return result

    def answer_streaming(self, question: str, history: str, debug: bool = False):
        """
        流式输出版本的answer方法 — 生成器，逐个 yield 事件字典。

        yield 的事件格式：
          {"type": "status",     "content": "..."}
          {"type": "debug",      "content": {...}}
          {"type": "sources",    "content": [...]}
          {"type": "answer",     "content": "..."}   # 逐 token
          {"type": "done",       "content": ""}
        """
        # 1. 意图识别
        yield {"type": "status", "content": "正在分析问题意图..."}
        intention = self.intent_router.classify(question)

        # 2. 非 RAG 意图 → 流式生成回答
        if intention != IntentionType.MEDICAL_RAG:
            debug_info = {
                "intention": intention.value,
                "intention_label": self._get_intention_label(intention),
                "rag_called": False,
                "knowledge_source": "模型自身知识",
            } if debug else None
            yield {"type": "debug", "content": debug_info}

            if intention == IntentionType.COMMON_DISEASE:
                # 常见病：用强模型流式生成
                yield {"type": "status", "content": "正在生成回答..."}
                common_disease_chain = self.common_disease_prompt | self.llm | StrOutputParser()
                full_answer = ""
                for chunk in common_disease_chain.stream({"question": question}):
                    full_answer += chunk
                    yield {"type": "answer", "content": chunk}
                # 追加声明
                disclaimer = "声明：以上回答来源于AI模型的通用医学知识，不属于罕见病知识库。如有不适，请及时咨询正规医疗机构专业医生。"
                if disclaimer not in full_answer:
                    yield {"type": "answer", "content": f"\n\n{disclaimer}"}
            else:
                # 问候/闲聊/超出范围：用弱模型流式生成
                non_rag_response = self.intent_router.get_response_for_non_rag(intention, question)
                if non_rag_response:
                    yield {"type": "answer", "content": non_rag_response}
                else:
                    # 兜底：用弱模型流式回答
                    yield {"type": "status", "content": "正在生成回答..."}
                    fallback_prompt = PromptTemplate.from_template(
                        "你是一个友善的助手。请简短回答用户的问题。\n\n用户问题：{question}\n\n回答："
                    )
                    fallback_chain = fallback_prompt | self.lightweight_llm | StrOutputParser()
                    for chunk in fallback_chain.stream({"question": question}):
                        yield {"type": "answer", "content": chunk}

            yield {"type": "done", "content": ""}
            return

        # ---- 以下为 RAG 流程 ----

        # 3. 结构化查询提取
        yield {"type": "status", "content": "正在提取结构化查询..."}
        structured_query = self.structured_query_extractor.extract(question)

        # 4. 混合检索
        yield {"type": "status", "content": "正在检索相关文档..."}
        optimal_top_k = self._get_optimal_top_k(structured_query)
        initial_documents = self.retriever.retrieve(
            structured_query,
            default_top_k=optimal_top_k * 2,
        )

        # 5. Rerank 重排序
        yield {"type": "status", "content": "正在重排序..."}
        retrieved_documents = self.reranker.rerank(
            query=question,
            documents=initial_documents,
            top_k=optimal_top_k,
        )

        # 6. 格式化上下文
        context = self._format_documents(retrieved_documents)

        # 7. 构建 debug 信息（在流式生成前发送）
        debug_info = None
        if debug:
            debug_info = {
                "intention": intention.value,
                "intention_label": self._get_intention_label(intention),
                "rag_called": True,
                "knowledge_source": "罕见病知识库",
                "models_used": {
                    "intent_recognition": LIGHTWEIGHT_MODEL_NAME,
                    "rerank": "gte-rerank-v2",
                    "rag_generation": CHAT_MODEL_NAME,
                },
                "structured_query": {
                    "disease_name": structured_query.disease_name,
                    "query_type": structured_query.query_type,
                    "keywords": structured_query.keywords,
                },
                "optimal_top_k": optimal_top_k,
                "initial_retrieved": len(initial_documents),
                "reranked_to": len(retrieved_documents),
                "retrieved_documents": [
                    {
                        "content_preview": doc.page_content[:220],
                        "metadata": doc.metadata,
                    }
                    for doc in retrieved_documents
                ],
            }

        sources = self._build_sources(retrieved_documents)

        yield {"type": "debug", "content": debug_info}
        yield {"type": "sources", "content": sources}
        yield {"type": "status", "content": "正在生成回答..."}

        # 8. 调用强模型 — 真正的 token 级流式
        full_answer = ""
        for chunk in self.chain.stream({
            "context": context,
            "history": history,
            "question": question,
        }):
            full_answer += chunk
            yield {"type": "answer", "content": chunk}

        # 9. 追加医疗免责声明
        if MEDICAL_DISCLAIMER not in full_answer:
            disclaimer_chunk = f"\n\n{MEDICAL_DISCLAIMER}"
            full_answer += disclaimer_chunk
            yield {"type": "answer", "content": disclaimer_chunk}

        yield {"type": "done", "content": ""}

    def _get_intention_label(self, intention: IntentionType) -> str:
        """获取意图标签"""
        labels = {
            IntentionType.GREETING: "问候/闲聊",
            IntentionType.SIMPLE_QA: "简单问答",
            IntentionType.COMMON_DISEASE: "常见病(模型知识回答)",
            IntentionType.MEDICAL_RAG: "医疗专业问答(罕见病)",
            IntentionType.OUT_OF_SCOPE: "超出范围",
        }
        return labels.get(intention, "未知")

    def _handle_common_disease(self, question: str, intention: IntentionType, debug: bool) -> dict[str, Any]:
        """处理常见病问题 - 调用模型自身知识回答"""
        # 使用常见病Prompt调用LLM
        common_disease_chain = self.common_disease_prompt | self.llm | StrOutputParser()
        result_answer = common_disease_chain.invoke({"question": question})

        return {
            "answer": result_answer,
            "sources": [],
            "debug_info": {
                "intention": intention.value,
                "intention_label": self._get_intention_label(intention),
                "rag_called": False,
                "knowledge_source": "模型自身通用医学知识(非知识库)",
            } if debug else None,
        }

    def _get_optimal_top_k(self, structured_query) -> int:
        """根据查询结构获取最优top_k"""
        # 如果有明确的疾病名称，只需要2个结果
        if structured_query.disease_name:
            return 2

        # 如果有明确的查询类型，可以减少结果数
        if structured_query.query_type:
            return 3

        # 默认返回4个结果
        return RETRIEVAL_TOP_K

    def _build_vectorstore(self, documents: list[Document]) -> FAISS:
        """构建FAISS向量库"""
        if not documents:
            return None

        embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
        batch_size = 10  # embedding-v3 限制

        # 分批构建向量库
        vectorstore = FAISS.from_documents(documents[:batch_size], embeddings)

        for start in range(batch_size, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            if batch:
                vectorstore.add_documents(batch)

        return vectorstore

    def _build_prompt(self) -> PromptTemplate:
        """构建Prompt模板"""
        return PromptTemplate.from_template("""
你是一个罕见病智能科普机器人。

请严格遵守以下规则：
1. 只能根据"罕见病知识库资料"回答。
2. 如果资料不足，请明确说明无法根据现有资料可靠回答。
3. 不要给出确诊结论。
4. 不要给出个体化用药方案。
5. 如果问题涉及诊断、治疗、用药或病情判断，必须建议咨询正规医疗机构专业医生。
6. 回答最后必须包含医疗安全声明：{disclaimer}

对话历史：
{history}

罕见病知识库资料：
{context}

用户当前问题：
{question}

请用清晰、谨慎、适合科普的中文回答。
""".replace("{disclaimer}", MEDICAL_DISCLAIMER))

    def _build_common_disease_prompt(self) -> PromptTemplate:
        """构建常见病问答Prompt模板"""
        return PromptTemplate.from_template("""
你是一个医疗健康科普助手。

用户询问的是常见疾病问题。请根据你的医学知识回答，但必须遵守以下规则：

1. 明确告知用户：以下回答来源于AI模型的通用医学知识，不属于罕见病知识库。
2. 提供准确、通俗易懂的科普信息。
3. 不要给出确诊结论。
4. 不要给出个体化用药方案。
5. 如果问题涉及诊断、治疗、用药或病情判断，必须建议咨询正规医疗机构专业医生。
6. 回答最后必须包含以下声明：
   "声明：以上回答来源于AI模型的通用医学知识，不属于罕见病知识库。如有不适，请及时咨询正规医疗机构专业医生。"

用户问题：{question}

请用清晰、谨慎、适合科普的中文回答。
""")

    def _build_chain(self):
        """构建RAG链"""
        return (
            {
                "context": itemgetter("context"),
                "history": itemgetter("history"),
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_documents(self, documents: list[Document]) -> str:
        """格式化检索到的文档"""
        if not documents:
            return "未检索到相关资料。"
        return "\n\n---\n\n".join(doc.page_content for doc in documents)

    def _build_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        """构建来源引用"""
        sources = []
        seen = set()

        for doc in documents:
            metadata = doc.metadata
            key = (
                metadata.get("source", ""),
                metadata.get("disease_name_zh", ""),
                metadata.get("row_index"),
            )

            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "source": metadata.get("source", ""),
                "disease_name_zh": metadata.get("disease_name_zh", ""),
                "disease_name_en": metadata.get("disease_name_en", ""),
                "category": metadata.get("category", ""),
                "row_index": metadata.get("row_index"),
            })

        return sources


# 保持向后兼容
RAGService = OptimizedRAGService
