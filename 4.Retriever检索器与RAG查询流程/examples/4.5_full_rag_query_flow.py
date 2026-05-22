"""
程序运行流程图：

读取 .env 中的 DASHSCOPE_API_KEY
   ↓
准备医疗知识 Document
   ↓
构建 Embedding 模型
   ↓
构建 FAISS 向量库
   ↓
转换成 Retriever
   ↓
用户提出问题
   ↓
Retriever 检索相关文档
   ↓
拼接 context 并构造 prompt
   ↓
调用 Qwen 大模型生成回答
   ↓
打印最终答案

本脚本的教学目标：
跑通“用户问题 → Retriever 检索 → Prompt 构造 → LLM 回答”的完整 RAG 查询流程。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入，少吃腌制食品和高盐零食。", metadata={"topic": "高血压饮食"}),
    Document(page_content="高血压患者如果出现明显头痛、胸闷、视物模糊、肢体无力，或血压持续明显升高，应及时就医。", metadata={"topic": "高血压危险信号"}),
    Document(page_content="高血压患者应定期测量血压，记录血压变化。", metadata={"topic": "高血压监测"}),
]


def build_prompt(question, documents):
    """根据检索结果和用户问题构造最终 prompt。"""

    context = "\n\n".join(document.page_content for document in documents)

    return f"""
你是一个智能医疗科普助手。
请严格根据下面资料回答问题，不要编造资料中没有的信息。
如果资料不足，请说明无法根据现有资料可靠回答。

资料：
{context}

用户问题：
{question}

回答要求：
1. 先直接回答用户问题。
2. 如果涉及风险症状，要提醒及时就医。
3. 最后说明本回答仅用于健康科普，不能替代医生诊断。
"""


def main():
    """运行一个完整的 RAG 查询示例。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 1. 创建 Embedding 模型，用于把文档和问题转换成向量。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # 2. 把文档写入 FAISS 向量库。
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    # 3. 把向量库转换成 Retriever。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. 创建通义千问聊天模型。
    # temperature=0 表示尽量稳定、少随机发挥。
    llm = ChatTongyi(model="qwen-turbo", temperature=0)

    question = "高血压患者什么时候需要及时就医？"

    # 5. 使用 Retriever 检索相关资料。
    documents = retriever.invoke(question)

    print("========== 检索到的资料 ==========")
    for index, document in enumerate(documents, start=1):
        print(f"\n--- 资料 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    # 6. 构造 prompt。
    prompt = build_prompt(question, documents)

    # 7. 调用大模型生成回答。
    answer = llm.invoke(prompt)

    print("\n========== 大模型回答 ==========")
    print(answer.content)

    # 观察重点：
    # 1. Retriever 决定了大模型能看到哪些资料。
    # 2. Prompt 决定了大模型如何使用这些资料。
    # 3. 医疗场景必须保留安全声明。


if __name__ == "__main__":
    main()
