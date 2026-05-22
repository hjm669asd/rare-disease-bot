"""
程序运行流程图：

构建同一个 FAISS 向量库
   ↓
方式一：直接调用 vectorstore.similarity_search()
   ↓
方式二：先 as_retriever()，再 retriever.invoke()
   ↓
对比两种方式返回的 Document

本脚本的教学目标：
理解 VectorStore 和 Retriever 的区别：VectorStore 更底层，Retriever 更适合接入完整 RAG 链路。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入。", metadata={"topic": "高血压饮食"}),
    Document(page_content="高血压患者如果出现胸闷、视物模糊，应及时就医。", metadata={"topic": "高血压危险信号"}),
    Document(page_content="普通感冒患者应注意休息。", metadata={"topic": "感冒护理"}),
]


def print_documents(title, documents):
    """统一打印 Document 列表，方便对比两种检索方式。"""

    print(f"\n========== {title} ==========")

    for index, document in enumerate(documents, start=1):
        print(f"\n--- 结果 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)


def main():
    """对比 VectorStore 和 Retriever 的调用方式。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者饮食要注意什么？"

    # 方式一：直接使用 VectorStore 的相似度检索。
    vectorstore_results = vectorstore.similarity_search(question, k=2)

    # 方式二：把 VectorStore 包装成 Retriever。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retriever_results = retriever.invoke(question)

    print(f"用户问题: {question}")
    print_documents("VectorStore similarity_search 结果", vectorstore_results)
    print_documents("Retriever invoke 结果", retriever_results)

    # 观察重点：
    # 1. 两种方式都能返回 Document。
    # 2. Retriever 是更统一、更适合链式编排的接口。
    # 3. 后续构建 RAG Chain 时，通常优先使用 Retriever。


if __name__ == "__main__":
    main()
