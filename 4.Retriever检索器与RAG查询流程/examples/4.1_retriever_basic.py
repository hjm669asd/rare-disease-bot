"""
程序运行流程图：

读取 .env 中的 DASHSCOPE_API_KEY
   ↓
准备几个医疗 Document
   ↓
创建 DashScopeEmbeddings
   ↓
使用 FAISS.from_documents() 构建向量库
   ↓
使用 as_retriever() 转成 Retriever
   ↓
调用 retriever.invoke() 检索相关文档
   ↓
打印检索结果

本脚本的教学目标：
理解 Retriever 是 RAG 中的检索器接口，它接收用户问题，返回相关 Document。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入。", metadata={"topic": "高血压饮食"}),
    Document(page_content="糖尿病患者应监测血糖。", metadata={"topic": "糖尿病监测"}),
]


def main():
    """演示 Retriever 的最基础用法。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 创建文本向量模型。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # 构建 FAISS 向量库。
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    # 把 VectorStore 包装成 Retriever。
    # k=1 表示只返回最相关的 1 条文档。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    question = "高血压患者饮食要注意什么？"

    # Retriever 的核心调用方式。
    results = retriever.invoke(question)

    print(f"用户问题: {question}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    # 观察重点：
    # 1. retriever.invoke() 的输入是用户问题。
    # 2. 返回结果是 Document 列表。
    # 3. Retriever 是后续 RAG 链路中最常用的检索接口。


if __name__ == "__main__":
    main()
