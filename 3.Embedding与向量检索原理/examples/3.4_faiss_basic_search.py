"""
程序运行流程图：

读取 .env 中的 API Key
   ↓
准备多个医疗知识 Document
   ↓
创建 DashScopeEmbeddings
   ↓
使用 FAISS.from_documents() 构建向量库
   ↓
输入一个用户问题
   ↓
使用 similarity_search() 检索相似 chunk
   ↓
打印检索结果

本脚本的教学目标：
理解 FAISS 在 RAG 中的作用：保存 chunk 向量，并根据用户问题向量返回最相似的 chunk。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入，少吃腌制食品。", metadata={"topic": "高血压饮食"}),
    Document(page_content="糖尿病患者应监测空腹血糖和餐后血糖。", metadata={"topic": "糖尿病监测"}),
    Document(page_content="感冒患者应注意休息，适量饮水。", metadata={"topic": "感冒护理"}),
]


def main():
    """演示 FAISS 基础相似度检索。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 创建 embedding 模型。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # FAISS.from_documents() 会自动完成：
    # 1. 把每个 Document 的 page_content 转成向量。
    # 2. 把向量存入 FAISS 索引。
    # 3. 保留 Document 原文和 metadata，方便检索后返回。
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者饮食要注意什么？"

    # similarity_search() 会把 question 转成向量，然后去 FAISS 中找相似内容。
    results = vectorstore.similarity_search(question, k=2)

    print(f"用户问题: {question}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    # 观察重点：
    # 1. 第一条通常应该命中高血压饮食。
    # 2. 第二条可能不是完全相关，说明 k 越大越可能带入噪声。


if __name__ == "__main__":
    main()
