"""
程序运行流程图：

准备医疗知识 Documents
   ↓
构建 FAISS 向量库
   ↓
输入用户问题
   ↓
调用 similarity_search_with_score()
   ↓
同时打印 Document 和 score
   ↓
观察分数与相关性的关系

本脚本的教学目标：
理解检索结果不只有文本，还可以查看分数。分数能帮助我们判断结果是否足够可靠。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入，少吃腌制食品。", metadata={"topic": "高血压饮食"}),
    Document(page_content="高血压患者应定期监测血压。", metadata={"topic": "高血压监测"}),
    Document(page_content="感冒患者应注意休息，多饮水。", metadata={"topic": "感冒护理"}),
]


def main():
    """演示带分数的相似度检索。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者饮食要注意什么？"

    # similarity_search_with_score() 返回 (Document, score) 元组列表。
    # 在当前 FAISS 设置下，通常 score 越小表示距离越近。
    results = vectorstore.similarity_search_with_score(question, k=3)

    print(f"用户问题: {question}")

    for index, (document, score) in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(f"score: {score}")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    # 观察重点：
    # 1. 第一条一般分数更好，也更相关。
    # 2. 分数差的内容即使被返回，也不一定适合放入 prompt。
    # 3. 医疗场景中可以设置阈值过滤低质量检索结果。


if __name__ == "__main__":
    main()
