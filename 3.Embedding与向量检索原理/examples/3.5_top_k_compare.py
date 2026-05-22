"""
程序运行流程图：

准备医疗知识 Documents
   ↓
构建 FAISS 向量库
   ↓
固定同一个用户问题
   ↓
分别设置 k=1、k=2、k=3
   ↓
观察返回结果数量和内容变化

本脚本的教学目标：
理解 top_k 参数如何影响召回数量、上下文完整性和噪声比例。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者如果出现明显头痛、胸闷、视物模糊，应及时就医。", metadata={"topic": "高血压危险信号"}),
    Document(page_content="高血压患者应定期监测血压，并记录血压变化。", metadata={"topic": "高血压监测"}),
    Document(page_content="糖尿病患者应监测空腹血糖和餐后血糖。", metadata={"topic": "糖尿病监测"}),
    Document(page_content="普通感冒患者应注意休息，适量饮水。", metadata={"topic": "感冒护理"}),
]


def main():
    """对比不同 k 值返回的检索结果。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者什么时候需要及时就医？"

    # 依次比较不同 k 值。
    for k in [1, 2, 3]:
        results = vectorstore.similarity_search(question, k=k)

        print(f"\n========== k={k} ==========")
        print(f"返回结果数量: {len(results)}")

        for index, document in enumerate(results, start=1):
            print(f"\n--- 结果 {index} ---")
            print(f"metadata: {document.metadata}")
            print(document.page_content)

    # 观察重点：
    # 1. k=1 噪声少，但可能信息不足。
    # 2. k=3 召回更多，但可能带入糖尿病或感冒等无关信息。
    # 3. 医疗 RAG 中需要在召回和噪声之间做平衡。


if __name__ == "__main__":
    main()
