"""
程序运行流程图：

准备医疗 Document
   ↓
构建 FAISS 向量库
   ↓
分别创建 k=1、k=2、k=3 的 Retriever
   ↓
使用同一个问题检索
   ↓
对比返回结果数量和内容

本脚本的教学目标：
理解 as_retriever(search_kwargs={"k": ...}) 中 k 参数对检索结果数量和噪声的影响。
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
]


def main():
    """对比不同 k 值的 Retriever 检索结果。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者什么时候需要及时就医？"

    for k in [1, 2, 3]:
        # 每次用不同 k 创建一个新的 retriever。
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(question)

        print(f"\n========== k={k} ==========")
        print(f"返回结果数量: {len(results)}")

        for index, document in enumerate(results, start=1):
            print(f"\n--- 结果 {index} ---")
            print(f"metadata: {document.metadata}")
            print(document.page_content)

    # 观察重点：
    # 1. k 越大，返回结果越多。
    # 2. 返回结果越多，不代表质量越高。
    # 3. 医疗场景中需要避免无关疾病内容进入上下文。


if __name__ == "__main__":
    main()
