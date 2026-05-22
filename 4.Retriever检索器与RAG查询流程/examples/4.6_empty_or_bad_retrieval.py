"""
程序运行流程图：

构建医疗知识库和 Retriever
   ↓
提出一个知识库中不一定能回答的问题
   ↓
Retriever 返回若干结果
   ↓
用简单规则判断结果是否匹配问题主题
   ↓
如果不匹配，就提示无法可靠回答
   ↓
如果匹配，再拼接上下文

本脚本的教学目标：
理解 RAG 系统需要处理“检索不到”和“检索到了但不相关”两类失败情况。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入，保持规律运动。", metadata={"topic": "高血压"}),
    Document(page_content="糖尿病患者应监测空腹血糖和餐后血糖。", metadata={"topic": "糖尿病"}),
    Document(page_content="普通感冒患者应注意休息，适量饮水。", metadata={"topic": "感冒"}),
]


def is_relevant_to_question(question, document):
    """
    用非常简单的规则判断检索结果是否和问题相关。

    注意：
    这只是教学示例，不是生产级相关性判断。
    真实系统可以使用相似度阈值、重排模型、分类器或人工评测。
    """

    content = document.page_content

    if "高血压" in question and "高血压" in content:
        return True

    if "糖尿病" in question and "糖尿病" in content:
        return True

    if "感冒" in question and "感冒" in content:
        return True

    return False


def main():
    """演示如何处理检索失败或检索不相关。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 这个问题和知识库中的内容不匹配。
    question = "胃溃疡患者应该如何用药？"

    documents = retriever.invoke(question)

    print(f"用户问题: {question}")

    if not documents:
        print("没有检索到相关资料，无法可靠回答。")
        return

    relevant_documents = [
        document for document in documents if is_relevant_to_question(question, document)
    ]

    print("\n========== 原始检索结果 ==========")
    for index, document in enumerate(documents, start=1):
        print(f"\n--- 结果 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    if not relevant_documents:
        print("\n检索到了资料，但资料主题与问题不匹配，建议不要直接生成医疗回答。")
        print("更安全的回复：根据现有资料无法可靠回答该问题，请咨询医生或补充更多信息。")
        return

    context = "\n\n".join(document.page_content for document in relevant_documents)
    print("\n========== 可用上下文 ==========")
    print(context)

    # 观察重点：
    # 1. 向量检索通常会返回最相似的结果，但不代表一定相关。
    # 2. 对医疗场景，宁可拒答，也不要基于错误资料回答。


if __name__ == "__main__":
    main()
