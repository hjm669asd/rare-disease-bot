"""
程序运行流程图：

准备医疗知识 Documents
   ↓
构建 FAISS 向量库
   ↓
输入医疗相关问题
   ↓
检索 Document 和 score
   ↓
用简单规则判断结果是否适合作为回答依据
   ↓
打印可靠性判断

本脚本的教学目标：
理解医疗 RAG 不能盲目信任所有检索结果。即使向量库返回了内容，也要判断它是否与问题足够相关、是否包含关键医学条件。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者如出现胸闷、视物模糊或血压持续明显升高，应及时就医。", metadata={"topic": "高血压危险信号"}),
    Document(page_content="高血压患者应减少钠盐摄入，保持规律运动。", metadata={"topic": "高血压生活方式"}),
    Document(page_content="感冒患者应注意休息，多饮水。", metadata={"topic": "感冒护理"}),
]


def is_reliable_medical_result(document, score):
    """
    使用简单规则判断检索结果是否适合作为回答依据。

    注意：
    这只是教学示例，不是生产级医学安全策略。
    真实系统需要更严格的评测、重排、人工审核和合规流程。
    """

    content = document.page_content

    # 规则 1：内容需要和高血压主题有关。
    has_correct_topic = "高血压" in content

    # 规则 2：如果问题问及时就医，最好包含就医或危险信号相关信息。
    has_safety_signal = "就医" in content or "胸闷" in content or "视物模糊" in content

    # 规则 3：score 不能太差。
    # 注意：这个阈值只是教学演示，不同模型和向量库需要重新实验。
    has_acceptable_score = score < 1.2

    return has_correct_topic and has_safety_signal and has_acceptable_score


def main():
    """演示如何对检索结果做简单可靠性判断。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)

    question = "高血压患者什么时候需要及时就医？"
    results = vectorstore.similarity_search_with_score(question, k=3)

    print(f"用户问题: {question}")

    for index, (document, score) in enumerate(results, start=1):
        reliable = is_reliable_medical_result(document, score)

        print(f"\n--- 检索结果 {index} ---")
        print(f"score: {score}")
        print(f"metadata: {document.metadata}")
        print(f"是否适合作为回答依据: {reliable}")
        print(document.page_content)

    print("\n安全提醒：本示例仅用于技术教学，不构成真实医疗建议。")

    # 观察重点：
    # 1. 检索到内容不等于内容可靠。
    # 2. 医疗场景需要更谨慎地筛选上下文。
    # 3. 后续进阶模块可以使用重排模型、规则过滤和人工评测提升可靠性。


if __name__ == "__main__":
    main()
