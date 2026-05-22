"""
程序运行流程图：

构建向量库和 Retriever
   ↓
用户提出问题
   ↓
Retriever 返回相关 Documents
   ↓
提取每个 Document 的 page_content
   ↓
拼接成 context
   ↓
把 context 和 question 放进 prompt
   ↓
打印最终 prompt

本脚本的教学目标：
理解检索结果不是直接等于答案，而是要先进入 Prompt，作为大模型生成答案的依据。
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者如果出现明显头痛、胸闷、视物模糊，应及时就医。", metadata={"topic": "高血压危险信号"}),
    Document(page_content="高血压患者应定期监测血压，并记录血压变化。", metadata={"topic": "高血压监测"}),
    Document(page_content="糖尿病患者应控制总能量摄入。", metadata={"topic": "糖尿病饮食"}),
]


def build_prompt(question, documents):
    """把检索到的 Document 拼接成适合大模型阅读的 prompt。"""

    # page_content 是 Document 中真正的文本内容。
    # 多个检索结果之间用空行隔开，便于模型阅读。
    context = "\n\n".join(document.page_content for document in documents)

    # 在医疗场景中，prompt 需要明确限制模型只能根据资料回答。
    # 这样可以降低大模型脱离资料自由发挥的风险。
    prompt = f"""
你是一个智能医疗科普助手。
请只根据下面的资料回答问题。
如果资料不足，请说明无法根据现有资料可靠回答。

资料：
{context}

用户问题：
{question}

安全要求：
本回答只用于健康科普，不能替代医生诊断。
"""

    return prompt


def main():
    """演示如何把 Retriever 的结果放入 Prompt。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    question = "高血压患者什么时候需要及时就医？"
    documents = retriever.invoke(question)
    prompt = build_prompt(question, documents)

    print(prompt)

    # 观察重点：
    # 1. 检索结果进入了“资料”部分。
    # 2. 用户问题保留在 prompt 中。
    # 3. prompt 明确要求模型不要脱离资料回答。


if __name__ == "__main__":
    main()
