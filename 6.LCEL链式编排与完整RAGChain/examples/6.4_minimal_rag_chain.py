"""
程序运行流程图：

准备医疗 Document
   ↓
创建 DashScopeEmbeddings
   ↓
构建 FAISS 向量库
   ↓
转换成 Retriever
   ↓
定义 format_docs 函数
   ↓
创建 PromptTemplate、LLM、StrOutputParser
   ↓
使用 LCEL 组装最小 RAG Chain
   ↓
输入用户问题并得到回答

本脚本的教学目标：
把 Retriever、Prompt、LLM、Parser 组合成一个最小可运行的 RAG Chain。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_dashscope import DashScopeEmbeddings


DOCUMENTS = [
    Document(page_content="高血压患者应减少钠盐摄入，少吃腌制食品。", metadata={"source": "高血压饮食管理"}),
    Document(page_content="高血压患者如果出现明显头痛、胸闷、视物模糊、肢体无力，或血压持续明显升高，应及时就医。", metadata={"source": "高血压危险信号"}),
    Document(page_content="糖尿病患者应控制总能量摄入。", metadata={"source": "糖尿病饮食管理"}),
]


def format_docs(documents):
    """把 Retriever 返回的 Document 列表拼接成字符串 context。"""

    return "\n\n".join(document.page_content for document in documents)


def main():
    """运行最小 RAG Chain。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = PromptTemplate.from_template("""
你是一个智能医疗科普助手。
请只根据下面资料回答问题。
如果资料不足，请说明无法根据现有资料可靠回答。

资料：
{context}

用户问题：
{question}

回答最后必须说明：本回答仅用于健康科普，不能替代医生诊断或治疗建议。
""")

    llm = ChatTongyi(model="qwen-turbo", temperature=0)

    # 这里是最小 RAG Chain 的核心：
    # context：用户问题先传给 retriever，检索出 Document，再用 format_docs 拼成字符串。
    # question：用户问题通过 RunnablePassthrough 原样传给 Prompt。
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "高血压患者什么时候需要及时就医？"
    answer = rag_chain.invoke(question)

    print(answer)

    # 观察重点：
    # 1. invoke() 输入的是一个字符串问题。
    # 2. LCEL 内部自动完成检索、拼接、填充 prompt、调用模型和解析输出。
    # 3. 这就是完整 RAG Chain 的基础形态。


if __name__ == "__main__":
    main()
