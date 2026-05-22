"""
程序运行流程图：

准备带 source metadata 的医疗 Document
   ↓
构建 FAISS 向量库
   ↓
转换成 Retriever
   ↓
定义 format_docs_with_sources
   ↓
把来源和内容一起放入 context
   ↓
用 LCEL 构建 RAG Chain
   ↓
输出带来源引用的回答

本脚本的教学目标：
学习如何在 LCEL RAG Chain 中把 Document.metadata 作为来源信息传给大模型。
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
    Document(page_content="高血压患者如果出现胸闷、视物模糊，应及时就医。", metadata={"source": "高血压危险信号"}),
    Document(page_content="糖尿病患者应控制总能量摄入。", metadata={"source": "糖尿病饮食管理"}),
]


def format_docs_with_sources(documents):
    """把 Document 的来源和内容一起拼接成 context。"""

    return "\n\n".join(
        f"来源：{document.metadata.get('source', '未知来源')}\n内容：{document.page_content}"
        for document in documents
    )


def main():
    """运行带来源信息的 RAG Chain。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = PromptTemplate.from_template("""
你是一个智能医疗科普助手。
请根据资料回答问题，并标注依据来源。
不要使用资料之外的信息。
回答最后必须包含安全声明。

资料：
{context}

用户问题：
{question}

请按照以下格式输出：
【回答】
【依据来源】
【安全声明】
""")

    llm = ChatTongyi(model="qwen-turbo", temperature=0)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke("高血压患者要注意什么？")
    print(answer)

    # 观察重点：
    # 1. context 中不仅有内容，还有来源。
    # 2. 模型可以根据来源生成引用。
    # 3. 医疗 RAG 中来源引用有助于安全审查。


if __name__ == "__main__":
    main()
