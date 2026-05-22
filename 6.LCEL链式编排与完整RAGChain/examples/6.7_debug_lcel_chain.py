"""
程序运行流程图：

准备医疗 Document
   ↓
构建 FAISS 和 Retriever
   ↓
输入用户问题
   ↓
手动调用 retriever 查看检索结果
   ↓
手动拼接 context
   ↓
手动渲染 prompt
   ↓
再运行完整 LCEL Chain
   ↓
对比中间结果和最终回答

本脚本的教学目标：
学习调试 RAG Chain 的方法：不要只看最终答案，还要检查检索结果、context 和 prompt。
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
    Document(page_content="高血压患者如果出现明显头痛、胸闷、视物模糊，应及时就医。", metadata={"source": "高血压危险信号"}),
    Document(page_content="普通感冒患者应注意休息，适量饮水。", metadata={"source": "感冒护理"}),
]


def format_docs(documents):
    """把 Document 列表转换成 context 字符串。"""

    return "\n\n".join(document.page_content for document in documents)


def main():
    """演示如何调试 LCEL RAG Chain。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = FAISS.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = PromptTemplate.from_template("""
你是一个智能医疗科普助手。
请只根据资料回答问题。

资料：
{context}

用户问题：
{question}

回答最后必须说明：本回答仅用于健康科普，不能替代医生诊断或治疗建议。
""")

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    question = "高血压患者什么时候需要及时就医？"

    # 第一步：单独检查 Retriever 检索到了什么。
    documents = retriever.invoke(question)
    print("========== 检索到的文档 ==========")
    for index, document in enumerate(documents, start=1):
        print(f"\n--- 文档 {index} ---")
        print(f"metadata: {document.metadata}")
        print(document.page_content)

    # 第二步：检查 context 是否包含关键内容。
    context = format_docs(documents)
    print("\n========== 拼接后的 context ==========")
    print(context)

    # 第三步：检查最终 prompt 是否符合预期。
    prompt_text = prompt.format(context=context, question=question)
    print("\n========== 最终 prompt ==========")
    print(prompt_text)

    # 第四步：运行完整 LCEL Chain。
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print("\n========== 最终回答 ==========")
    print(answer)

    # 观察重点：
    # 1. 如果答案不好，先检查检索结果。
    # 2. 再检查 context 是否包含关键医学信息。
    # 3. 最后检查 prompt 是否有足够清晰的约束。


if __name__ == "__main__":
    main()
