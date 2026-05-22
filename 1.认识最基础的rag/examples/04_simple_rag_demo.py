"""
程序运行流程图：

┌────────────────────────────────┐
│ 1. 读取 .env 中的 DashScope Key │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 2. 加载智能医疗知识库文本       │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 3. 使用文本切分器生成 chunks    │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 4. 使用 DashScope 生成向量      │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 5. 使用 FAISS 构建向量库        │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 6. 把向量库转换为 Retriever     │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 7. Retriever 检索相关医疗资料   │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 8. 将资料和问题填入 Prompt      │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 9. 调用通义千问生成回答         │
└────────────────┬───────────────┘
                 ↓
┌────────────────────────────────┐
│ 10. 输出检索资料和最终回答      │
└────────────────────────────────┘
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_dashscope import ChatDashScope, DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 当前脚本所在目录：examples/beginner。
BASE_DIR = Path(__file__).resolve().parent

# 智能医疗机器人使用的本地知识库文本。
DATA_PATH = BASE_DIR / "data" / "sample.txt"


def format_documents(documents):
    """把检索到的多个 Document 拼接成一段可放入 Prompt 的上下文。"""

    # documents 是 Retriever 返回的 Document 列表。
    # document.page_content 是每个文档片段的正文。
    # "\n\n".join(...) 用两个换行分隔不同片段，让 Prompt 更容易阅读。
    return "\n\n".join(document.page_content for document in documents)


def build_retriever():
    """构建智能医疗 RAG Demo 的检索器 Retriever。"""

    # TextLoader：文档加载组件。
    # str(DATA_PATH)：医疗知识库文件路径。
    # encoding="utf-8"：使用 UTF-8 读取中文文本。
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")

    # load()：把本地 txt 文件转换成 LangChain Document 列表。
    documents = loader.load()

    # RecursiveCharacterTextSplitter：文本切分组件。
    # 医疗知识文档通常较长，不能整篇都直接向量化和塞入 Prompt。
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size：每个文本片段的目标字符长度。
        # 值越小，片段越细；值越大，保留上下文越多。
        chunk_size=180,
        # chunk_overlap：相邻片段之间的重叠字符数。
        # 医疗文本中一句话前后可能有关联，适当重叠可以避免语义被切断。
        chunk_overlap=40,
        # separators：递归切分时使用的分隔符优先级。
        # 先尽量按段落、换行、句号切分，最后才按字符切分。
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    # split_documents()：输入 Document 列表，输出切分后的 Document chunk 列表。
    chunks = splitter.split_documents(documents)

    # DashScopeEmbeddings：向量化组件。
    # model="text-embedding-v2"：指定 DashScope 的 embedding 模型。
    # 它会把医疗文本片段和用户问题转换到同一个向量空间。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # FAISS：本地向量库组件。
    # from_documents() 会把 chunks 向量化，并建立可相似度检索的 FAISS 索引。
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # as_retriever()：把向量库包装成 LangChain Retriever。
    # search_kwargs={"k": 3}：每次检索返回最相关的 3 个 chunk。
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def main():
    """运行完整的智能医疗机器人 RAG 问答流程。"""

    # load_dotenv()：读取 .env 文件中的环境变量。
    # 本项目通过 .env 提供 DASHSCOPE_API_KEY，避免把密钥写死在代码中。
    load_dotenv()

    # 检查 DashScope API Key 是否存在。
    # 如果没有配置，embedding 和大模型调用都会失败。
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 构建 Retriever。
    # Retriever 负责根据用户问题，从医疗知识库中找出最相关的资料片段。
    retriever = build_retriever()

    # ChatDashScope：通义千问聊天模型组件。
    # model="qwen-plus"：指定使用 qwen-plus 模型。
    # temperature=0：降低随机性，让教学 Demo 的输出更稳定、更可复现。
    llm = ChatDashScope(model="qwen-plus", temperature=0)

    # ChatPromptTemplate：Prompt 模板组件。
    # from_template() 会创建一个可填充变量的模板。
    # {context} 会被替换成检索到的医疗资料。
    # {question} 会被替换成用户问题。
    prompt = ChatPromptTemplate.from_template(
        """
    你是一个用于教学演示的智能医疗机器人。
    请只根据下面的资料回答用户问题。
    如果资料中没有足够信息，请说“当前资料不足，建议咨询医生或补充权威资料”。
    回答必须包含必要的医疗安全提醒，不能替代医生诊断。

    资料：
    {context}

    用户问题：
    {question}
    """.strip()
    )

    # 用户问题：模拟用户向智能医疗机器人咨询。
    question = "艾滋病患者日常应该注意什么？"

    # retriever.invoke(question)：执行检索。
    # 输入是用户问题，输出是最相关的 Document 列表。
    retrieved_documents = retriever.invoke(question)

    # format_documents()：把多个 Document 片段拼接为 Prompt 中的 context。
    context = format_documents(retrieved_documents)

    # LCEL 链式表达式：
    # prompt：把 context 和 question 填入模板。
    # llm：把组装好的消息发送给通义千问。
    # StrOutputParser：把模型输出解析为普通字符串。
    chain = prompt | llm | StrOutputParser()

    # chain.invoke()：运行完整链路。
    # 参数是字典，键名必须和 Prompt 模板中的变量名一致。
    answer = chain.invoke({"context": context, "question": question})

    print(f"用户问题: {question}")
    print("\n--- 检索到的资料 ---")
    print(context)
    print("\n--- 智能医疗机器人回答 ---")
    print(answer)


# 直接运行本文件时，执行 main()。
if __name__ == "__main__":
    main()
