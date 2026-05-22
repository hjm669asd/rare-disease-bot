"""
程序运行流程图：

┌──────────────────────────────┐
│ 1. 读取 .env 中的 API Key     │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 2. 加载医疗知识库文本         │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 3. 切分为多个 Document chunk  │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 4. DashScope 生成文本向量     │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 5. FAISS 构建本地向量库       │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 6. 保存向量库并测试相似检索   │
└──────────────────────────────┘
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 当前脚本所在目录：examples/beginner。
BASE_DIR = Path(__file__).resolve().parent

# 医疗知识库文本路径。
DATA_PATH = BASE_DIR / "data" / "sample.txt"

# FAISS 本地向量库保存目录。
# 运行本脚本后，会在 examples/beginner/faiss_index 下生成索引文件。
VECTORSTORE_PATH = BASE_DIR / "faiss_index"


def load_and_split_documents():
    """加载医疗知识文本，并切分成适合 embedding 和检索的小片段。"""

    # TextLoader 是文档加载组件。
    # str(DATA_PATH)：目标文本路径。
    # encoding="utf-8"：保证中文内容正确读取。
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")

    # load() 返回 Document 列表。
    documents = loader.load()

    # RecursiveCharacterTextSplitter 是文本切分组件。
    splitter = RecursiveCharacterTextSplitter(
        # 每个 chunk 的目标长度。
        chunk_size=180,
        # 相邻 chunk 的重叠长度，用于保留上下文连续性。
        chunk_overlap=40,
        # 切分优先级，从段落到字符逐级降级。
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    # 返回切分后的 Document chunk 列表。
    return splitter.split_documents(documents)


def main():
    """构建 FAISS 向量库，并用一个医疗问题测试相似度检索效果。"""

    # load_dotenv() 会读取项目根目录或当前运行目录附近的 .env 文件。
    # 本项目用它加载 DASHSCOPE_API_KEY。
    load_dotenv()
    
    # os.getenv("DASHSCOPE_API_KEY") 从环境变量中读取 DashScope API Key。
    # 如果没有配置，后续调用 embedding 模型会失败，所以这里提前报错。
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 加载并切分医疗知识文本。
    chunks = load_and_split_documents()
    #展示所有chunk

    # 创建嵌入模型实例
    # DashScopeEmbeddings 是 embedding 组件。
    # 参数 model="text-embedding-v2" 表示使用 DashScope 的文本向量模型。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # FAISS.from_documents() 会做两件事：
    # 1. 调用 embeddings 把每个 chunk 转成向量。
    # 2. 把向量和原始文本一起写入 FAISS 向量索引。
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # save_local() 把 FAISS 向量库保存到本地目录，方便后续复用。
    vectorstore.save_local(str(VECTORSTORE_PATH))

    print(f"已构建 FAISS 向量库，chunk 数量: {len(chunks)}")
    print(f"保存路径: {VECTORSTORE_PATH}")

    # 测试问题1：模拟用户向智能医疗机器人提问。
    question1 = "高血压患者日常应该注意什么？"

    # 测试问题2：
    question2 = "新冠怎么治？"

    # similarity_search() 执行向量相似度检索。
    # 参数 question：用户问题，会先被 embedding 成查询向量。
    # 参数 k=2：返回最相似的 2 个文档片段。
    results = vectorstore.similarity_search(question2, k=2)

    print(f"\n测试检索问题: {question2}")
    for index, document in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(document.page_content)


# 直接运行本文件时，执行 main()。
if __name__ == "__main__":
    main()
