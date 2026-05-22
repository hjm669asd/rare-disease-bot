"""
程序运行流程图：

┌────────────────────────────────────┐
│ 1. 读取 .env 中的 DashScope API Key │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 2. 读取结构化医疗 Markdown 知识库   │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 3. 准备多组 chunk 参数实验          │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 4. 每组参数分别切分成 Document      │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 5. 使用 DashScopeEmbeddings 向量化  │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 6. 使用 FAISS 构建临时向量库        │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 7. 对同一个医疗问题做相似度检索     │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 8. 对比不同 chunk 参数的检索结果    │
└────────────────────────────────────┘

本脚本的教学目标：
通过同一个问题、同一份知识库、同一个 embedding 模型，只改变 chunk_size 和 chunk_overlap，
观察“切分参数”如何影响最终检索结果。
"""

# os 用于读取环境变量，例如 DASHSCOPE_API_KEY。
import os

# pathlib.Path 用于处理本地文件路径。
from pathlib import Path

# load_dotenv 用于从 .env 文件加载环境变量。
from dotenv import load_dotenv

# FAISS 是本地向量库组件。
# 它负责保存文本向量，并根据查询向量做相似度检索。
from langchain_community.vectorstores import FAISS

# Document 是 LangChain 的标准文档对象。
# 一个 Document 通常包含 page_content 和 metadata。
from langchain_core.documents import Document

# DashScopeEmbeddings 是通义千问/DashScope 的文本向量模型封装。
# 它负责把文本转换成向量。
from langchain_dashscope import DashScopeEmbeddings

# RecursiveCharacterTextSplitter 是递归字符切分器。
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 当前脚本所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 本实验使用的结构化医疗知识库。
DATA_PATH = BASE_DIR / "data" / "clean_medical_text.md"


# EXPERIMENTS 保存多组切分参数。
# 每组实验只改变 chunk_size 和 chunk_overlap，其它条件保持一致。
# 这样才能观察“切分参数”本身对检索效果的影响。
EXPERIMENTS = [
    # 小 chunk：内容更短，通常更聚焦，但可能缺少完整上下文。
    {"name": "小 chunk：更精准但上下文少", "chunk_size": 80, "chunk_overlap": 20},

    # 中等 chunk：通常是入门阶段比较平衡的选择。
    {"name": "中等 chunk：入门推荐平衡值", "chunk_size": 180, "chunk_overlap": 40},

    # 大 chunk：上下文更完整，但可能把多个主题混在一起，带来噪声。
    {"name": "大 chunk：上下文多但可能带噪声", "chunk_size": 300, "chunk_overlap": 60},
]


def split_text(text, chunk_size, chunk_overlap):
    """
    按照指定参数进行递归字符切分，并包装成 Document 列表。

    参数说明：
    text：原始医疗知识库文本。
    chunk_size：每个 chunk 的目标字符长度。
    chunk_overlap：相邻 chunk 的重叠字符数。

    返回值：
    Document 列表。每个 Document 都包含：
    page_content：chunk 正文。
    metadata：记录本 chunk 来自哪组切分参数。
    """

    # 创建递归字符切分器。
    # 这里每次实验都会创建一个新的 splitter，保证参数独立。
    splitter = RecursiveCharacterTextSplitter(
        # 当前实验使用的 chunk_size。
        chunk_size=chunk_size,

        # 当前实验使用的 chunk_overlap。
        chunk_overlap=chunk_overlap,

        # 中文医疗文本的切分优先级。
        # 优先按段落切，再按换行、句号、分号、逗号切，最后按字符切。
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    # split_text() 返回普通字符串列表。
    # 每个字符串是一个 chunk。
    chunks = splitter.split_text(text)

    # 把字符串 chunk 包装成 LangChain Document。
    # 为什么要包装？
    # 因为 FAISS.from_documents() 接收的是 Document 列表，
    # 并且 Document 可以携带 metadata，方便调试和追踪来源。
    return [
        Document(
            # page_content 是真正会被 embedding 和检索的文本内容。
            page_content=chunk,

            # metadata 是附加信息。
            # 这里记录切分参数，方便之后分析某个检索结果来自哪组实验。
            metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
        for chunk in chunks
    ]


def run_experiment(text, embeddings, question, experiment):
    """
    运行一次切分参数实验，并打印检索结果。

    参数说明：
    text：完整医疗知识库文本。
    embeddings：embedding 模型实例，用于把文本转换成向量。
    question：用户问题，所有实验都使用同一个问题。
    experiment：当前实验配置，包含 name、chunk_size、chunk_overlap。
    """

    # 使用当前实验参数切分文本。
    documents = split_text(
        text=text,
        chunk_size=experiment["chunk_size"],
        chunk_overlap=experiment["chunk_overlap"],
    )

    # FAISS.from_documents() 会完成两件事：
    # 1. 调用 embeddings，把每个 Document 的 page_content 转成向量。
    # 2. 把向量和 Document 存入 FAISS 索引，形成一个可检索的向量库。
    vectorstore = FAISS.from_documents(documents, embeddings)

    # similarity_search() 是相似度检索。
    # 它会先把 question 转成向量，再从 FAISS 中找最相似的 chunk。
    # 参数 k=2 表示返回最相似的 2 个结果。
    results = vectorstore.similarity_search(question, k=2)

    # 打印当前实验名称和参数。
    print(f"\n========== {experiment['name']} ==========")
    print(f"chunk_size: {experiment['chunk_size']}")
    print(f"chunk_overlap: {experiment['chunk_overlap']}")
    print(f"chunk 数量: {len(documents)}")
    print(f"检索问题: {question}")

    # 打印检索结果。
    # 观察重点：
    # 1. 是否检索到和问题真正相关的内容。
    # 2. 是否包含完整条件和建议。
    # 3. 是否混入无关疾病或无关主题。
    for index, result in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(f"metadata: {result.metadata}")
        print(result.page_content)


def main():
    """对比不同 chunk 参数对检索结果的影响。"""

    # 读取 .env 文件中的环境变量。
    # 本脚本需要 DASHSCOPE_API_KEY 来调用 DashScope embedding 模型。
    load_dotenv()

    # 如果没有配置 API Key，就提前报错。
    # 这样比等到 embedding 调用时报错更容易定位问题。
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 读取结构化医疗知识库文本。
    text = DATA_PATH.read_text(encoding="utf-8")

    # 创建 embedding 模型实例。
    # model="text-embedding-v2" 表示使用 DashScope 的文本向量模型。
    # 这个模型会把“知识库 chunk”和“用户问题”映射到同一个向量空间。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # 实验问题。
    # 所有切分参数都使用同一个问题，方便横向对比。
    question = "高血压患者什么时候需要及时就医？"

    # 依次执行每组实验。
    for experiment in EXPERIMENTS:
        run_experiment(text, embeddings, question, experiment)


# Python 脚本入口。
if __name__ == "__main__":
    main()
