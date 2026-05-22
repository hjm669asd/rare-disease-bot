"""
程序运行流程图：

┌──────────────────────────────┐
│ 1. 定位医疗知识库文本路径     │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 2. 使用 TextLoader 加载文档   │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 3. 创建文本切分器             │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 4. 把 Document 切分成 chunks  │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 5. 打印切分数量和部分片段     │
└──────────────────────────────┘
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 当前脚本所在目录：examples/beginner。
BASE_DIR = Path(__file__).resolve().parent

# 示例医疗知识库文本路径：examples/beginner/data/sample.txt。
DATA_PATH = BASE_DIR / "data" / "sample.txt"


def main():
    """加载医疗知识文本，并演示如何把长文档切分成适合检索的小片段。"""

    # TextLoader 负责把本地 txt 文件加载成 LangChain Document。
    # str(DATA_PATH)：传入文本文件路径。
    # encoding="utf-8"：正确读取中文内容。
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")

    # documents 是 Document 列表。
    # 当前示例只有一个 txt 文件，所以通常只会得到 1 个 Document。
    documents = loader.load()

    # RecursiveCharacterTextSplitter 是递归字符切分器。
    # 它会按照 separators 指定的分隔符优先级，从大到小尝试切分文本。
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size 表示每个文本片段的目标长度。
        # 初学阶段设置小一点，方便在控制台观察切分效果。
        chunk_size=180,
        # chunk_overlap 表示相邻片段之间保留多少重叠字符。
        # 重叠可以减少切分边界造成的语义丢失。
        chunk_overlap=40,
        # separators 表示切分优先级。
        # "\n\n"：优先按段落切分。
        # "\n"：其次按换行切分。
        # "。"：再按中文句号切分。
        # "，"：再按中文逗号切分。
        # " "：再按空格切分。
        # ""：最后按单个字符强制切分，确保不会超过目标长度太多。
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    # split_documents() 会保留 Document 的 metadata，并把长文档切成多个短 Document。
    # 每一个短 Document 就是后续要进入向量库的 chunk。
    chunks = splitter.split_documents(documents)

    print(f"原始 Document 数量: {len(documents)}")
    print(f"切分后 chunk 数量: {len(chunks)}")

    # chunks[:5] 表示只展示前 5 个 chunk，方便观察。
    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.page_content)


# 直接运行本文件时，执行 main()。
if __name__ == "__main__":
    main()
