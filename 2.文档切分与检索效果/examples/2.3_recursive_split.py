"""
程序运行流程图：

┌────────────────────────────────────┐
│ 1. 读取结构化医疗 Markdown 文本     │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 2. 创建 RecursiveCharacterTextSplitter │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 3. 配置 chunk_size、overlap、分隔符 │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 4. 优先按段落、换行、句号递归切分   │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 5. 得到更自然的文本 chunks          │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 6. 打印 chunk 数量和每个 chunk 内容 │
└────────────────────────────────────┘

本脚本的教学目标：
理解 LangChain 中最常用的入门切分器 RecursiveCharacterTextSplitter。
它比固定长度切分更聪明，因为它会优先保留段落、句子等自然边界。
"""

# pathlib.Path 用于处理文件路径。
from pathlib import Path

# RecursiveCharacterTextSplitter 是 LangChain 提供的递归字符切分器。
# 它会按照 separators 列表中的分隔符优先级逐级尝试切分。
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 当前脚本所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 要切分的结构化医疗 Markdown 文本。
DATA_PATH = BASE_DIR / "data" / "clean_medical_text.md"


def main():
    """演示递归字符切分的效果。"""

    # 读取完整文本。
    # 当前文本已经经过 2.1 的预处理，具有 Markdown 标题和清晰段落。
    text = DATA_PATH.read_text(encoding="utf-8")

    # 创建递归字符切分器。
    # 它不是简单硬切，而是尽量从更自然的边界切开文本。
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size：每个 chunk 的目标字符长度。
        # 如果设置太小，医学建议可能被拆散。
        # 如果设置太大，检索结果可能包含太多无关内容。
        chunk_size=300,

        # chunk_overlap：相邻 chunk 的重叠字符数。
        # 作用是减少“关键句子刚好被切断”的问题。
        chunk_overlap=100,

        # separators：递归切分的分隔符优先级。
        # "\n\n"：优先按空行分隔的段落切。
        # "\n"：其次按普通换行切。
        # "。"：再按中文句号切。
        # "；"：再按中文分号切。
        # "，"：再按中文逗号切。
        # " "：再按空格切。
        # ""：最后按单字符强制切，保证文本一定能被切开。
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    # split_text() 输入普通字符串，输出字符串 chunk 列表。
    # 如果输入的是 Document 列表，可以使用 split_documents()。
    chunks = splitter.split_text(text)

    # 打印统计信息。
    print(f"原始文本字符数: {len(text)}")
    print(f"切分后 chunk 数量: {len(chunks)}")

    # 打印每个 chunk。
    # 观察重点：标题是否保留、句子是否完整、不同疾病主题是否混在一起。
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)


# Python 脚本入口。
if __name__ == "__main__":
    main()
