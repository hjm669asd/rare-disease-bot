"""
程序运行流程图：

┌────────────────────────────────────┐
│ 1. 读取结构化医疗 Markdown 文本     │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 2. 设置 chunk_size 和 chunk_overlap │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 3. 从 start=0 开始截取固定长度文本  │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 4. 每次移动 chunk_size-overlap 距离 │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 5. 得到多个固定长度 chunk           │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 6. 打印 chunk 数量和内容            │
└────────────────────────────────────┘

本脚本的教学目标：
理解最朴素的切分方式：不考虑标题、段落、句子和语义，只按照固定字符长度切。
它简单，但很容易切断一句完整的医疗建议。
"""

# pathlib.Path 用于处理本地文件路径。
from pathlib import Path


# 当前脚本所在目录：2.文档切分与检索效果/examples。
BASE_DIR = Path(__file__).resolve().parent

# 要切分的结构化医疗知识库文件。
# 这里使用清洗后的 Markdown 文件，而不是混乱文本。
DATA_PATH = BASE_DIR / "data" / "clean_medical_text.md"


def fixed_length_split(text, chunk_size, chunk_overlap):
    """
    使用固定字符长度切分文本。

    参数说明：
    text：要被切分的完整文本。
    chunk_size：每个 chunk 的目标字符长度。
    chunk_overlap：相邻 chunk 之间重叠的字符数量。

    返回值：
    chunks：切分后的字符串列表。

    注意：
    这是教学版固定长度切分函数，目的是帮助你看懂切分机制。
    真实项目中通常优先使用 LangChain 内置 splitter。
    """

    # chunks 用于保存切分出来的所有文本片段。
    chunks = []

    # start 表示当前 chunk 的起始字符下标。
    # Python 字符串下标从 0 开始。
    start = 0

    # 只要 start 没有超过文本总长度，就继续切分。
    while start < len(text):
        # end 表示当前 chunk 的结束位置。
        # Python 切片 text[start:end] 包含 start，不包含 end。
        end = start + chunk_size

        # 按固定长度截取一个 chunk。
        # 如果 end 超过文本长度，Python 会自动截到文本末尾，不会报错。
        chunk = text[start:end]

        # 把当前 chunk 加入结果列表。
        chunks.append(chunk)

        # 下一轮起点向前移动。
        # 为什么不是 start = end？
        # 因为需要 overlap，让相邻 chunk 有一段重叠上下文。
        start = end - chunk_overlap

        # 防御性处理：如果 chunk_overlap 设置异常导致 start 小于 0，就重置为 0。
        if start < 0:
            start = 0

        # 注意：如果 chunk_overlap >= chunk_size，这里会导致死循环。
        # 所以真实项目中应保证 chunk_overlap < chunk_size。

    return chunks


def main():
    """演示固定长度字符切分的效果。"""

    # 读取完整 Markdown 医疗知识库文本。
    text = DATA_PATH.read_text(encoding="utf-8")

    # chunk_size 表示每个 chunk 的目标字符长度。
    # 你可以把它改成 80、180、300，观察切分效果。
    chunk_size = 120

    # chunk_overlap 表示相邻 chunk 重叠多少字符。
    # 重叠可以减少重要句子刚好被切断的问题。
    chunk_overlap = 30

    # 调用自定义固定长度切分函数。
    chunks = fixed_length_split(text, chunk_size, chunk_overlap)

    # 打印基础统计信息，方便比较不同参数下的结果。
    print(f"原始文本字符数: {len(text)}")
    print(f"chunk_size: {chunk_size}")
    print(f"chunk_overlap: {chunk_overlap}")
    print(f"切分后 chunk 数量: {len(chunks)}")

    # 只打印前 8 个 chunk，避免控制台输出过长。
    # 观察重点：是否切断标题、句子、医疗风险提示。
    for index, chunk in enumerate(chunks[:8], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)


# Python 脚本入口。
if __name__ == "__main__":
    main()
