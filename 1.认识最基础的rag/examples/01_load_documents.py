"""
程序运行流程图：

┌──────────────────────────────┐
│ 1. 定位当前脚本所在目录       │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 2. 拼接医疗知识库文本路径     │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 3. 使用 TextLoader 加载文本   │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 4. 得到 LangChain Document    │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│ 5. 打印文档来源和部分内容     │
└──────────────────────────────┘
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader


# BASE_DIR 表示当前 Python 文件所在的目录。
# __file__ 是当前脚本路径。
# Path(__file__).resolve() 会得到当前脚本的绝对路径。
# .parent 会取当前脚本所在文件夹，也就是 examples/beginner。
BASE_DIR = Path(__file__).resolve().parent

# DATA_PATH 表示示例医疗知识库文件的完整路径。
# 这里使用 Path 拼接路径，可以兼容 Windows、macOS、Linux。
DATA_PATH = BASE_DIR / "data" / "sample.txt"


def main():
    """加载本地医疗知识文本，并打印 LangChain Document 的基础信息。"""

    # TextLoader 是 LangChain 的文档加载器组件。
    # 参数 str(DATA_PATH)：要读取的文本文件路径。
    # 参数 encoding="utf-8"：指定用 UTF-8 编码读取中文文本，避免乱码。
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")

    # load() 会把文本文件读取为 Document 列表。
    # Document 是 LangChain 的标准文档对象，通常包含：
    # page_content：文档正文。
    # metadata：文档元数据，例如来源路径。
    documents = loader.load()

    # len(documents) 表示加载出的 Document 数量。
    print(f"加载到 {len(documents)} 个 Document")

    # enumerate(documents, start=1) 用于遍历文档，并让编号从 1 开始。
    for index, document in enumerate(documents, start=1):
        print(f"\n--- Document {index} ---")

        # document.metadata.get("source") 读取文档来源路径。
        # get() 的好处是当 source 不存在时不会报错，而是返回 None。
        print(f"来源: {document.metadata.get('source')}")

        # document.page_content 是文档正文。
        # [:300] 表示只打印前 300 个字符，避免控制台输出太长。
        print(document.page_content[:300])


# 这是 Python 脚本的标准入口。
# 只有直接运行本文件时才会执行 main()。
# 如果这个文件被其他文件 import，main() 不会自动执行。
if __name__ == "__main__":
    main()
