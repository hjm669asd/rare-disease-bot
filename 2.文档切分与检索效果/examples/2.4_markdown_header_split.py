"""
程序运行流程图：

┌────────────────────────────────────┐
│ 1. 读取 Markdown 医疗知识库         │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 2. 配置要识别的标题层级             │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 3. 创建 MarkdownHeaderTextSplitter  │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 4. 按 #、##、### 标题结构切分文本    │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 5. 每个 chunk 自动携带标题 metadata │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 6. 打印 metadata 和正文内容         │
└────────────────────────────────────┘

本脚本的教学目标：
理解“结构化文档不要只按长度切”。如果文档本身有 Markdown 标题，
就应该尽量利用标题层级，因为标题能告诉检索系统 chunk 属于哪个主题。
"""

# pathlib.Path 用于处理本地文件路径。
from pathlib import Path

# MarkdownHeaderTextSplitter 是 LangChain 的 Markdown 标题切分器。
# 它会根据 #、##、### 等标题层级切分文本，并把标题写入 metadata。
from langchain_text_splitters import MarkdownHeaderTextSplitter


# 当前脚本所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 清洗后的 Markdown 医疗知识库。
DATA_PATH = BASE_DIR / "data" / "clean_medical_text.md"


def main():
    """演示 Markdown 标题结构切分。"""

    # 读取 Markdown 文本。
    text = DATA_PATH.read_text(encoding="utf-8")

    # headers_to_split_on 用来告诉切分器识别哪些标题层级。
    # 每个元组的第一个元素是 Markdown 标题标记。
    # 每个元组的第二个元素是写入 metadata 时使用的字段名。
    headers_to_split_on = [
        # 识别一级标题，例如：# 智能医疗机器人知识库。
        ("#", "一级标题"),
        # 识别二级标题，例如：## 高血压日常管理。
        ("##", "二级标题"),
        # 识别三级标题，例如：### 危险信号。
        ("###", "三级标题"),
    ]

    # 创建 Markdown 标题切分器。
    # 参数 headers_to_split_on：标题识别规则。
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # split_text() 会返回 Document 列表。
    # 每个 Document 包含：
    # page_content：正文内容。
    # metadata：该正文所属的标题层级。
    documents = splitter.split_text(text)

    print(f"切分后 Document 数量: {len(documents)}")

    # 打印每个 Document 的标题元数据和正文。
    # 观察重点：metadata 是否能说明这个 chunk 属于哪个疾病、哪个小节。
    for index, document in enumerate(documents, start=1):
        print(f"\n--- Document {index} ---")
        print(f"标题元数据: {document.metadata}")
        print(document.page_content)


# Python 脚本入口。
if __name__ == "__main__":
    main()
