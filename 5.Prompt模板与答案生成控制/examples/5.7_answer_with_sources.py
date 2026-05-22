"""
程序运行流程图：

准备带 metadata 的 Document
   ↓
从 Document 中提取 page_content 和 source
   ↓
把来源和内容一起拼接进 context
   ↓
构造要求“带来源回答”的 Prompt
   ↓
调用 Qwen 大模型
   ↓
打印带来源引用的回答

本脚本的教学目标：
学习如何把 Document.metadata 中的来源信息放进 Prompt，让回答更透明、可追溯。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document


DOCUMENTS = [
    Document(
        page_content="高血压患者应减少钠盐摄入，少吃腌制食品。",
        metadata={"source": "高血压饮食管理"},
    ),
    Document(
        page_content="高血压患者如果出现胸闷、视物模糊，应及时就医。",
        metadata={"source": "高血压危险信号"},
    ),
]


def build_context_with_sources(documents):
    """把每个 Document 的来源和内容拼接成上下文。"""

    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "未知来源")
        content = document.page_content
        context_parts.append(f"来源：{source}\n内容：{content}")

    return "\n\n".join(context_parts)


def main():
    """演示带来源引用的回答 Prompt。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    question = "高血压患者要注意什么？"
    context = build_context_with_sources(DOCUMENTS)

    prompt = f"""
你是一个智能医疗科普助手。
请根据资料回答问题，并在回答中标注依据来源。
不要使用资料之外的信息。
回答最后必须包含医疗安全声明。

资料：
{context}

用户问题：
{question}

请按照以下格式输出：
【回答】
【依据来源】
【安全声明】
"""

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    answer = llm.invoke(prompt)

    print(answer.content)

    # 观察重点：
    # 1. metadata 中的 source 进入了 Prompt。
    # 2. 模型可以在答案中引用来源。
    # 3. 来源引用有助于调试和安全评估。


if __name__ == "__main__":
    main()
