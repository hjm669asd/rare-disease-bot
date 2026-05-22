"""
程序运行流程图：

准备检索资料 context
   ↓
准备用户问题 question
   ↓
构造“只根据资料回答”的 Prompt
   ↓
调用 Qwen 大模型
   ↓
打印模型回答

本脚本的教学目标：
学习如何通过 Prompt 限制大模型只能根据检索资料回答，减少资料外发挥。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi


def main():
    """演示只根据资料回答的 Prompt。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    context = "高血压患者应减少钠盐摄入，少吃腌制食品。"
    question = "高血压患者饮食要注意什么？"

    # 关键约束：只根据资料回答，不使用资料之外的信息。
    prompt = f"""
你是一个智能医疗科普助手。
请只根据下面资料回答问题，不要使用资料之外的信息。
如果资料中没有答案，请说明无法根据现有资料可靠回答。

资料：
{context}

用户问题：
{question}

请用简洁中文回答。
"""

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    answer = llm.invoke(prompt)

    print(answer.content)

    # 观察重点：
    # 1. Prompt 明确禁止使用资料之外的信息。
    # 2. 这能降低模型胡编风险。
    # 3. 医疗场景中应优先保证答案可追溯。


if __name__ == "__main__":
    main()
