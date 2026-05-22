"""
程序运行流程图：

准备与问题不匹配的 context
   ↓
准备用户问题
   ↓
构造“资料不足就拒答”的 Prompt
   ↓
调用 Qwen 大模型
   ↓
观察模型是否拒绝编造答案

本脚本的教学目标：
理解资料不足时，医疗 RAG 应该允许模型拒答，而不是强行生成没有依据的医疗建议。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi


def main():
    """演示资料不足时如何通过 Prompt 引导模型拒答。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 资料只有高血压饮食，并没有胃溃疡用药信息。
    context = "高血压患者应减少钠盐摄入。"

    # 用户问题超出了当前资料范围。
    question = "胃溃疡患者应该吃什么药？"

    prompt = f"""
你是一个智能医疗科普助手。
请只根据下面资料回答问题。
如果资料不足或资料与问题无关，请不要编造答案，直接说明无法根据现有资料可靠回答。
涉及用药问题时，如果资料没有明确依据，必须建议咨询医生。

资料：
{context}

用户问题：
{question}
"""

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    answer = llm.invoke(prompt)

    print(answer.content)

    # 观察重点：
    # 1. 问题是胃溃疡用药。
    # 2. 资料是高血压饮食。
    # 3. 正确行为是拒答，而不是编造胃溃疡用药建议。


if __name__ == "__main__":
    main()
