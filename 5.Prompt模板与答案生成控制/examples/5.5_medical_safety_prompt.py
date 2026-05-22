"""
程序运行流程图：

准备包含危险信号的医疗资料
   ↓
准备用户问题
   ↓
构造包含风险提示和安全声明要求的 Prompt
   ↓
调用 Qwen 大模型
   ↓
检查回答是否包含及时就医提醒和安全声明

本脚本的教学目标：
学习如何在医疗 RAG Prompt 中稳定加入风险提示和安全声明。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi


def main():
    """演示医疗安全声明与风险提示 Prompt。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    context = "高血压患者如果出现胸闷、视物模糊，应及时就医。"
    question = "高血压患者出现胸闷怎么办？"

    prompt = f"""
你是一个智能医疗科普助手。
请根据资料回答问题。
如果资料中出现危险信号，请明确提醒及时就医。
回答最后必须包含医疗安全声明：本回答仅用于健康科普，不能替代医生诊断或治疗建议。

资料：
{context}

用户问题：
{question}
"""

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    answer = llm.invoke(prompt)

    print(answer.content)

    # 观察重点：
    # 1. 胸闷属于需要重视的风险信号。
    # 2. Prompt 要求模型明确提醒及时就医。
    # 3. Prompt 要求回答必须包含医疗安全声明。


if __name__ == "__main__":
    main()
