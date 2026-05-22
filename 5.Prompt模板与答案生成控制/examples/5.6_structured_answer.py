"""
程序运行流程图：

准备医疗 context 和 question
   ↓
构造强制结构化输出的 Prompt
   ↓
调用 Qwen 大模型
   ↓
打印结构化回答
   ↓
观察回答是否包含固定栏目

本脚本的教学目标：
学习如何通过 Prompt 让模型稳定输出固定结构，便于用户阅读和后续评估。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi


def main():
    """演示结构化答案 Prompt。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    context = "高血压患者应减少钠盐摄入。高血压患者如果出现胸闷、视物模糊，应及时就医。"
    question = "高血压患者要注意什么？"

    prompt = f"""
你是一个智能医疗科普助手。
请只根据资料回答，并严格按照以下格式输出：

【简要回答】

【依据资料】

【风险提醒】

【安全声明】

资料：
{context}

用户问题：
{question}

注意：
安全声明必须写明：本回答仅用于健康科普，不能替代医生诊断或治疗建议。
"""

    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    answer = llm.invoke(prompt)

    print(answer.content)

    # 观察重点：
    # 1. 结构化输出让答案更稳定。
    # 2. 固定栏目方便人工和程序检查。
    # 3. 医疗安全声明不容易遗漏。


if __name__ == "__main__":
    main()
