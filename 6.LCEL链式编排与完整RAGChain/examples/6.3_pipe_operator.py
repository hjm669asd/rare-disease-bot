"""
程序运行流程图：

读取 .env 中的 DASHSCOPE_API_KEY
   ↓
创建 PromptTemplate
   ↓
创建 ChatTongyi
   ↓
创建 StrOutputParser
   ↓
用 | 串联 prompt、llm、parser
   ↓
传入 concept 变量并运行 chain
   ↓
打印最终回答

本脚本的教学目标：
理解 LCEL 管道符 | 的含义：左侧组件输出会传给右侧组件。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def main():
    """演示用管道符组合 LCEL 组件。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    prompt = PromptTemplate.from_template("请解释这个医学科普概念：{concept}")
    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    parser = StrOutputParser()

    # 管道符会按照从左到右的顺序执行。
    chain = prompt | llm | parser

    result = chain.invoke({"concept": "高血压日常监测"})
    print(result)

    # 观察重点：
    # 1. prompt 接收字典输入。
    # 2. prompt 输出模型消息输入。
    # 3. llm 输出消息对象。
    # 4. parser 输出字符串。


if __name__ == "__main__":
    main()
