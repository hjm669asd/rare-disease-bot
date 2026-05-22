"""
程序运行流程图：

读取 .env 中的 DASHSCOPE_API_KEY
   ↓
创建 PromptTemplate
   ↓
创建 ChatTongyi 大模型
   ↓
创建 StrOutputParser
   ↓
组合成 prompt | llm | parser
   ↓
运行链并打印返回值类型
   ↓
确认最终输出是字符串

本脚本的教学目标：
理解 StrOutputParser 的作用：把大模型消息对象转换成普通字符串。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def main():
    """演示 StrOutputParser 的作用。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    prompt = PromptTemplate.from_template("请用一句话回答：{question}")
    llm = ChatTongyi(model="qwen-turbo", temperature=0)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    answer = chain.invoke({"question": "RAG 是什么？"})

    print(type(answer))
    print(answer)

    # 观察重点：
    # 1. 没有 parser 时，llm 返回通常是消息对象。
    # 2. 加上 StrOutputParser 后，最终结果是 str。
    # 3. 字符串更适合直接展示、保存或继续处理。


if __name__ == "__main__":
    main()
