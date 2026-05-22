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
使用管道符 | 组合成 LCEL Chain
   ↓
调用 chain.invoke()
   ↓
打印字符串结果

本脚本的教学目标：
理解 LCEL 的最小用法：PromptTemplate | LLM | StrOutputParser。
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def main():
    """演示最基础的 LCEL 链。"""

    load_dotenv()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # PromptTemplate 负责把输入变量渲染成 prompt。
    prompt = PromptTemplate.from_template("请用一句话解释：{topic}")

    # ChatTongyi 是通义千问聊天模型。
    # temperature=0 表示尽量稳定输出，减少随机性。
    llm = ChatTongyi(model="qwen-turbo", temperature=0)

    # StrOutputParser 会把模型返回的消息对象转换成字符串。
    parser = StrOutputParser()

    # LCEL 核心写法：用 | 把组件串成一条链。
    chain = prompt | llm | parser

    # invoke() 用于运行链。
    result = chain.invoke({"topic": "RAG"})

    print(result)

    # 观察重点：
    # 1. prompt 的输出会传给 llm。
    # 2. llm 的输出会传给 parser。
    # 3. 最终结果是一个普通字符串。


if __name__ == "__main__":
    main()
