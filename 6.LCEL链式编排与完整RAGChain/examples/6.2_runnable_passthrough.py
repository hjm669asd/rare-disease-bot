"""
程序运行流程图：

导入 RunnablePassthrough
   ↓
创建 passthrough 对象
   ↓
传入一个用户问题
   ↓
原样输出这个用户问题
   ↓
理解它在 RAG Chain 中保留 question 的作用

本脚本的教学目标：
理解 RunnablePassthrough() 会原样传递输入，常用于把用户问题传给 Prompt 的 question 字段。
"""

from langchain_core.runnables import RunnablePassthrough


def main():
    """演示 RunnablePassthrough 的基础行为。"""

    # 创建一个原样传递输入的 Runnable。
    passthrough = RunnablePassthrough()

    question = "高血压患者饮食要注意什么？"

    # invoke() 输入什么，就输出什么。
    result = passthrough.invoke(question)

    print(result)

    # 观察重点：
    # 1. RunnablePassthrough 不做检索。
    # 2. RunnablePassthrough 不调用模型。
    # 3. 它只是把原始输入保留下来，供后续 Prompt 使用。


if __name__ == "__main__":
    main()
