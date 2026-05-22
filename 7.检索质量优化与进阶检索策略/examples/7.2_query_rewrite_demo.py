"""
程序运行流程图：

准备历史对话
   ↓
准备当前用户追问
   ↓
判断追问是否缺少上下文
   ↓
把追问改写成完整检索问题
   ↓
打印改写前和改写后的 query

本脚本的教学目标：
理解查询改写如何解决多轮对话中的省略、指代和问题不完整问题。
"""


def rewrite_query(chat_history, question):
    """用简单规则模拟查询改写。"""

    # 如果历史中提到高血压，当前问题又提到运动，说明用户大概率在问高血压运动。
    if "高血压" in chat_history and "运动" in question:
        return "高血压患者是否可以运动？运动时需要注意什么？"

    # 如果没有命中规则，就返回原始问题。
    return question


def main():
    """演示查询改写。"""

    chat_history = "用户之前询问了高血压患者饮食注意事项。"
    question = "那我还能运动吗？"

    rewritten_question = rewrite_query(chat_history, question)

    print(f"原始问题: {question}")
    print(f"改写后问题: {rewritten_question}")

    # 观察重点：
    # 1. 原始问题中“那”依赖历史上下文。
    # 2. 改写后问题更完整，更适合 Retriever 检索。
    # 3. 企业级项目中可以用 LLM 自动完成查询改写。


if __name__ == "__main__":
    main()
