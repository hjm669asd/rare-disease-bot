"""
程序运行流程图：

准备 context 和 question
   ↓
使用 f-string 构造最基础 Prompt
   ↓
打印 Prompt
   ↓
观察资料和用户问题如何被组织成模型输入

本脚本的教学目标：
理解 Prompt 在 RAG 中的作用：把检索资料和用户问题组织成大模型可以理解的输入。
"""


def main():
    """演示最基础的 RAG Prompt。"""

    # context 表示 Retriever 检索到的资料。
    context = "高血压患者应减少钠盐摄入，少吃腌制食品。"

    # question 表示用户提出的问题。
    question = "高血压患者饮食要注意什么？"

    # 这是最基础的 Prompt。
    # 它告诉模型角色、资料和用户问题。
    prompt = f"""
你是一个智能医疗科普助手。
请根据下面资料回答问题。

资料：
{context}

用户问题：
{question}
"""

    print(prompt)

    # 观察重点：
    # 1. Prompt 不是答案，而是模型输入。
    # 2. context 来自检索结果。
    # 3. question 来自用户问题。


if __name__ == "__main__":
    main()
