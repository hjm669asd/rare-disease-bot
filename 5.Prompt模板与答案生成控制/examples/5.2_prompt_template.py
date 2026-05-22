"""
程序运行流程图：

导入 PromptTemplate
   ↓
定义包含 {context} 和 {question} 的模板
   ↓
调用 format() 填充真实资料和问题
   ↓
打印生成后的 Prompt

本脚本的教学目标：
理解 PromptTemplate 的作用：把固定 Prompt 结构保存下来，只在运行时填入变量。
"""

from langchain_core.prompts import PromptTemplate


def main():
    """演示 LangChain PromptTemplate 基础用法。"""

    # PromptTemplate 中的 {context} 和 {question} 是占位符。
    prompt_template = PromptTemplate.from_template("""
你是一个智能医疗科普助手。
请根据下面资料回答用户问题。

资料：
{context}

用户问题：
{question}
""")

    # format() 会把变量填入模板。
    prompt = prompt_template.format(
        context="高血压患者应减少钠盐摄入。",
        question="高血压患者饮食要注意什么？",
    )

    print(prompt)

    # 观察重点：
    # 1. 模板提高了 Prompt 复用性。
    # 2. 修改模板可以统一影响所有回答风格。
    # 3. 后续构建复杂 RAG 链时，PromptTemplate 很常用。


if __name__ == "__main__":
    main()
