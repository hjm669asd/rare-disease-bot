"""
程序运行流程图：

准备一个用户问题
   ↓
生成多个不同表达的查询
   ↓
分别打印这些查询
   ↓
理解多个 query 可以覆盖更多表达方式

本脚本的教学目标：
理解 MultiQuery 的思想：用多个不同问法提高召回率。
"""


def generate_queries(question):
    """用简单规则生成多查询。"""

    return [
        question,
        "高血压患者如何控制盐分摄入？",
        "血压高的人应该避免哪些食物？",
        "高血压日常饮食管理建议有哪些？",
    ]


def main():
    """演示多查询生成。"""

    question = "高血压患者饮食要注意什么？"
    queries = generate_queries(question)

    print("生成的检索查询：")
    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")

    # 观察重点：
    # 1. 多个 query 表达相近但角度不同。
    # 2. 可以提高召回相关文档的概率。
    # 3. 但 query 过多也可能增加成本和噪声。


if __name__ == "__main__":
    main()
