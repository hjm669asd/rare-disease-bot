"""
程序运行流程图：

准备初步召回的候选文档
   ↓
定义简单打分规则
   ↓
根据疾病名、症状、及时就医等关键词加分
   ↓
按分数从高到低重新排序
   ↓
打印重排序后的结果

本脚本的教学目标：
用规则模拟 Rerank，理解“先召回，再精选排序”的思想。
"""


def simple_rerank(question, documents):
    """根据简单医学关键词规则对候选文档重新排序。"""

    def score(document):
        points = 0

        if "高血压" in document:
            points += 2

        if "胸闷" in document or "视物模糊" in document:
            points += 2

        if "及时就医" in document:
            points += 2

        if "感冒" in document:
            points -= 2

        return points

    return sorted(documents, key=score, reverse=True)


def main():
    """演示简单 Rerank。"""

    question = "高血压患者什么时候需要及时就医？"

    documents = [
        "高血压患者应减少钠盐摄入。",
        "感冒患者应注意休息。",
        "高血压患者出现胸闷、视物模糊，应及时就医。",
    ]

    reranked_documents = simple_rerank(question, documents)

    print(f"用户问题: {question}")
    print("\n重排序后结果：")
    for document in reranked_documents:
        print(document)

    # 观察重点：
    # 1. 最能回答问题的文档被排到第一位。
    # 2. 不相关的感冒内容被降低排序。
    # 3. 真实项目中可以使用专业 rerank 模型替代规则。


if __name__ == "__main__":
    main()
