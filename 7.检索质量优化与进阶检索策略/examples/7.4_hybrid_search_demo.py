"""
程序运行流程图：

准备医学文档列表
   ↓
准备包含精确术语的问题
   ↓
执行关键词检索
   ↓
模拟向量检索结果
   ↓
合并关键词检索和向量检索结果
   ↓
打印混合检索结果

本脚本的教学目标：
理解混合检索思想：结合关键词精确命中和向量语义召回。
"""


DOCUMENTS = [
    "糖尿病患者应关注糖化血红蛋白 HbA1c。",
    "血脂异常患者应关注 LDL-C 水平。",
    "高血压患者应减少钠盐摄入。",
    "普通感冒患者应注意休息。",
]


def keyword_search(keyword, documents):
    """非常简单的关键词检索。"""

    return [document for document in documents if keyword in document]


def merge_results(*result_groups):
    """合并多组检索结果，并去重。"""

    merged = []

    for group in result_groups:
        for item in group:
            if item not in merged:
                merged.append(item)

    return merged


def main():
    """演示混合检索。"""

    question = "HbA1c 指标需要关注什么？"

    # 关键词检索特别适合精确医学术语。
    keyword_results = keyword_search("HbA1c", DOCUMENTS)

    # 这里用手动列表模拟向量检索结果。
    vector_results = [
        "糖尿病患者应关注糖化血红蛋白 HbA1c。",
        "血脂异常患者应关注 LDL-C 水平。",
    ]

    hybrid_results = merge_results(keyword_results, vector_results)

    print(f"用户问题: {question}")
    print("\n混合检索结果：")
    for result in hybrid_results:
        print(result)

    # 观察重点：
    # 1. 关键词检索可以精准命中 HbA1c。
    # 2. 向量检索可以补充语义相关内容。
    # 3. 企业级 RAG 常常结合多种检索方式。


if __name__ == "__main__":
    main()
