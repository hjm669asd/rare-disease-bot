"""
程序运行流程图：

准备带 metadata 的文档列表
   ↓
设置目标疾病 disease=高血压
   ↓
根据 metadata 过滤文档
   ↓
打印过滤后的内容
   ↓
理解 metadata filter 如何减少无关疾病干扰

本脚本的教学目标：
理解元数据过滤在医疗 RAG 中的作用：先缩小检索范围，再进行问答。
"""


DOCUMENTS = [
    {
        "content": "高血压患者应减少钠盐摄入。",
        "metadata": {"disease": "高血压", "topic": "饮食"},
    },
    {
        "content": "糖尿病患者应监测血糖。",
        "metadata": {"disease": "糖尿病", "topic": "监测"},
    },
    {
        "content": "高血压患者出现胸闷应及时就医。",
        "metadata": {"disease": "高血压", "topic": "危险信号"},
    },
]


def filter_by_metadata(documents, key, value):
    """根据 metadata 中的 key=value 过滤文档。"""

    return [document for document in documents if document["metadata"].get(key) == value]


def main():
    """演示 metadata 过滤。"""

    filtered_documents = filter_by_metadata(DOCUMENTS, "disease", "高血压")

    print("过滤后的高血压文档：")
    for document in filtered_documents:
        print(document["metadata"])
        print(document["content"])

    # 观察重点：
    # 1. 糖尿病文档被过滤掉。
    # 2. 高血压相关资料被保留。
    # 3. 企业级医疗 RAG 应认真设计 metadata。


if __name__ == "__main__":
    main()
