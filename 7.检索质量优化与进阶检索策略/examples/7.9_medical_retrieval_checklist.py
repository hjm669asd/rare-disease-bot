"""
程序运行流程图：

准备医疗检索结果
   ↓
定义安全检查函数
   ↓
检查是否命中疾病
   ↓
检查是否包含就医建议
   ↓
检查是否混入无关疾病
   ↓
打印检查结果

本脚本的教学目标：
理解医疗 RAG 不仅要看检索是否相关，还要用安全清单检查结果是否可靠、完整、无明显噪声。
"""


def check_medical_retrieval(question, results):
    """对医疗检索结果做简单安全检查。"""

    checks = {
        "命中高血压": any("高血压" in result for result in results),
        "包含就医建议": any("及时就医" in result for result in results),
        "混入感冒内容": any("感冒" in result for result in results),
        "包含危险信号": any("胸闷" in result or "视物模糊" in result for result in results),
    }

    return checks


def main():
    """演示医疗 RAG 检索安全清单。"""

    question = "高血压什么时候需要及时就医？"

    results = [
        "高血压患者出现胸闷、视物模糊，应及时就医。",
        "高血压患者应定期监测血压。",
    ]

    checks = check_medical_retrieval(question, results)

    print(f"用户问题: {question}")
    print("\n检索结果：")
    for result in results:
        print(result)

    print("\n安全检查：")
    for key, value in checks.items():
        print(f"{key}: {value}")

    # 观察重点：
    # 1. 是否命中正确疾病。
    # 2. 是否包含风险症状和及时就医建议。
    # 3. 是否混入无关疾病内容。
    # 4. 这类检查可以作为企业级 RAG 的质量门槛。


if __name__ == "__main__":
    main()
