"""
程序运行流程图：

准备一组评估问题
   ↓
为每个问题设置期望关键词
   ↓
准备模拟检索结果
   ↓
判断检索结果是否包含期望关键词
   ↓
统计命中情况

本脚本的教学目标：
理解检索评估的基本思想：用问题集和期望答案检查检索效果，而不是只凭感觉调参。
"""


EVAL_CASES = [
    {
        "question": "高血压患者什么时候需要及时就医？",
        "expected_keyword": "及时就医",
    },
    {
        "question": "糖尿病患者要关注什么指标？",
        "expected_keyword": "HbA1c",
    },
    {
        "question": "高血压患者饮食要注意什么？",
        "expected_keyword": "钠盐",
    },
]

RETRIEVED_RESULTS = {
    "高血压患者什么时候需要及时就医？": "高血压患者出现胸闷、视物模糊，应及时就医。",
    "糖尿病患者要关注什么指标？": "糖尿病患者应关注糖化血红蛋白 HbA1c。",
    "高血压患者饮食要注意什么？": "高血压患者应减少钠盐摄入。",
}


def main():
    """演示简单检索评估。"""

    hit_count = 0

    for case in EVAL_CASES:
        question = case["question"]
        expected_keyword = case["expected_keyword"]
        result = RETRIEVED_RESULTS[question]
        hit = expected_keyword in result

        if hit:
            hit_count += 1

        print(f"问题: {question}")
        print(f"检索结果: {result}")
        print(f"是否命中期望关键词: {hit}")
        print()

    print(f"命中数量: {hit_count}/{len(EVAL_CASES)}")

    # 观察重点：
    # 1. 评估集可以帮助判断检索修改是否有效。
    # 2. expected_keyword 是最简单的评估标准。
    # 3. 真实项目可以扩展为人工标注、MRR、Recall@K 等指标。


if __name__ == "__main__":
    main()
