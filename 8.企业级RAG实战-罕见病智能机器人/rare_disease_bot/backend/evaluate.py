"""RAG系统评测脚本 - 自动生成测试集、运行pipeline、输出评估报告"""

import json
import time
from pathlib import Path

import pandas as pd

from config import DATASET_PATH


# ==================== 问题模板 ====================

SYMPTOM_TEMPLATES = [
    "{disease}有哪些常见症状？",
    "{disease}的主要表现是什么？",
    "{disease}的临床症状有哪些？",
    "得了{disease}会出现什么症状？",
]

CAUSE_TEMPLATES = [
    "{disease}的病因是什么？",
    "{disease}是怎么引起的？",
    "{disease}的发病原因有哪些？",
    "为什么会得{disease}？",
]

TREATMENT_TEMPLATES = [
    "{disease}怎么治疗？",
    "{disease}的治疗方法有哪些？",
    "{disease}可以治愈吗？",
    "得了{disease}该怎么办？",
]

INHERITANCE_TEMPLATES = [
    "{disease}的遗传方式是什么？",
    "{disease}会遗传吗？",
    "{disease}是遗传病吗？",
]

QUERY_TYPE_MAP = {
    "症状": SYMPTOM_TEMPLATES,
    "病因": CAUSE_TEMPLATES,
    "治疗": TREATMENT_TEMPLATES,
    "遗传方式": INHERITANCE_TEMPLATES,
}

# 意图分类测试用例（不走RAG）
INTENT_TEST_CASES = [
    {"question": "你好", "expected_intent": "greeting", "category": "intent"},
    {"question": "你是谁？", "expected_intent": "greeting", "category": "intent"},
    {"question": "今天天气怎么样？", "expected_intent": "out_of_scope", "category": "intent"},
    {"question": "帮我写一首诗", "expected_intent": "out_of_scope", "category": "intent"},
    {"question": "感冒了怎么办？", "expected_intent": "common_disease", "category": "intent"},
    {"question": "发烧怎么退烧？", "expected_intent": "common_disease", "category": "intent"},
    {"question": "高血压吃什么药？", "expected_intent": "common_disease", "category": "intent"},
    {"question": "胃炎怎么治？", "expected_intent": "common_disease", "category": "intent"},
]


def generate_test_cases(max_diseases: int = 0) -> list[dict]:
    """
    从Excel数据集自动生成测试用例

    Args:
        max_diseases: 最多取几个疾病（0=全部）

    Returns:
        list[dict]: 测试用例列表
    """
    df = pd.read_excel(DATASET_PATH).fillna("")
    disease_names = df["disease_name_zh"].astype(str).str.strip().tolist()

    if max_diseases > 0:
        disease_names = disease_names[:max_diseases]

    test_cases = []

    # 为每个疾病生成检索类问题
    for disease in disease_names:
        for query_type, templates in QUERY_TYPE_MAP.items():
            # 每种类型随机选1个模板
            import random
            template = random.choice(templates)
            question = template.format(disease=disease)

            test_cases.append({
                "question": question,
                "expected_disease": disease,
                "expected_query_type": query_type,
                "expected_intent": "medical_rag",
                "category": "retrieval",
            })

    # 添加意图分类测试用例
    test_cases.extend(INTENT_TEST_CASES)

    return test_cases


def evaluate_single(rag_service, test_case: dict) -> dict:
    """
    评测单条测试用例

    Returns:
        dict: 评测结果
    """
    question = test_case["question"]
    expected_disease = test_case.get("expected_disease", "")
    expected_query_type = test_case.get("expected_query_type", "")
    expected_intent = test_case.get("expected_intent", "")
    category = test_case["category"]

    # 运行pipeline
    start_time = time.time()
    result = rag_service.answer(question=question, history="", debug=True)
    latency = time.time() - start_time

    debug_info = result.get("debug_info") or {}
    actual_intent = debug_info.get("intention", "")
    structured_query = debug_info.get("structured_query") or {}
    sources = result.get("sources", [])

    # 评估意图分类
    intent_correct = actual_intent == expected_intent

    # 评估疾病名提取
    actual_disease = structured_query.get("disease_name", "")
    disease_extract_correct = False
    if category == "retrieval":
        disease_extract_correct = (
            expected_disease and actual_disease
            and expected_disease in actual_disease or actual_disease in expected_disease
        )

    # 评估查询类型提取
    actual_query_type = structured_query.get("query_type", "")
    query_type_correct = False
    if category == "retrieval" and expected_query_type:
        query_type_correct = actual_query_type == expected_query_type

    # 评估检索命中率
    retrieval_hit = False
    top1_hit = False
    if category == "retrieval" and sources:
        retrieved_diseases = [s.get("disease_name_zh", "") for s in sources]
        retrieval_hit = any(
            expected_disease in d or d in expected_disease
            for d in retrieved_diseases
        )
        if retrieved_diseases:
            top1_hit = (
                expected_disease in retrieved_diseases[0]
                or retrieved_diseases[0] in expected_disease
            )

    return {
        "question": question,
        "category": category,
        "expected_disease": expected_disease,
        "expected_query_type": expected_query_type,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "actual_disease": actual_disease,
        "actual_query_type": actual_query_type,
        "sources_count": len(sources),
        "sources": [s.get("disease_name_zh", "") for s in sources],
        "intent_correct": intent_correct,
        "disease_extract_correct": disease_extract_correct,
        "query_type_correct": query_type_correct,
        "retrieval_hit": retrieval_hit,
        "top1_hit": top1_hit,
        "latency": round(latency, 2),
        "answer_preview": result.get("answer", "")[:100],
    }


def calculate_metrics(results: list[dict]) -> dict:
    """计算汇总指标"""
    total = len(results)
    if total == 0:
        return {}

    # 意图分类
    intent_results = [r for r in results if r["category"] == "intent"]
    intent_correct = sum(1 for r in intent_results if r["intent_correct"])
    intent_acc = intent_correct / len(intent_results) if intent_results else 0

    # 检索类
    retrieval_results = [r for r in results if r["category"] == "retrieval"]
    n_retrieval = len(retrieval_results)

    disease_correct = sum(1 for r in retrieval_results if r["disease_extract_correct"])
    disease_acc = disease_correct / n_retrieval if n_retrieval else 0

    qt_correct = sum(1 for r in retrieval_results if r["query_type_correct"])
    qt_acc = qt_correct / n_retrieval if n_retrieval else 0

    recall_at_k = sum(1 for r in retrieval_results if r["retrieval_hit"]) / n_retrieval if n_retrieval else 0
    top1_acc = sum(1 for r in retrieval_results if r["top1_hit"]) / n_retrieval if n_retrieval else 0

    avg_latency = sum(r["latency"] for r in results) / total

    return {
        "total_cases": total,
        "retrieval_cases": n_retrieval,
        "intent_cases": len(intent_results),
        "intent_accuracy": round(intent_acc, 4),
        "disease_name_accuracy": round(disease_acc, 4),
        "query_type_accuracy": round(qt_acc, 4),
        "retrieval_recall_at_k": round(recall_at_k, 4),
        "top1_accuracy": round(top1_acc, 4),
        "avg_latency_seconds": round(avg_latency, 2),
    }


def print_report(metrics: dict, results: list[dict]):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("  RAG 系统评测报告")
    print("=" * 60)

    print(f"\n总测试用例数: {metrics['total_cases']}")
    print(f"  - 检索类: {metrics['retrieval_cases']} 条")
    print(f"  - 意图分类: {metrics['intent_cases']} 条")

    print("\n--- 核心指标 ---")
    print(f"意图分类准确率:     {metrics['intent_accuracy']:.1%}")
    print(f"疾病名提取准确率:   {metrics['disease_name_accuracy']:.1%}")
    print(f"查询类型提取准确率: {metrics['query_type_accuracy']:.1%}")
    print(f"检索命中率 (Recall@K): {metrics['retrieval_recall_at_k']:.1%}")
    print(f"Top-1 命中率:       {metrics['top1_accuracy']:.1%}")
    print(f"平均延迟:           {metrics['avg_latency_seconds']:.2f}s")

    # 打印失败用例
    failures = [r for r in results if not r.get("retrieval_hit", True) and r["category"] == "retrieval"]
    if failures:
        print(f"\n--- 检索失败用例 ({len(failures)}条) ---")
        for f in failures[:10]:
            print(f"  Q: {f['question']}")
            print(f"    期望: {f['expected_disease']} | 实际sources: {f.get('sources', [])}")
            print()

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG系统评测")
    parser.add_argument("--max-diseases", type=int, default=0,
                        help="最多评测几个疾病（0=全部，默认全部）")
    parser.add_argument("--output", type=str, default="eval_report",
                        help="输出文件名前缀（默认 eval_report）")
    args = parser.parse_args()

    print("[1/4] 生成测试集...")
    test_cases = generate_test_cases(max_diseases=args.max_diseases)
    print(f"  共生成 {len(test_cases)} 条测试用例")

    print("[2/4] 初始化RAG服务...")
    from rag_service_optimized import OptimizedRAGService
    rag_service = OptimizedRAGService()

    print("[3/4] 运行评测...")
    results = []
    for i, case in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {case['question'][:40]}...", end=" ", flush=True)
        result = evaluate_single(rag_service, case)
        results.append(result)
        status = "OK" if result.get("retrieval_hit", True) or result["category"] == "intent" else "MISS"
        print(f"{status} ({result['latency']}s)")

    print("[4/4] 生成报告...")
    metrics = calculate_metrics(results)
    print_report(metrics, results)

    # 保存JSON报告
    output_dir = Path(__file__).parent
    json_path = output_dir / f"{args.output}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {json_path}")

    # 保存可读报告
    txt_path = output_dir / f"{args.output}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RAG 系统评测报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总测试用例数: {metrics['total_cases']}\n")
        f.write(f"  检索类: {metrics['retrieval_cases']} 条\n")
        f.write(f"  意图分类: {metrics['intent_cases']} 条\n\n")
        f.write("--- 核心指标 ---\n")
        f.write(f"意图分类准确率:       {metrics['intent_accuracy']:.1%}\n")
        f.write(f"疾病名提取准确率:     {metrics['disease_name_accuracy']:.1%}\n")
        f.write(f"查询类型提取准确率:   {metrics['query_type_accuracy']:.1%}\n")
        f.write(f"检索命中率 (Recall@K): {metrics['retrieval_recall_at_k']:.1%}\n")
        f.write(f"Top-1 命中率:         {metrics['top1_accuracy']:.1%}\n")
        f.write(f"平均延迟:             {metrics['avg_latency_seconds']:.2f}s\n\n")

        f.write("--- 逐条结果 ---\n")
        for r in results:
            hit = "PASS" if r.get("retrieval_hit", True) or r["category"] == "intent" else "FAIL"
            f.write(f"[{hit}] {r['question']}\n")
            if r["category"] == "retrieval":
                f.write(f"  期望疾病: {r['expected_disease']}\n")
                f.write(f"  提取疾病: {r['actual_disease']} ({'OK' if r['disease_extract_correct'] else 'WRONG'})\n")
                f.write(f"  提取类型: {r['actual_query_type']} ({'OK' if r['query_type_correct'] else 'WRONG'})\n")
                f.write(f"  检索命中: {'YES' if r['retrieval_hit'] else 'NO'} | Top1: {'YES' if r['top1_hit'] else 'NO'}\n")
            else:
                f.write(f"  期望意图: {r['expected_intent']} | 实际: {r['actual_intent']} ({'OK' if r['intent_correct'] else 'WRONG'})\n")
            f.write(f"  延迟: {r['latency']}s\n\n")

    print(f"可读报告已保存: {txt_path}")


if __name__ == "__main__":
    main()
