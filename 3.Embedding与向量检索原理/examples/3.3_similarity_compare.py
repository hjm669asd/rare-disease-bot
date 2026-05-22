"""
程序运行流程图：

准备几个二维向量
   ↓
定义余弦相似度函数
   ↓
分别计算 query 与候选向量的相似度
   ↓
打印相似度结果
   ↓
理解“越接近 1 越相似”

本脚本的教学目标：
不用调用外部模型，用最简单的二维向量手动理解相似度计算。
"""

# math 是 Python 标准库，用于开平方等数学计算。
import math


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。

    参数说明：
    vector_a：第一个向量。
    vector_b：第二个向量。

    返回值：
    一个相似度分数。越接近 1，方向越接近，语义上可理解为越相似。
    """

    # 点积：对应位置相乘后求和。
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    # 向量 A 的长度。
    norm_a = math.sqrt(sum(a * a for a in vector_a))

    # 向量 B 的长度。
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    # 余弦相似度公式。
    return dot_product / (norm_a * norm_b)


def main():
    """演示余弦相似度计算。"""

    # 假设 query_vector 表示“高血压饮食问题”的方向。
    query_vector = [1, 0]

    # hypertension_vector 与 query_vector 方向接近。
    hypertension_vector = [0.9, 0.1]

    # cold_vector 与 query_vector 方向差异较大。
    cold_vector = [0.1, 0.9]

    print("高血压饮食向量相似度:")
    print(cosine_similarity(query_vector, hypertension_vector))

    print("感冒护理向量相似度:")
    print(cosine_similarity(query_vector, cold_vector))

    # 观察重点：
    # 1. 相似度高的向量会优先被检索到。
    # 2. 实际 embedding 向量不是二维，而是高维。
    # 3. 原理仍然是比较向量之间的接近程度。


if __name__ == "__main__":
    main()
