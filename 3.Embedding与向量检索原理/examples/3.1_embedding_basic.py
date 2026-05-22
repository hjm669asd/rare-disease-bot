"""
程序运行流程图：

┌────────────────────────────────────┐
│ 1. 从 .env 加载 DASHSCOPE_API_KEY   │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 2. 准备几句医疗示例文本             │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 3. 创建 DashScopeEmbeddings         │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 4. 将文本转换成 embedding 向量      │
└──────────────────┬─────────────────┘
                   ↓
┌────────────────────────────────────┐
│ 5. 打印向量维度和前几个数字         │
└────────────────────────────────────┘

本脚本的教学目标：
理解 Embedding 的最基本概念：文本会被转换成一串数字向量，后续向量检索就是基于这些数字完成的。
"""

# os 用于读取环境变量。
import os

# load_dotenv 用于从项目根目录的 .env 文件中加载 API Key。
from dotenv import load_dotenv

# DashScopeEmbeddings 是 LangChain 对 DashScope 文本向量模型的封装。
from langchain_dashscope import DashScopeEmbeddings


# 医疗教学示例文本。
# 前两句都与高血压饮食相关，第三句与感冒相关。
TEXTS = [
    "高血压患者要少吃盐。",
    "血压高的人应减少钠盐摄入。",
    "感冒患者应注意休息。",
]


def main():
    """演示如何把多段文本转换为 embedding 向量。"""

    # 加载 .env 文件中的 DASHSCOPE_API_KEY。
    load_dotenv()

    # 如果没有配置 API Key，就提前给出清晰错误。
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 创建 embedding 模型。
    # text-embedding-v2 会把文本转换成向量。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # embed_documents() 用于一次性向量化多个文档文本。
    vectors = embeddings.embed_documents(TEXTS)

    # 逐条打印文本和对应向量信息。
    for text, vector in zip(TEXTS, vectors):
        print("\n--- 文本 ---")
        print(text)
        print(f"向量维度: {len(vector)}")
        print(f"前 5 个数字: {vector[:5]}")

    # 观察重点：
    # 1. 每段文本都变成了一个数字列表。
    # 2. 单个数字不需要人工解释。
    # 3. 向量整体用于表示文本语义。


# Python 脚本入口。
if __name__ == "__main__":
    main()
