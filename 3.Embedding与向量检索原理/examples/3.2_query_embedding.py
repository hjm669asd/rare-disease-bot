"""
程序运行流程图：

读取 .env 中的 API Key
   ↓
准备一个用户问题
   ↓
创建 DashScopeEmbeddings
   ↓
使用 embed_query() 将问题转换成向量
   ↓
打印问题向量的维度和前 10 个数字

本脚本的教学目标：
理解用户问题也需要被 embedding。RAG 检索时不是直接拿中文问题去搜索，而是先把问题转换成 query vector。
"""

import os

from dotenv import load_dotenv
from langchain_dashscope import DashScopeEmbeddings


QUESTION = "高血压患者饮食要注意什么？"


def main():
    """演示如何把用户问题转换成查询向量。"""

    # 加载环境变量。
    load_dotenv()

    # 检查 API Key。
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 创建 DashScope embedding 模型。
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # embed_query() 用于向量化一个用户查询。
    # 它和 embed_documents() 的目标类似，但语义上更适合查询文本。
    query_vector = embeddings.embed_query(QUESTION)

    print(f"用户问题: {QUESTION}")
    print(f"问题向量维度: {len(query_vector)}")
    print(f"前 10 个数字: {query_vector[:10]}")

    # 观察重点：
    # 1. 用户问题被转换成了数字向量。
    # 2. 后续会用这个向量去匹配知识库 chunk 向量。


if __name__ == "__main__":
    main()
