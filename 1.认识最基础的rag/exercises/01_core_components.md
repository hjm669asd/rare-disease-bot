# 练习 01：识别 RAG 核心组件

## 练习目标

通过阅读示例代码，理解一个最小 RAG 系统由哪些组件组成。

## 任务

打开 `examples/beginner/04_simple_rag_demo.py`，回答下面的问题：

1. 哪一段代码负责加载本地医疗知识文本？
    DATA_PATH = BASE_DIR / "data" / "sample.txt"
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")

2. 哪一个组件负责把长文本切成小片段？
    文本切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=40,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

3. 哪一个组件负责把文本转换成向量？
    嵌入模型


4. 哪一个组件负责保存和检索向量？
    faiss向量数据库

5. Prompt 中有哪些医疗安全约束？
    不说没有依据的话
    有安全提示词，提醒用户ai不能代替医生

6. 为什么医疗机器人不能只让大模型直接回答？
    1.模型不能获取最新数据
    2.模型存在幻觉，会瞎编一些内容

## 思考题

如果用户问“胸痛应该怎么办？”，当前知识库中是否有相关内容？系统可能会检索到哪一段资料？
    当前知识库中并没有相关内容，但是系统可能会按token检索，就是说把词相关的也检索出来，比如“胸痛”“应该”“怎么办”，按关键词匹配，看哪篇资料有和关键词类似的词，然后检索出来。
