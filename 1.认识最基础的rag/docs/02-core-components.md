# 02 LangChain RAG 核心组件

LangChain 把 RAG 流程拆成多个可组合组件。初学时不需要一次理解所有高级能力，先掌握最核心的几个组件即可。

## Document Loader

Loader 负责把外部资料加载成 LangChain 能处理的 `Document` 对象。

在智能医疗机器人场景中，资料可能来自：

- 医疗科普文档
- 药品说明书
- 就诊流程说明
- 院内 FAQ
- 临床路径或护理宣教材料

本项目初级 Demo 先使用本地 `txt` 文件。

## Text Splitter

大模型和向量模型都有上下文长度限制，所以通常不会把整篇文档直接入库，而是切分成较小片段。

常见参数：

- `chunk_size`：每个片段的目标长度。
- `chunk_overlap`：相邻片段之间保留多少重叠内容。

初级阶段使用字符切分。中级阶段再学习 token 切分、语义切分等策略。

## Embedding

Embedding 模型负责把文本转换成向量。语义相近的文本，向量距离也会更近。

本项目使用 DashScope 提供的 embedding 模型，让医疗知识片段和用户问题都能进入同一个向量空间。

## VectorStore

VectorStore 负责保存向量，并支持相似度检索。

本项目初级 Demo 使用 FAISS，因为它适合本地快速实验，不需要额外启动数据库服务。

## Retriever

Retriever 是检索器。它接收用户问题，返回与问题最相关的若干个文档片段。

例如用户问：

```text
糖尿病患者饮食要注意什么？
```

Retriever 会从知识库中找出糖尿病饮食、运动、血糖监测等相关片段。

## Prompt

Prompt 负责把“检索到的资料”和“用户问题”组织成模型更容易遵循的格式。

医疗场景中，Prompt 应明确要求模型：

- 只能基于给定资料回答。
- 不确定时说明资料不足。
- 不替代医生诊断。

## LLM

LLM 负责最终生成自然语言答案。

本项目使用 DashScope 上的通义千问系列模型。

## Chain

Chain 把多个组件连接起来。一个最小 RAG Chain 可以理解为：

```text
Retriever → Prompt → LLM → Output Parser
```

初级阶段先看清楚每个组件的职责，再逐步学习更复杂的链式编排。
