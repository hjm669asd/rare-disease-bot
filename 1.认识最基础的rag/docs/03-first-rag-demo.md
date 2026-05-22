# 03 跑通第一个智能医疗 RAG Demo

本节目标是跑通一个最小可用的智能医疗机器人 RAG Demo。

## Demo 做什么

Demo 会完成下面几件事：

1. 读取本地医疗科普文本。
2. 把文本切成多个片段。
3. 使用 DashScope embedding 模型生成向量。
4. 把向量写入 FAISS 本地向量库。
5. 根据用户问题检索相关片段。
6. 调用通义千问模型生成回答。

## 运行前准备

安装依赖：

```bash
pip install -r requirements.txt
```

配置 `.env`：

```text
DASHSCOPE_API_KEY=你的_api_key
```

## 推荐运行顺序

先分步骤理解每个组件：

```bash
python examples/beginner/01_load_documents.py
python examples/beginner/02_split_documents.py
python examples/beginner/03_build_vectorstore.py
```

再运行完整 Demo：

```bash
python examples/beginner/04_simple_rag_demo.py
```

## 你应该观察什么

运行脚本时重点观察：

- 原始文档被加载成了几个 `Document`。
- 文档被切成了几个 chunk。
- 每个 chunk 大概包含什么内容。
- 检索器针对问题返回了哪些片段。
- 最终回答是否只基于检索内容生成。

## 医疗场景下的安全边界

这个 Demo 只用于理解 RAG 技术，不应该用于真实诊疗。

真实医疗机器人还需要增加：

- 权威知识来源管理
- 医生审核流程
- 风险提示
- 用户身份与权限控制
- 回答可追溯引用
- 医疗安全评测

这些内容会放到后续高级模块中展开。
