# LangChain RAG 从 0 到 1：智能医疗机器人教程

这个项目是一个面向初学者的 RAG 教程仓库，以“智能医疗机器人”为业务背景，带你使用 LangChain、通义千问系列模型和 FAISS 从零跑通一个最小可用的 RAG 系统。

> 重要说明：本项目中的医疗内容只用于技术教学，不构成真实医疗建议、诊断或治疗方案。

## 你会学到什么

- RAG 系统的完整数据流
- LangChain 中的核心组件
- 如何加载医疗知识文本
- 如何进行文档切分
- 如何使用 DashScope 生成向量
- 如何用 FAISS 构建本地向量库
- 如何让千问模型基于检索结果回答问题

## 项目结构

```text
RAG0-1/
  README.md
  requirements.txt
  .env.example
  1.认识最基础的rag/
    docs/
      01-rag-overview.md
      02-core-components.md
      03-first-rag-demo.md
    examples/
      data/
        sample.txt
      01_load_documents.py
      02_split_documents.py
      03_build_vectorstore.py
      04_simple_rag_demo.py
    exercises/
      01_core_components.md
      02_run_first_demo.md
    solutions/
      01_core_components.md
      02_run_first_demo.md
  2.文档切分与检索效果/
    docs/
      2.1-数据预处理.md
      2.2-固定长度字符切分.md
      2.3-递归字符切分.md
      2.4-Markdown标题结构切分.md
      2.5-Token切分.md
      2.6-语义切分.md
      2.7-父子文档切分.md
      2.8-检索效果对比实验.md
    examples/
      data/
        messy_medical_text.txt
        clean_medical_text.md
      2.1_preprocess_compare.py
      2.2_fixed_length_split.py
      2.3_recursive_split.py
      2.4_markdown_header_split.py
      2.8_retrieval_compare.py
    exercises/
      2.文档切分与检索效果练习.md
    solutions/
      2.文档切分与检索效果参考答案.md
  3.Embedding与向量检索原理/
    docs/
      3.1-Embedding是什么.md
      3.2-文本如何变成向量.md
      3.3-相似度计算.md
      3.4-FAISS向量库原理.md
      3.5-top_k与检索结果数量.md
      3.6-相似度分数与结果解释.md
      3.7-医疗场景中的检索可靠性.md
    examples/
      data/
        embedding_medical_text.md
      3.1_embedding_basic.py
      3.2_query_embedding.py
      3.3_similarity_compare.py
      3.4_faiss_basic_search.py
      3.5_top_k_compare.py
      3.6_similarity_score.py
      3.7_retrieval_reliability.py
    exercises/
      3.Embedding与向量检索原理练习.md
    solutions/
      3.Embedding与向量检索原理参考答案.md
  4.Retriever检索器与RAG查询流程/
    docs/
      4.1-Retriever是什么.md
      4.2-as_retriever基础用法.md
      4.3-VectorStore与Retriever区别.md
      4.4-检索结果如何进入Prompt.md
      4.5-完整RAG查询流程.md
      4.6-检索失败时怎么办.md
    examples/
      data/
        retriever_medical_text.md
      4.1_retriever_basic.py
      4.2_retriever_k_compare.py
      4.3_vectorstore_vs_retriever.py
      4.4_build_prompt_with_context.py
      4.5_full_rag_query_flow.py
      4.6_empty_or_bad_retrieval.py
    exercises/
      4.Retriever检索器与RAG查询流程练习.md
    solutions/
      4.Retriever检索器与RAG查询流程参考答案.md
  5.Prompt模板与答案生成控制/
    docs/
      5.1-Prompt在RAG中的作用.md
      5.2-基础Prompt模板.md
      5.3-只根据资料回答.md
      5.4-资料不足时如何拒答.md
      5.5-医疗安全声明与风险提示.md
      5.6-结构化答案格式.md
      5.7-带来源引用的回答.md
    examples/
      data/
        prompt_medical_text.md
      5.1_basic_prompt.py
      5.2_prompt_template.py
      5.3_grounded_answer.py
      5.4_refuse_when_context_missing.py
      5.5_medical_safety_prompt.py
      5.6_structured_answer.py
      5.7_answer_with_sources.py
    exercises/
      5.Prompt模板与答案生成控制练习.md
    solutions/
      5.Prompt模板与答案生成控制参考答案.md
  6.LCEL链式编排与完整RAGChain/
    docs/
      6.1-LCEL是什么.md
      6.2-RunnablePassthrough基础.md
      6.3-用管道符组合组件.md
      6.4-构建最小RAGChain.md
      6.5-输出解析器StrOutputParser.md
      6.6-带来源信息的RAGChain.md
      6.7-调试LCEL链路.md
    examples/
      data/
        lcel_medical_text.md
      6.1_lcel_basic.py
      6.2_runnable_passthrough.py
      6.3_pipe_operator.py
      6.4_minimal_rag_chain.py
      6.5_str_output_parser.py
      6.6_rag_chain_with_sources.py
      6.7_debug_lcel_chain.py
    exercises/
      6.LCEL链式编排与完整RAGChain练习.md
    solutions/
      6.LCEL链式编排与完整RAGChain参考答案.md
  7.检索质量优化与进阶检索策略/
    docs/
      7.1-为什么检索质量会影响RAG效果.md
      7.2-查询改写QueryRewrite.md
      7.3-多查询检索MultiQuery.md
      7.4-混合检索HybridSearch.md
      7.5-最大边际相关性MMR.md
      7.6-重排序Rerank.md
      7.7-元数据过滤MetadataFilter.md
      7.8-检索评估与调参.md
      7.9-医疗RAG检索质量安全清单.md
    examples/
      data/
        retrieval_quality_medical_text.md
      7.1_retrieval_failure_cases.py
      7.2_query_rewrite_demo.py
      7.3_multi_query_demo.py
      7.4_hybrid_search_demo.py
      7.5_mmr_demo.py
      7.6_simple_rerank_demo.py
      7.7_metadata_filter_demo.py
      7.8_retrieval_eval_demo.py
      7.9_medical_retrieval_checklist.py
    exercises/
      7.检索质量优化与进阶检索策略练习.md
    solutions/
      7.检索质量优化与进阶检索策略参考答案.md
  8.企业级RAG实战-罕见病智能机器人/
    docs/
      8.1-企业级RAG项目架构.md
      8.2-罕见病知识库设计.md
      8.3-后端API服务设计.md
      8.4-RAG服务分层与配置管理.md
      8.5-会话历史与多轮问答.md
      8.6-检索调试信息与来源引用.md
      8.7-静态WebUI前后端交互.md
      8.8-企业级RAG安全与上线检查清单.md
    rare_disease_bot/
      backend/
        app.py
        config.py
        schemas.py
        rag_service.py
        session_store.py
      frontend/
        index.html
        style.css
        app.js
      README.md
    exercises/
      8.企业级RAG实战练习.md
    solutions/
      8.企业级RAG实战参考答案.md
```

## 环境准备

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制环境变量示例文件：

```bash
copy .env.example .env
```

然后在 `.env` 中填写你的 DashScope API Key：

```text
DASHSCOPE_API_KEY=你的_api_key
```

## 第一章运行顺序

```bash
python "1.认识最基础的rag/examples/01_load_documents.py"
python "1.认识最基础的rag/examples/02_split_documents.py"
python "1.认识最基础的rag/examples/03_build_vectorstore.py"
python "1.认识最基础的rag/examples/04_simple_rag_demo.py"
```

## 第二章运行顺序

```bash
python "2.文档切分与检索效果/examples/2.1_preprocess_compare.py"
python "2.文档切分与检索效果/examples/2.2_fixed_length_split.py"
python "2.文档切分与检索效果/examples/2.3_recursive_split.py"
python "2.文档切分与检索效果/examples/2.4_markdown_header_split.py"
python "2.文档切分与检索效果/examples/2.8_retrieval_compare.py"
```

## 第三章运行顺序

```bash
python "3.Embedding与向量检索原理/examples/3.1_embedding_basic.py"
python "3.Embedding与向量检索原理/examples/3.2_query_embedding.py"
python "3.Embedding与向量检索原理/examples/3.3_similarity_compare.py"
python "3.Embedding与向量检索原理/examples/3.4_faiss_basic_search.py"
python "3.Embedding与向量检索原理/examples/3.5_top_k_compare.py"
python "3.Embedding与向量检索原理/examples/3.6_similarity_score.py"
python "3.Embedding与向量检索原理/examples/3.7_retrieval_reliability.py"
```

## 第四章运行顺序

```bash
python "4.Retriever检索器与RAG查询流程/examples/4.1_retriever_basic.py"
python "4.Retriever检索器与RAG查询流程/examples/4.2_retriever_k_compare.py"
python "4.Retriever检索器与RAG查询流程/examples/4.3_vectorstore_vs_retriever.py"
python "4.Retriever检索器与RAG查询流程/examples/4.4_build_prompt_with_context.py"
python "4.Retriever检索器与RAG查询流程/examples/4.5_full_rag_query_flow.py"
python "4.Retriever检索器与RAG查询流程/examples/4.6_empty_or_bad_retrieval.py"
```

## 第五章运行顺序

```bash
python "5.Prompt模板与答案生成控制/examples/5.1_basic_prompt.py"
python "5.Prompt模板与答案生成控制/examples/5.2_prompt_template.py"
python "5.Prompt模板与答案生成控制/examples/5.3_grounded_answer.py"
python "5.Prompt模板与答案生成控制/examples/5.4_refuse_when_context_missing.py"
python "5.Prompt模板与答案生成控制/examples/5.5_medical_safety_prompt.py"
python "5.Prompt模板与答案生成控制/examples/5.6_structured_answer.py"
python "5.Prompt模板与答案生成控制/examples/5.7_answer_with_sources.py"
```

## 第六章运行顺序

```bash
python "6.LCEL链式编排与完整RAGChain/examples/6.1_lcel_basic.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.2_runnable_passthrough.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.3_pipe_operator.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.4_minimal_rag_chain.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.5_str_output_parser.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.6_rag_chain_with_sources.py"
python "6.LCEL链式编排与完整RAGChain/examples/6.7_debug_lcel_chain.py"
```

## 第七章运行顺序

```bash
python "7.检索质量优化与进阶检索策略/examples/7.1_retrieval_failure_cases.py"
python "7.检索质量优化与进阶检索策略/examples/7.2_query_rewrite_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.3_multi_query_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.4_hybrid_search_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.5_mmr_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.6_simple_rerank_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.7_metadata_filter_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.8_retrieval_eval_demo.py"
python "7.检索质量优化与进阶检索策略/examples/7.9_medical_retrieval_checklist.py"
```

## 第八章运行方式

第八章是前后端分离的企业级 RAG 实战项目。

后端启动：

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir "8.企业级RAG实战-罕见病智能机器人/rare_disease_bot/backend"
```

前端打开：

```text
8.企业级RAG实战-罕见病智能机器人/rare_disease_bot/frontend/index.html
```

详细说明见：

```text
8.企业级RAG实战-罕见病智能机器人/rare_disease_bot/README.md
```

## 学习路径

1. 学习 `1.认识最基础的rag`，理解最小 RAG 流程。
2. 学习 `2.文档切分与检索效果`，理解数据预处理、主流切分策略和检索效果对比。
3. 学习 `3.Embedding与向量检索原理`，理解文本向量化、相似度、FAISS、top_k 和检索可靠性。
4. 学习 `4.Retriever检索器与RAG查询流程`，理解 Retriever、Prompt 拼接、完整查询链路和检索失败处理。
5. 学习 `5.Prompt模板与答案生成控制`，理解 Prompt 模板、资料约束、拒答、安全声明、结构化输出和来源引用。
6. 学习 `6.LCEL链式编排与完整RAGChain`，理解 LCEL、RunnablePassthrough、管道符、完整 RAG Chain、来源引用和链路调试。
7. 学习 `7.检索质量优化与进阶检索策略`，理解查询改写、多查询、混合检索、MMR、Rerank、Metadata Filter、检索评估和医疗安全清单。
8. 学习 `8.企业级RAG实战-罕见病智能机器人`，理解 FastAPI 服务化、静态 WebUI、会话历史、来源引用和企业级安全检查。
9. 按章节运行 examples 或项目 Demo。
10. 完成 exercises 中的练习。
11. 对照 solutions 中的参考答案。

## 后续路线图

- 中级模块：token 切分、语义切分、查询改写、意图识别、多路检索、重排。
- 高级模块：企业级医疗 RAG 架构、权限控制、日志追踪、评测体系、服务化部署。
