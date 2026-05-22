# 罕见病智能机器人

基于 LangChain + FAISS + DashScope + FastAPI 的企业级 RAG 教学项目。

> 本项目仅用于健康科普和 RAG 技术教学，不能替代医生诊断或治疗建议。

## 核心特性

- **多阶段 RAG Pipeline**：意图识别 → 结构化查询提取 → 混合检索 → Rerank重排序 → LLM生成
- **双模型架构**：弱模型（qwen-turbo）负责意图识别/rerank，强模型（qwen-plus）负责RAG生成
- **混合检索**：精确匹配 + 向量搜索，主索引（完整疾病文档）+ 辅助索引（字段片段）
- **意图路由**：自动过滤问候、常见病、超出范围的问题，减少不必要的RAG调用
- **流式输出**：所有回答均支持SSE流式输出
- **文件上传**：支持TXT/PDF/Word/Excel上传到知识库，增量更新+去重
- **向量库持久化**：FAISS索引缓存到磁盘，启动时间从2分钟降至1秒
- **会话持久化**：基于SQLite的会话历史，重启不丢失

## 项目结构

```
rare_disease_bot/
├── backend/
│   ├── app.py                    # FastAPI入口（含流式输出、文件上传API）
│   ├── config.py                 # 配置管理（从.env加载）
│   ├── schemas.py                # Pydantic请求/响应模型
│   ├── rag_service_optimized.py  # 核心RAG服务（意图→检索→rerank→生成）
│   ├── intent_router.py          # 意图识别模块
│   ├── structured_query.py       # 结构化查询提取
│   ├── hybrid_chunker.py         # 混合切分器（主索引+辅助索引）
│   ├── hybrid_retriever.py       # 混合检索器（精确匹配+向量搜索）
│   ├── reranker.py               # Rerank重排序（gte-rerank-v2）
│   ├── file_parser.py            # 多格式文件解析器
│   ├── file_uploader.py          # 文件上传管理器
│   ├── session_store.py          # SQLite会话存储
│   ├── evaluate.py               # RAG评测脚本
│   └── storage/                  # 运行时数据（自动创建）
├── frontend/
│   ├── index.html                # 多窗口聊天界面
│   ├── style.css                 # 样式
│   └── app.js                    # 前端逻辑（流式输出、文件上传）
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

在项目根目录（`RAG0-1/`）创建 `.env` 文件：

```text
DASHSCOPE_API_KEY=你的DashScope API Key
CHAT_MODEL_NAME=qwen-plus
LIGHTWEIGHT_MODEL_NAME=qwen-turbo
EMBEDDING_MODEL_NAME=text-embedding-v3
```

### 3. 启动后端

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

首次启动会构建向量库（约2分钟），后续启动自动加载缓存（约1秒）。

### 4. 打开前端

浏览器打开 `frontend/index.html`，即可开始对话。

## 系统流程

```
用户输入
  ↓
意图识别（弱模型）
  ├── 问候/闲聊 → 弱模型直接回答
  ├── 常见病    → 强模型自身知识回答（不调用RAG）
  ├── 超出范围  → 预设回复
  └── 罕见病相关 → 进入RAG流程 ↓
        ↓
结构化查询提取（疾病名、查询类型、关键词）
        ↓
混合检索（精确匹配 + 向量搜索，主索引 + 辅助索引）
        ↓
Rerank重排序（gte-rerank-v2）
        ↓
强模型生成回答（流式输出）
        ↓
追加医疗免责声明
```

## API接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | RAG问答（非流式） |
| POST | `/api/chat/stream` | RAG问答（SSE流式输出） |
| POST | `/api/upload` | 上传文件到知识库 |
| GET | `/api/files` | 获取已上传文件列表 |
| DELETE | `/api/files/{file_id}` | 删除已上传文件 |
| GET | `/api/files/supported-formats` | 获取支持的文件格式 |
| GET | `/api/sessions/{session_id}` | 获取会话历史 |
| DELETE | `/api/sessions/{session_id}` | 清空会话历史 |

## 数据集

默认读取上级目录的 `罕见病数据_完整版.xlsx`，表头需包含：

```text
disease_name_zh, disease_name_en, category, description,
symptoms, prevalence, causes, treatment, prognosis, inheritance
```

## 评测

```bash
cd backend

# 评测全部疾病
python evaluate.py

# 评测指定数量
python evaluate.py --max-diseases 10
```

评测报告输出到 `eval_report.json` 和 `eval_report.txt`。

## 教学观察重点

- 意图识别如何过滤非RAG问题，降低token消耗
- 结构化查询如何提升检索精度
- 混合检索（精确匹配+向量搜索）对比纯向量搜索的优势
- Rerank重排序如何提升结果相关性
- 流式输出的SSE协议实现
- FAISS向量库持久化对启动速度的优化
