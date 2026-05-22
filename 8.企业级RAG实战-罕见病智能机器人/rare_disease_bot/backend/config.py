import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CHAPTER_DIR = PROJECT_DIR.parent
ROOT_DIR = CHAPTER_DIR.parent

load_dotenv(ROOT_DIR / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen-turbo")
LIGHTWEIGHT_MODEL_NAME = os.getenv("LIGHTWEIGHT_MODEL_NAME", "qwen-turbo")  # 轻量模型，用于意图识别
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")

DATASET_PATH = Path(os.getenv("RARE_DISEASE_DATASET_PATH", CHAPTER_DIR / "罕见病数据_完整版.xlsx"))

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "4"))
MAX_HISTORY_ROUNDS = int(os.getenv("RAG_MAX_HISTORY_ROUNDS", "3"))

SERVICE_NAME = "rare-disease-rag-bot"
MEDICAL_DISCLAIMER = "本回答仅用于健康科普和技术演示，不能替代医生诊断或治疗建议。如有不适，请及时咨询正规医疗机构专业医生。"
