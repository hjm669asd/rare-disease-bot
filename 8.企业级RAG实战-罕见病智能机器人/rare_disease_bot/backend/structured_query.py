"""结构化查询提取模块 - 从用户问题中提取结构化信息"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import PromptTemplate


@dataclass
class StructuredQuery:
    """结构化查询对象"""
    disease_name: Optional[str] = None      # 疾病名称
    query_type: Optional[str] = None        # 查询类型
    keywords: list[str] = None              # 关键词列表
    original_question: str = ""             # 原始问题

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


# 查询类型映射
QUERY_TYPE_MAP = {
    "症状": ["症状", "表现", "征兆", "特征"],
    "病因": ["病因", "原因", "发病机制", "遗传", "基因"],
    "治疗": ["治疗", "疗法", "用药", "药物", "手术", "管理"],
    "预后": ["预后", "结局", "生存", "寿命", "并发症"],
    "流行病学": ["发病率", "患病率", "流行", "人群"],
    "诊断": ["诊断", "检查", "检测", "筛查"],
    "遗传方式": ["遗传方式", "遗传模式", "常染色体", "X连锁"],
}

# 疾病别名映射
DISEASE_ALIASES = {
    "PKU": "苯丙酮尿症",
    "苯丙酮尿": "苯丙酮尿症",
    "尿崩症": "尿崩症",
    "血友病": "血友病",
    "白化病": "白化病",
    "蚕豆病": "G6PD缺乏症",
    "G6PD": "G6PD缺乏症",
}


class StructuredQueryExtractor:
    """结构化查询提取器"""

    def __init__(self, llm=None, disease_names: list[str] = None):
        """
        初始化提取器

        Args:
            llm: 语言模型（可选，用于复杂查询提取）
            disease_names: 疾病名称列表（用于精确匹配）
        """
        self.llm = llm
        self.disease_names = disease_names or []
        self._build_disease_patterns()

        self.extraction_prompt = PromptTemplate.from_template("""
从用户问题中提取以下结构化信息：

用户问题: {question}

请用JSON格式返回（只返回JSON，不要其他内容）：
{{
    "disease_name": "疾病名称（中文全称，如果问题中提到了的话，否则为null）",
    "query_type": "查询类型（症状/病因/治疗/预后/流行病学/诊断/遗传方式，如果能判断的话，否则为null）",
    "keywords": ["关键词1", "关键词2", "关键词3"]
}}
""")

    def _build_disease_patterns(self):
        """构建疾病名称匹配模式"""
        self.disease_patterns = []
        for name in self.disease_names:
            # 去除空格和常见后缀
            clean_name = name.strip()
            if clean_name:
                self.disease_patterns.append(clean_name)

    def extract(self, question: str) -> StructuredQuery:
        """
        从用户问题中提取结构化信息

        Args:
            question: 用户原始问题

        Returns:
            StructuredQuery: 结构化查询对象
        """
        query = StructuredQuery(original_question=question)

        # 1. 尝试规则匹配（快速、准确）
        self._extract_by_rules(query)

        # 2. 如果有LLM且规则匹配不完整，使用LLM提取
        if self.llm and (not query.disease_name or not query.query_type):
            try:
                self._extract_by_llm(query)
            except Exception:
                pass  # LLM提取失败，使用规则匹配结果

        # 3. 提取关键词
        if not query.keywords:
            query.keywords = self._extract_keywords(question)

        return query

    def _extract_by_rules(self, query: StructuredQuery):
        """使用规则提取结构化信息"""
        question = query.original_question

        # 1. 提取疾病名称
        query.disease_name = self._extract_disease_name(question)

        # 2. 提取查询类型
        query.query_type = self._extract_query_type(question)

    def _extract_disease_name(self, question: str) -> Optional[str]:
        """提取疾病名称"""
        # 1. 先检查别名
        for alias, full_name in DISEASE_ALIASES.items():
            if alias in question:
                return full_name

        # 2. 检查已知疾病名称列表
        for disease_name in self.disease_patterns:
            if disease_name in question:
                return disease_name

        # 3. 使用正则表达式匹配常见疾病名称模式
        # 中文疾病名称通常以"症"、"病"、"综合征"结尾
        pattern = r'([一-龥]{2,15}(?:症|病|综合征|缺乏症|过多症))'
        match = re.search(pattern, question)
        if match:
            return match.group(1)

        return None

    def _extract_query_type(self, question: str) -> Optional[str]:
        """提取查询类型"""
        for query_type, keywords in QUERY_TYPE_MAP.items():
            for keyword in keywords:
                if keyword in question:
                    return query_type
        return None

    def _extract_keywords(self, question: str) -> list[str]:
        """提取关键词"""
        keywords = []

        # 提取疾病名称作为关键词
        disease_name = self._extract_disease_name(question)
        if disease_name:
            keywords.append(disease_name)

        # 提取查询类型相关关键词
        for query_type, type_keywords in QUERY_TYPE_MAP.items():
            for keyword in type_keywords:
                if keyword in question and keyword not in keywords:
                    keywords.append(keyword)

        # 如果关键词太少，使用简单的分词
        if len(keywords) < 2:
            # 移除常见停用词
            stop_words = ["的", "了", "吗", "呢", "啊", "是", "有", "哪些", "什么", "怎么", "如何"]
            words = list(question)
            for word in stop_words:
                question = question.replace(word, "")
            # 提取2-4字的词组
            simple_keywords = re.findall(r'[一-龥]{2,4}', question)
            for word in simple_keywords:
                if word not in keywords and len(word) >= 2:
                    keywords.append(word)

        return keywords[:5]  # 最多返回5个关键词

    def _extract_by_llm(self, query: StructuredQuery):
        """使用LLM提取结构化信息"""
        response = self.llm.invoke(
            self.extraction_prompt.format(question=query.original_question)
        )

        try:
            # 解析JSON响应
            result = json.loads(response.content.strip())

            if not query.disease_name and result.get("disease_name"):
                query.disease_name = result["disease_name"]

            if not query.query_type and result.get("query_type"):
                query.query_type = result["query_type"]

            if result.get("keywords"):
                # 合并关键词，去重
                existing_keywords = set(query.keywords)
                for kw in result["keywords"]:
                    if kw not in existing_keywords:
                        query.keywords.append(kw)
                        existing_keywords.add(kw)

        except json.JSONDecodeError:
            pass  # JSON解析失败，使用规则匹配结果


def build_search_query(structured_query: StructuredQuery) -> str:
    """
    将结构化查询转换为搜索查询字符串

    Args:
        structured_query: 结构化查询对象

    Returns:
        str: 优化后的搜索查询
    """
    parts = []

    if structured_query.disease_name:
        parts.append(structured_query.disease_name)

    if structured_query.query_type:
        parts.append(structured_query.query_type)

    # 添加关键词
    for keyword in structured_query.keywords:
        if keyword not in parts:
            parts.append(keyword)

    return " ".join(parts) if parts else structured_query.original_question
