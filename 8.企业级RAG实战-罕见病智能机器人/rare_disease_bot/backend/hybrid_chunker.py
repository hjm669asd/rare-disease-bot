"""混合切分器 - 针对结构化Excel数据的切分策略"""

from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.documents import Document


# 字段配置：定义每个字段的用途和切分方式
FIELD_CONFIG = {
    "description": {
        "label": "疾病介绍",
        "index_type": "primary",  # 主索引
    },
    "symptoms": {
        "label": "常见症状",
        "index_type": "secondary",  # 辅助索引
        "query_types": ["症状"],
    },
    "causes": {
        "label": "可能病因",
        "index_type": "secondary",
        "query_types": ["病因", "遗传方式"],
    },
    "treatment": {
        "label": "治疗与管理",
        "index_type": "secondary",
        "query_types": ["治疗"],
    },
    "prognosis": {
        "label": "预后",
        "index_type": "secondary",
        "query_types": ["预后"],
    },
    "prevalence": {
        "label": "流行病学",
        "index_type": "secondary",
        "query_types": ["流行病学"],
    },
    "inheritance": {
        "label": "遗传方式",
        "index_type": "secondary",
        "query_types": ["遗传方式"],
    },
}


class HybridChunker:
    """混合切分器 - 为Excel结构化数据创建主索引和辅助索引"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        """
        初始化混合切分器

        Args:
            chunk_size: 主索引的chunk大小
            chunk_overlap: 主索引的chunk重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_excel(self, dataset_path: Path) -> dict[str, list[Document]]:
        """
        切分Excel文件，返回主索引和辅助索引

        Args:
            dataset_path: Excel文件路径

        Returns:
            dict: 包含 'primary' 和 'secondary' 两个Document列表
        """
        dataframe = pd.read_excel(dataset_path).fillna("")

        # 验证必需列
        required_columns = [
            "disease_name_zh", "disease_name_en", "category",
            "description", "symptoms", "prevalence", "causes",
            "treatment", "prognosis", "inheritance",
        ]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"数据集缺少列: {missing_columns}")

        primary_docs = []
        secondary_docs = []

        for index, row in dataframe.iterrows():
            disease_name_zh = str(row["disease_name_zh"]).strip()
            disease_name_en = str(row["disease_name_en"]).strip()
            category = str(row["category"]).strip()
            row_index = int(index) + 2  # Excel行号（从第2行开始）

            # 基础metadata
            base_metadata = {
                "source": dataset_path.name,
                "row_index": row_index,
                "disease_name_zh": disease_name_zh,
                "disease_name_en": disease_name_en,
                "category": category,
            }

            # 1. 创建主索引文档（完整疾病信息）
            primary_content = self._build_primary_content(row)
            primary_metadata = {
                **base_metadata,
                "index_type": "primary",
                "doc_type": "full_disease",
            }
            primary_docs.append(Document(
                page_content=primary_content,
                metadata=primary_metadata,
            ))

            # 2. 创建辅助索引文档（按字段拆分）
            for field_name, field_config in FIELD_CONFIG.items():
                if field_config["index_type"] != "secondary":
                    continue

                field_value = str(row.get(field_name, "")).strip()
                if not field_value:
                    continue

                # 构建字段文档内容
                secondary_content = self._build_secondary_content(
                    disease_name_zh, disease_name_en, category,
                    field_config["label"], field_value
                )

                secondary_metadata = {
                    **base_metadata,
                    "index_type": "secondary",
                    "doc_type": "field_segment",
                    "field_name": field_name,
                    "field_label": field_config["label"],
                    "query_types": field_config.get("query_types", []),
                }

                secondary_docs.append(Document(
                    page_content=secondary_content,
                    metadata=secondary_metadata,
                ))

        return {
            "primary": primary_docs,
            "secondary": secondary_docs,
        }

    def _build_primary_content(self, row: pd.Series) -> str:
        """构建主索引文档内容"""
        return f"""
疾病中文名：{str(row['disease_name_zh']).strip()}
疾病英文名：{str(row['disease_name_en']).strip()}
疾病类别：{str(row['category']).strip()}
疾病介绍：{str(row['description']).strip()}
常见症状：{str(row['symptoms']).strip()}
流行病学：{str(row['prevalence']).strip()}
可能病因：{str(row['causes']).strip()}
治疗与管理：{str(row['treatment']).strip()}
预后：{str(row['prognosis']).strip()}
遗传方式：{str(row['inheritance']).strip()}
""".strip()

    def _build_secondary_content(
        self,
        disease_name_zh: str,
        disease_name_en: str,
        category: str,
        field_label: str,
        field_value: str,
    ) -> str:
        """构建辅助索引文档内容"""
        return f"""
疾病：{disease_name_zh} ({disease_name_en})
类别：{category}
{field_label}：{field_value}
""".strip()

    def get_disease_names(self, dataset_path: Path) -> list[str]:
        """获取所有疾病名称列表"""
        dataframe = pd.read_excel(dataset_path).fillna("")
        return dataframe["disease_name_zh"].astype(str).str.strip().tolist()


class SmartChunker:
    """智能切分器 - 根据query_type选择性切分"""

    def __init__(self):
        pass

    def should_use_secondary_index(self, query_type: Optional[str]) -> bool:
        """判断是否应该使用辅助索引"""
        if not query_type:
            return False

        # 症状、治疗、病因等查询适合使用辅助索引
        secondary_query_types = ["症状", "病因", "治疗", "预后", "流行病学", "遗传方式"]
        return query_type in secondary_query_types

    def get_optimal_top_k(
        self,
        query_type: Optional[str],
        disease_name: Optional[str],
        default_k: int = 4,
    ) -> int:
        """根据查询类型返回最优的top_k值"""
        # 如果有明确的疾病名称，只需要1-2个结果
        if disease_name:
            return 2

        # 如果有明确的查询类型，可以减少结果数
        if query_type:
            return 3

        # 默认返回4个结果
        return default_k
