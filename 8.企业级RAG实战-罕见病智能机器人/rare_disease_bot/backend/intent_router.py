"""意图识别模块 - 使用轻量模型判断用户输入是否需要调用RAG"""

from enum import Enum
from typing import Optional

from langchain_core.prompts import PromptTemplate


class IntentionType(Enum):
    GREETING = "greeting"           # 问候/闲聊
    SIMPLE_QA = "simple_qa"         # 简单问答
    COMMON_DISEASE = "common_disease"  # 常见病（不需要RAG）
    MEDICAL_RAG = "medical_rag"     # 医疗专业问答（罕见病）
    OUT_OF_SCOPE = "out_of_scope"   # 超出范围


# 常见病列表（不需要调用RAG）
COMMON_DISEASES = [
    # 呼吸系统
    "感冒", "流感", "发烧", "咳嗽", "咽炎", "扁桃体炎", "支气管炎", "肺炎",
    "鼻炎", "鼻窦炎", "哮喘", "上呼吸道感染",
    # 消化系统
    "胃炎", "胃溃疡", "肠炎", "腹泻", "便秘", "消化不良", "胃食管反流",
    "胆囊炎", "胰腺炎", "阑尾炎",
    # 心血管
    "高血压", "低血压", "心律失常", "冠心病", "心绞痛", "心肌炎",
    # 内分泌
    "糖尿病", "甲亢", "甲减", "甲状腺炎",
    # 神经系统
    "偏头痛", "神经衰弱", "失眠", "焦虑症", "抑郁症",
    # 骨骼肌肉
    "关节炎", "风湿", "痛风", "腰椎间盘突出", "颈椎病", "肩周炎",
    "骨折", "扭伤", "肌肉拉伤",
    # 皮肤
    "湿疹", "荨麻疹", "皮炎", "痤疮", "脚气", "真菌感染",
    # 眼耳鼻喉
    "结膜炎", "中耳炎", "牙周炎", "龋齿", "口腔溃疡",
    # 泌尿系统
    "尿路感染", "肾结石", "膀胱炎",
    # 妇科
    "月经不调", "痛经", "阴道炎",
    # 其他
    "贫血", "过敏", "水痘", "麻疹", "腮腺炎", "手足口病",
]


INTENTION_PROMPT = PromptTemplate.from_template("""
你是一个意图分类器，请判断用户输入的意图类型。

请只返回以下类别之一（不要返回任何其他内容）：
- greeting: 问候、感谢、告别、闲聊（如：你好、谢谢、再见、在吗）
- simple_qa: 简单的常识性问题，不需要专业知识（如：1+1=?、今天星期几）
- medical_rag: 需要查询罕见病知识库的专业医疗问题（如：某疾病有哪些症状、某病怎么治疗）
- out_of_scope: 与医疗无关的其他专业问题（如：Python怎么爬虫、怎么做菜）

用户输入: {input}
""")


class IntentionRouter:
    """意图路由器 - 判断用户问题类型"""

    def __init__(self, llm):
        self.llm = llm

    def classify(self, user_input: str) -> IntentionType:
        """对用户输入进行意图分类"""
        # 简单规则预过滤（避免调用LLM）
        simple_result = self._simple_classify(user_input)
        if simple_result:
            return simple_result

        # 调用LLM进行意图分类
        try:
            response = self.llm.invoke(
                INTENTION_PROMPT.format(input=user_input)
            )
            intention_str = response.content.strip().lower()

            # 映射到枚举类型
            intention_map = {
                "greeting": IntentionType.GREETING,
                "simple_qa": IntentionType.SIMPLE_QA,
                "medical_rag": IntentionType.MEDICAL_RAG,
                "out_of_scope": IntentionType.OUT_OF_SCOPE,
            }

            return intention_map.get(intention_str, IntentionType.MEDICAL_RAG)
        except Exception:
            # 分类失败时默认走RAG流程
            return IntentionType.MEDICAL_RAG

    def _simple_classify(self, user_input: str) -> Optional[IntentionType]:
        """简单规则预过滤，减少LLM调用"""
        input_lower = user_input.lower().strip()

        # 问候/闲聊关键词
        greeting_keywords = [
            "你好", "您好", "hi", "hello", "嗨", "在吗",
            "谢谢", "感谢", "再见", "拜拜", "bye",
            "你是谁", "你叫什么", "自我介绍",
        ]
        if any(keyword in input_lower for keyword in greeting_keywords):
            return IntentionType.GREETING

        # 简单问答关键词
        simple_keywords = [
            "1+1", "2+2", "今天星期", "几点", "天气",
            "一年有几个月", "太阳从哪里升起",
        ]
        if any(keyword in input_lower for keyword in simple_keywords):
            return IntentionType.SIMPLE_QA

        # 明显的非医疗专业问题
        out_of_scope_keywords = [
            "python", "java", "编程", "代码", "爬虫",
            "怎么做菜", "红烧肉", "炒菜",
            "化妆", "护肤", "穿搭",
        ]
        if any(keyword in input_lower for keyword in out_of_scope_keywords):
            return IntentionType.OUT_OF_SCOPE

        # 检查是否是常见病（不需要RAG）
        if self._is_common_disease(input_lower):
            return IntentionType.COMMON_DISEASE

        return None  # 无法确定，交给LLM判断

    def _is_common_disease(self, user_input: str) -> bool:
        """检查是否是常见病"""
        for disease in COMMON_DISEASES:
            if disease in user_input:
                return True
        return False

    def get_response_for_non_rag(self, intention: IntentionType, user_input: str) -> Optional[str]:
        """为非RAG意图生成预设回复"""
        if intention == IntentionType.GREETING:
            return "你好！我是罕见病智能科普机器人，可以帮你查询罕见病相关的医学知识。请问有什么可以帮你的？"

        elif intention == IntentionType.SIMPLE_QA:
            return "这是一个常识性问题，不需要查询罕见病知识库。不过我是专注于罕见病科普的机器人，如果你有关于罕见病的问题，我很乐意帮你解答！"

        elif intention == IntentionType.COMMON_DISEASE:
            return "您提到的是常见疾病，不是罕见病，我的知识库中没有收录。我是专注于罕见病科普的机器人，如果您有关于罕见病（如苯丙酮尿症、血友病、白化病等）的问题，我很乐意帮您解答！"

        elif intention == IntentionType.OUT_OF_SCOPE:
            return "抱歉，我是专注于罕见病科普的机器人，这个问题超出了我的专业范围。如果你有关于罕见病的问题，我很乐意帮你解答！"

        return None
