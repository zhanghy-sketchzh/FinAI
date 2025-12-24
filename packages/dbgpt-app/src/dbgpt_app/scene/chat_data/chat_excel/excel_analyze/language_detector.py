"""
语言检测工具
检测用户输入是中文还是英文，用于动态选择prompt语言
"""

import logging
import re

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    检测文本的主要语言

    Args:
        text: 要检测的文本

    Returns:
        "zh" 如果主要是中文，"en" 如果主要是英文或其他语言
    """
    if not text or not text.strip():
        # 默认返回英文
        return "en"

    # 统计中文字符数量
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    chinese_chars = chinese_pattern.findall(text)
    chinese_count = sum(len(match) for match in chinese_chars)

    # 计算总字符数（排除空格和标点）
    total_chars = len(re.sub(r'[\s\.,;:!?\'"\-_()\[\]{}]', "", text))

    if total_chars == 0:
        return "en"  # 默认英文

    # 计算中文占比
    chinese_ratio = chinese_count / total_chars if total_chars > 0 else 0

    # 如果中文占比超过30%，认为是中文输入
    if chinese_ratio > 0.3:
        logger.debug(f"检测到中文输入 (中文占比: {chinese_ratio:.2%})")
        return "zh"
    else:
        logger.debug(f"检测到英文输入 (中文占比: {chinese_ratio:.2%})")
        return "en"


def get_prompt_language(user_input: str, default_language: str = "en") -> str:
    """
    根据用户输入获取prompt语言

    Args:
        user_input: 用户输入文本
        default_language: 默认语言（如果无法确定）

    Returns:
        "zh" 或 "en"
    """
    detected = detect_language(user_input)
    logger.info(f"用户输入语言检测: {detected} (输入: {user_input[:50]}...)")
    return detected
