"""
Query改写Agent - 参考format_sql/backend/agents/query_rewrite_assistant.py
负责根据数据理解信息，补充完善用户问题，明确相关列和分析建议
"""
# ruff: noqa: E501

import json
import logging
from typing import AsyncIterator, Dict, List, Union

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """自定义异常：JSON解析失败"""

    pass


class InvalidColumnError(Exception):
    """自定义异常：字段名不存在"""

    def __init__(self, message: str, invalid_columns: list):
        super().__init__(message)
        self.invalid_columns = invalid_columns


def detect_language(text: str) -> str:
    """
    检测文本的主要语言

    Args:
        text: 要检测的文本

    Returns:
        "zh" 如果主要是中文，"en" 如果主要是英文或其他语言
    """
    if not text or not text.strip():
        return "en"

    import re

    # 统计中文字符数量
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    chinese_chars = chinese_pattern.findall(text)
    chinese_count = sum(len(match) for match in chinese_chars)

    # 计算总字符数（排除空格和标点）
    total_chars = len(re.sub(r'[\s\.,;:!?\'"\-_()\[\]{}]', "", text))

    if total_chars == 0:
        return "en"

    # 计算中文占比
    chinese_ratio = chinese_count / total_chars if total_chars > 0 else 0

    # 如果中文占比超过30%，认为是中文输入
    return "zh" if chinese_ratio > 0.3 else "en"


class QueryRewriteAgent:
    """
    Query改写Agent - 简化版本
    功能：
    1. 根据数据字段信息，补充完善用户的提问
    2. 明确指出可能用到的列
    3. 提供分析建议和逻辑支撑
    """

    def __init__(self, llm_client=None, model_name=None):
        """
        Args:
            llm_client: LLM客户端（可选，如果为None则返回默认结果）
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def _extract_valid_column_names(self, table_schema_json: str) -> set:
        """
        从schema中提取所有有效的字段名
        
        Returns:
            有效字段名的集合
        """
        try:
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            if not isinstance(schema_obj, dict):
                return set()

            columns = schema_obj.get("columns", [])
            return {col.get("column_name", "") for col in columns if col.get("column_name")}
        except Exception as e:
            logger.warning(f"提取字段名失败: {e}")
            return set()

    def _simplify_schema_for_rewrite(self, table_schema_json: str) -> str:
        """
        精简schema用于query改写，只保留必要字段信息
        移除 suggested_questions_zh、suggested_questions_en 等不必要字段
        """
        try:
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            if not isinstance(schema_obj, dict):
                return table_schema_json

            # 只保留columns字段，移除建议问题等
            simplified = {}
            if "columns" in schema_obj:
                # 精简每个列的信息
                simplified_columns = []
                for col in schema_obj["columns"]:
                    simplified_col = {
                        "column_name": col.get("column_name", ""),
                        "data_type": col.get("data_type", ""),
                    }
                    # 保留关键字段标记
                    if col.get("is_key_field"):
                        simplified_col["is_key_field"] = True
                    # 保留业务知识
                    if col.get("domain_knowledge"):
                        simplified_col["domain_knowledge"] = col["domain_knowledge"]
                    # 保留分类字段的可选值（精简版）
                    if col.get("unique_values_top20"):
                        values = col["unique_values_top20"]
                        # 最多保留10个值
                        simplified_col["possible_values"] = values[:10] if len(values) > 10 else values
                    # 保留数值统计摘要
                    if col.get("statistics_summary"):
                        simplified_col["stats"] = col["statistics_summary"]
                    simplified_columns.append(simplified_col)
                simplified["columns"] = simplified_columns

            # 保留表名等基本信息
            if "table_name" in schema_obj:
                simplified["table_name"] = schema_obj["table_name"]

            return json.dumps(simplified, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"精简schema失败: {e}，使用原始schema")
            return table_schema_json if isinstance(table_schema_json, str) else json.dumps(table_schema_json, ensure_ascii=False)

    async def rewrite_query_stream(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
        sample_rows: list = None,
    ) -> AsyncIterator[Union[str, Dict]]:
        """
        流式改写用户query
        
        先流式输出LLM的原始输出（文本和JSON），然后输出解析后的结果
        
        Args:
            sample_rows: 样本数据行，格式为 [(columns, data_rows)] 或直接的行列表
        
        Yields:
            str: 流式输出的原始文本chunk
            Dict: 最终解析后的改写结果（当流式输出完成时）
        """
        logger.info(f"Query改写（流式） - 原始问题: {user_query}")

        # 尝试使用LLM改写
        if self.llm_client:
            try:
                logger.info("尝试使用LLM进行Query改写（流式）...")
                full_text = ""
                async for chunk in self._llm_based_rewrite_stream(
                    user_query,
                    table_schema_json,
                    table_description,
                    chat_history=chat_history,
                    sample_rows=sample_rows,
                ):
                    if isinstance(chunk, str):
                        # 流式输出原始文本chunk
                        full_text = chunk
                        yield chunk
                    elif isinstance(chunk, dict):
                        # 返回最终解析结果
                        logger.info(f"✅ LLM改写成功（流式） - 改写结果: {chunk.get('rewritten_query', '')}")
                        yield chunk
                        return
            except JSONParseError as e:
                logger.error(f"❌ LLM改写失败（JSON解析错误）: {e}")
                logger.warning("⚠️ 使用规则改写作为fallback")
            except Exception as e:
                logger.warning(f"⚠️ LLM改写失败: {e}，使用规则改写作为fallback")
        else:
            logger.info("LLM客户端未配置，使用规则改写")

        # Fallback: 基于规则的简单改写
        result = self._rule_based_rewrite(user_query, table_schema_json)
        logger.info(f"规则改写 - 改写结果: {result['rewritten_query']}")
        yield result

    
    
    def _rule_based_rewrite(self, user_query: str, table_schema_json: str) -> Dict:
        """
        基于规则的Query改写（不依赖LLM，快速可靠）
        """
        try:
            # 规则改写不进行相关性判断，统一返回 is_relevant=True
            # 相关性判断交给LLM处理（更准确）
            
            # 解析schema JSON
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            # 获取列信息
            schema_data = (
                schema_obj.get("columns", []) if isinstance(schema_obj, dict) else []
            )

            relevant_columns = []
            analysis_suggestions = []

            # 分析用户query中提到的关键词
            query_lower = user_query.lower()

            # 检测时间相关的分析
            if any(
                keyword in query_lower for keyword in ["同比", "环比", "yoy", "mom"]
            ):
                # 查找日期字段
                date_cols = [
                    col
                    for col in schema_data
                    if any(
                        kw in col.get("column_name", "").lower()
                        for kw in ["date", "日期", "time", "时间"]
                    )
                ]
                if date_cols:
                    relevant_columns.append(
                        {
                            "column_name": date_cols[0]["column_name"],
                            "usage": "时间筛选和分组条件，需要包含足够的历史数据",
                        }
                    )

                analysis_suggestions.append(
                    "同比分析需要去年同期数据，确保WHERE条件包含至少两年的数据"
                )
                analysis_suggestions.append(
                    "环比分析需要上一个周期数据，使用LAG窗口函数"
                )
                analysis_suggestions.append(
                    "时间范围示例：WHERE 订单日期 >= '2021-01-01' AND 订单日期 < '2023-01-01'"
                )

            # 检测地域分析
            if any(
                keyword in user_query
                for keyword in [
                    "区域",
                    "地区",
                    "华北",
                    "华东",
                    "华南",
                    "东北",
                    "西北",
                    "西南",
                    "中南",
                ]
            ):
                region_cols = [
                    col for col in schema_data if "区域" in col.get("column_name", "")
                ]
                if region_cols:
                    relevant_columns.append(
                        {
                            "column_name": region_cols[0]["column_name"],
                            "usage": "地域筛选或分组条件",
                        }
                    )
                    analysis_suggestions.append("使用'区域'字段进行筛选或分组")

            # 检测指标分析
            metric_keywords = {
                "利润": "profit",
                "销售额": "sales",
                "销量": "quantity",
                "数量": "quantity",
            }

            for cn_keyword, en_keyword in metric_keywords.items():
                if cn_keyword in user_query:
                    metric_cols = [
                        col
                        for col in schema_data
                        if cn_keyword in col.get("column_name", "")
                    ]
                    if metric_cols:
                        relevant_columns.append(
                            {
                                "column_name": metric_cols[0]["column_name"],
                                "usage": "聚合指标，需要进行SUM/AVG等聚合运算",
                            }
                        )
                        analysis_suggestions.append(
                            f"对'{cn_keyword}'字段进行聚合计算（SUM求和）"
                        )

            # 构建改写后的query
            rewritten_query = self._enhance_query(
                user_query, relevant_columns, analysis_suggestions
            )

            # 构建分析逻辑
            analysis_logic = self._build_analysis_logic(user_query, relevant_columns)

            return {
                "original_query": user_query,
                "is_relevant": True,
                "rewritten_query": rewritten_query,
                "relevant_columns": relevant_columns,
                "analysis_suggestions": analysis_suggestions,
                "analysis_logic": analysis_logic,
            }

        except Exception as e:
            logger.error(f"规则改写失败: {e}", exc_info=True)
            return self._default_result(user_query)

    def _enhance_query(
        self, user_query: str, relevant_columns: List[Dict], suggestions: List[str]
    ) -> str:
        """
        增强用户query
        """
        if not relevant_columns:
            return user_query

        # 添加明确的字段引用
        col_names = [col["column_name"] for col in relevant_columns]
        enhanced = user_query

        # 如果是简短的query，添加更多上下文
        if len(user_query) < 20:
            enhanced = f"基于数据表，使用字段【{', '.join(col_names)}】来{user_query}"

        return enhanced

    def _build_analysis_logic(
        self, user_query: str, relevant_columns: List[Dict]
    ) -> str:
        """
        构建分析逻辑
        """
        logic_parts = []

        # 筛选条件
        filter_cols = [
            col for col in relevant_columns if "筛选" in col.get("usage", "")
        ]
        if filter_cols:
            logic_parts.append(
                f"1. 筛选条件：{', '.join([c['column_name'] for c in filter_cols])}"
            )

        # 分组维度
        group_cols = [col for col in relevant_columns if "分组" in col.get("usage", "")]
        if group_cols:
            logic_parts.append(
                f"2. 分组维度：{', '.join([c['column_name'] for c in group_cols])}"
            )

        # 聚合指标
        agg_cols = [col for col in relevant_columns if "聚合" in col.get("usage", "")]
        if agg_cols:
            logic_parts.append(
                f"3. 聚合指标：{', '.join([c['column_name'] for c in agg_cols])}"
            )

        if logic_parts:
            return "\n".join(logic_parts)
        else:
            return "基于用户问题进行标准的数据查询和分析"

    def _format_sample_rows(self, sample_rows: list) -> str:
        """
        格式化从外部传入的样本数据（列名只显示一次）
        
        Args:
            sample_rows: 格式为 (columns, data_rows) 的元组，
                        其中 columns 是列名列表，data_rows 是数据行列表
        """
        try:
            if not sample_rows:
                return ""
            
            # 检查是否是 (columns, data_rows) 格式
            if isinstance(sample_rows, tuple) and len(sample_rows) == 2:
                columns, data_rows = sample_rows
                if not data_rows or not columns:
                    return ""
                
                # 先显示列名
                lines = [f"列名: {json.dumps(list(columns), ensure_ascii=False)}"]
                
                # 只取前2行，只显示值
                for i, row in enumerate(data_rows[:2], 1):
                    values = []
                    for v in row:
                        if v is None or (isinstance(v, float) and str(v) == 'nan'):
                            values.append(None)
                        elif hasattr(v, 'strftime'):
                            values.append(v.strftime("%Y-%m-%d"))
                        elif isinstance(v, (int, float, bool)):
                            values.append(v)
                        else:
                            values.append(str(v))
                    lines.append(f"行{i}: {json.dumps(values, ensure_ascii=False)}")
                return "\n".join(lines)
            
            # 如果是其他格式（直接的行列表，每行是dict），提取共用列名
            if isinstance(sample_rows, list) and len(sample_rows) > 0:
                first_row = sample_rows[0]
                if isinstance(first_row, dict):
                    columns = list(first_row.keys())
                    lines = [f"列名: {json.dumps(columns, ensure_ascii=False)}"]
                    for i, row in enumerate(sample_rows[:2], 1):
                        values = [row.get(col) for col in columns]
                        lines.append(f"行{i}: {json.dumps(values, ensure_ascii=False)}")
                    return "\n".join(lines)
                else:
                    # 非dict格式，直接输出
                    lines = []
                    for i, row in enumerate(sample_rows[:2], 1):
                        lines.append(f"行{i}: {str(row)}")
                    return "\n".join(lines)
            
            return ""
        except Exception as e:
            logger.warning(f"格式化样本数据失败: {e}")
            return ""

    def _extract_sample_rows(self, table_schema_json: str) -> str:
        """
        从schema中提取样本数据并格式化为字符串（列名只显示一次）
        """
        try:
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            sample_rows = schema_obj.get("sample_rows", [])
            if not sample_rows:
                return ""

            # 提取列名（从第一行的keys）
            first_row = sample_rows[0]
            if isinstance(first_row, dict):
                columns = list(first_row.keys())
                lines = [f"列名: {json.dumps(columns, ensure_ascii=False)}"]
                for i, row in enumerate(sample_rows[:2], 1):
                    values = [row.get(col) for col in columns]
                    lines.append(f"行{i}: {json.dumps(values, ensure_ascii=False)}")
                return "\n".join(lines)
            else:
                # 非dict格式
                lines = []
                for i, row in enumerate(sample_rows[:2], 1):
                    lines.append(f"行{i}: {json.dumps(row, ensure_ascii=False)}")
                return "\n".join(lines)
        except Exception as e:
            logger.warning(f"提取样本数据失败: {e}")
            return ""

    def _build_rewrite_prompt(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
        sample_rows: list = None,
    ) -> str:
        """
        构建改写prompt
        
        Args:
            sample_rows: 样本数据，格式为 (columns, data_rows) 或直接从参数传入
        """
        # 精简schema，移除建议问题等不必要信息
        simplified_schema = self._simplify_schema_for_rewrite(table_schema_json)
        
        # 提取样本数据：优先使用传入的 sample_rows，其次从 schema 中提取
        sample_rows_str = ""
        if sample_rows:
            sample_rows_str = self._format_sample_rows(sample_rows)
        if not sample_rows_str:
            sample_rows_str = self._extract_sample_rows(table_schema_json)
        
        # 构建历史对话上下文
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n=== 历史对话上下文 ===\n"
            # 只保留最近3轮对话（6条消息），避免prompt过长
            recent_history = (
                chat_history[-6:] if len(chat_history) > 6 else chat_history
            )

            for msg in recent_history:
                role = msg.get("role", "user") if isinstance(msg, dict) else "user"
                content = (
                    str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
                )

                # 根据实际角色显示
                if "human" in role.lower() or "user" in role.lower():
                    role_display = "用户"
                elif "ai" in role.lower() or "assistant" in role.lower():
                    role_display = "助手"
                else:
                    role_display = role

                history_context += f"\n{role_display}: {content}\n"

        # 检测用户输入语言
        user_language = detect_language(user_query)
        logger.info(f"🌐 Query改写 - 检测到用户输入语言: {user_language}")

        # 根据语言选择prompt
        if user_language == "zh":
            prompt = f"""你是一个数据分析专家。用户提出了一个数据分析问题，你需要：
1. 根据用户历史问题和回答，充分理解用户的真实意图，补充改写当前问题
2. 根据数据表的字段信息，补充完善用户的问题
3. 明确指出可能用到的列（包括筛选条件列、分组维度列、聚合指标列）
4. 提供3-5条分析建议，说明如何分析这个问题
5. 给出清晰的分析逻辑
6. 如果用户在对话或当前历史问答上下文中纠正或补充了字段的使用方法、业务规则、数据处理技巧等关键知识，请提取并记录作为domain_knowledge字段

=== 数据表字段信息 ===
{simplified_schema}

**注意**：字段信息中可能包含 `domain_knowledge` 字段，这是之前从用户对话中学习到的业务知识，请优先参考使用。

=== 真实数据样本（2行） ===
{sample_rows_str if sample_rows_str else "暂无样本数据"}

=== 数据表描述 ===
{table_description}

{history_context}
=== 用户当前问题 ===
{user_query}

=== 输出格式（JSON） ===
请严格按照以下JSON格式输出：
{{
  "is_relevant": true,  // 布尔值，表示用户问题是否与数据表分析相关。如果是闲聊（如"今天天气怎么样"、"你吃饭了吗"）则为false
  "rewritten_query": "改写后的完整问题，明确指出需要分析的维度和指标",
  "relevant_columns": [
    {{
      "column_name": "列名",
      "usage": "用途说明（如：筛选条件/分组维度/聚合指标）"
    }}
  ],
  "analysis_suggestions": [
    "建议1：具体的分析步骤或注意事项",
    "建议2：...",
    "建议3：..."
    ...
  ],
  "analysis_logic": "分析逻辑的详细说明，包括：1) 需要筛选哪些数据 2) 按什么维度分组 3) 计算哪些指标 4) 如何排序或对比",
  "domain_knowledge": {{
    "column_name": "字段名（如果用户纠正或补充了某个字段的使用方法）",
    "knowledge": "用户补充的业务知识或数据处理技巧（例如：'该字段格式为H1,H2，逗号前是H1绩效，逗号后是H2绩效，需要用SPLIT_PART函数分割'）"
  }}
}}


**关于 is_relevant 的说明**：
- 判断用户问题是否与当前数据表的分析相关
- 如果是数据分析问题（如"销售额是多少"、"利润排名"、"同比增长"等），设为 true
- 如果是闲聊或与数据表无关的问题（如"今天天气怎么样"、"你吃饭了吗"、"讲个笑话"等），设为 false
- 当 is_relevant 为 false 时，其他字段可以简化或省略

**关于 domain_knowledge 的说明**：
- 只有当用户明确纠正、补充或说明了某个字段的使用方法时才需要填写
- 如果用户只是普通提问，不需要填写此字段（可以省略或设为 null）
- 知识应该是可复用的、对未来分析有帮助的关键信息,比如业务规则、数据处理技巧等,这部分知识会作为领域知识保存到数据库中,用于后续的分析和推理,可以复用这些知识来回答用户的问题，如果上面已经记录了领域知识，请不要重复记录。

**重要 - 语言要求**：
- 用户的问题是**中文**
- 你必须用**中文**回复JSON中的所有字段
- "rewritten_query"、"usage"、"analysis_suggestions"、"analysis_logic" 等字段必须使用**中文**
- 即使表字段名是中英文混合，你的描述和分析也必须使用**中文**

**重要 - 字符串/字段精确匹配要求**：
- 在改写问题时，如果用户提到了具体的部门名称、分类值等字符串，必须保持完全一致
- 如果用户问题中包含具体的字符串值，在"rewritten_query"中必须保持原样，不能修改

**严格要求 - 字段名必须来自提供的列表**：
- relevant_columns 中的 column_name 必须**严格从上面提供的字段信息中选择**
- **禁止推测、创造或编造任何不在字段列表中的字段名**
- 如果找不到完全匹配的字段，宁可不填写该字段，也不要编造类似的字段名

现在请结合历史上下文及用户当前问题，分析用户的真实意图，补充改写当前问题并用中文输出JSON：
"""
        else:
            prompt = f"""You are a data analysis expert. The user has asked a data analysis question. You need to:
1. Understand the user's real intent based on historical questions and answers, and enhance the current question
2. Enhance the user's question based on the data table field information
3. Clearly identify the columns that may be used (including filter condition columns, grouping dimension columns, and aggregation indicator columns)
4. Provide 3-5 analysis suggestions explaining how to analyze this question
5. Give a clear analysis logic
6. If the user has corrected or supplemented field usage methods, business rules, data processing techniques, or other key knowledge in the conversation or current historical Q&A context, extract and record it as the domain_knowledge field

=== Data Table Field Information ===
{simplified_schema}

**Note**: The field information may contain a `domain_knowledge` field, which is business knowledge learned from previous user conversations. Please prioritize using this.

=== Sample Data (2 rows) ===
{sample_rows_str if sample_rows_str else "No sample data available"}

=== Data Table Description ===
{table_description}

{history_context}
=== User's Current Question ===
{user_query}

=== Output Format (JSON) ===
Please strictly follow the following JSON format:
{{
  "is_relevant": true,  // Boolean value indicating whether the user's question is related to data table analysis. If it's small talk (e.g., "How's the weather today", "Did you eat"), set to false
  "rewritten_query": "The enhanced complete question, clearly indicating the dimensions and indicators to be analyzed",
  "relevant_columns": [
    {{
      "column_name": "Column name",
      "usage": "Usage description (e.g., filter condition/grouping dimension/aggregation indicator)"
    }}
  ],
  "analysis_suggestions": [
    "Suggestion 1: Specific analysis steps or considerations",
    "Suggestion 2: ...",
    "Suggestion 3: ..."
    ...
  ],
  "analysis_logic": "Detailed explanation of analysis logic, including: 1) Which data to filter 2) What dimension to group by 3) Which indicators to calculate 4) How to sort or compare",
  "domain_knowledge": {{
    "column_name": "Field name (if the user has corrected or supplemented the usage method of a field)",
    "knowledge": "Business knowledge or data processing techniques supplemented by the user (e.g., 'This field format is H1,H2, before the comma is H1 performance, after the comma is H2 performance, need to use SPLIT_PART function to split')"
  }}
}}



**About is_relevant**:
- Determine whether the user's question is related to the analysis of the current data table
- If it's a data analysis question (e.g., "What is the sales amount", "Profit ranking", "Year-over-year growth"), set to true
- If it's small talk or unrelated to the data table (e.g., "How's the weather today", "Did you eat", "Tell me a joke"), set to false
- When is_relevant is false, other fields can be simplified or omitted

**About domain_knowledge**:
- Only fill in when the user explicitly corrects, supplements, or explains the usage method of a field
- If the user is just asking a normal question, this field is not required (can be omitted or set to null)
- Knowledge should be reusable and helpful for future analysis, such as business rules, data processing techniques, etc. This knowledge will be saved to the database as domain knowledge for subsequent analysis and reasoning, and can be reused to answer user questions. If domain knowledge has already been recorded above, please do not repeat it.

**IMPORTANT - Language Requirement**:
- The user's question is in ENGLISH
- You MUST respond in ENGLISH for ALL fields in the JSON output
- The "rewritten_query", "usage", "analysis_suggestions", and "analysis_logic" fields MUST be in ENGLISH
- Even though the table field names are in Chinese, your descriptions and analysis MUST be in ENGLISH

**IMPORTANT - String Exact Matching Requirement**:
- When rewriting the question, if the user mentions specific department names, category values, or other strings, they must be kept exactly as they are
- If the user's question contains specific string values, they must be kept unchanged in "rewritten_query"

**STRICT REQUIREMENT - Column names must come from the provided list**:
- The column_name in relevant_columns MUST be **strictly selected from the field information provided above**
- **DO NOT guess, create, or fabricate any column names that are not in the field list**
- If you cannot find an exact match, leave it out rather than inventing a similar column name

Now please combine the historical context and the user's current question, analyze the user's real intent, enhance the current question and output JSON IN ENGLISH:
"""
        return prompt

    async def _llm_based_rewrite_stream(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
        sample_rows: list = None,
        invalid_columns_hint: list = None,
        retry_count: int = 0,
    ) -> AsyncIterator[Union[str, Dict]]:
        """
        使用LLM进行流式Query改写
        
        Args:
            sample_rows: 样本数据行
            invalid_columns_hint: 上次返回的无效字段名列表，用于提示LLM重新生成
            retry_count: 当前重试次数
        
        Yields:
            str: 流式输出的原始文本chunk（累积的完整文本）
            Dict: 最终解析后的改写结果
        """
        import inspect

        from dbgpt.core import (
            ModelMessage,
            ModelMessageRoleType,
            ModelRequest,
            ModelRequestContext,
        )

        MAX_RETRY = 2  # 最多重试2次

        # 提取有效字段名用于校验
        valid_column_names = self._extract_valid_column_names(table_schema_json)
        logger.debug(f"有效字段名数量: {len(valid_column_names)}")

        # 检测用户语言
        user_language = detect_language(user_query)

        # 构建prompt
        prompt = self._build_rewrite_prompt(
            user_query, table_schema_json, table_description, 
            chat_history=chat_history, sample_rows=sample_rows
        )
        
        # 如果有无效字段提示，追加到prompt中（根据语言选择）
        if invalid_columns_hint:
            if user_language == "zh":
                correction_hint = f"""

**错误纠正**：
你上次返回的以下字段名在数据表中不存在，请勿使用：
{invalid_columns_hint}

请仔细检查上面提供的字段列表，只使用实际存在的字段名重新生成。如果找不到对应的字段，请在analysis_suggestions中说明该分析可能无法完成。
"""
            else:
                correction_hint = f"""

**Error Correction**:
The following column names you returned do not exist in the data table, DO NOT use them:
{invalid_columns_hint}

Please carefully check the field list provided above and regenerate using only column names that actually exist. If you cannot find the corresponding field, please explain in analysis_suggestions that this analysis may not be possible.
"""
            prompt += correction_hint
            logger.info(f"🔄 重试第{retry_count}次，添加无效字段提示: {invalid_columns_hint}")
        
        logger.debug(f"🔍 query_rewrite_agent prompt: {prompt[:200]}...")
        
        # 调用LLM（流式）
        request_params = {
            "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            "max_new_tokens": 2000,
            "context": ModelRequestContext(stream=True),
        }

        # 如果有model_name，添加到请求中
        if self.model_name:
            request_params["model"] = self.model_name

        request = ModelRequest(**request_params)

        # 获取流式响应
        stream_response = self.llm_client.generate_stream(request)

        full_text = ""
        if inspect.isasyncgen(stream_response):
            async for chunk in stream_response:
                # 安全地获取文本内容
                try:
                    chunk_text = ""
                    if hasattr(chunk, "has_text") and chunk.has_text:
                        chunk_text = chunk.text
                    elif hasattr(chunk, "text"):
                        try:
                            chunk_text = chunk.text
                        except ValueError:
                            # 可能只有 thinking 内容，继续等待 text 内容
                            continue
                    
                    if chunk_text:
                        full_text = chunk_text
                        # 流式输出累积的完整文本
                        yield full_text
                except Exception as e:
                    logger.debug(f"获取chunk.text失败: {e}")
                    continue
        elif inspect.isgenerator(stream_response):
            for chunk in stream_response:
                try:
                    chunk_text = ""
                    if hasattr(chunk, "has_text") and chunk.has_text:
                        chunk_text = chunk.text
                    elif hasattr(chunk, "text"):
                        try:
                            chunk_text = chunk.text
                        except ValueError:
                            continue
                    
                    if chunk_text:
                        full_text = chunk_text
                        # 流式输出累积的完整文本
                        yield full_text
                except Exception as e:
                    logger.debug(f"获取chunk.text失败: {e}")
                    continue
        else:
            raise Exception(f"Unexpected response type: {type(stream_response)}")

        # 流式输出完成后，解析结果并返回（带字段名校验）
        if full_text:
            try:
                result = self._parse_rewrite_result(
                    full_text, user_query, valid_column_names=valid_column_names
                )
                yield result
            except InvalidColumnError as e:
                # 字段名无效，尝试重试
                if retry_count < MAX_RETRY:
                    logger.warning(f"⚠️ 检测到无效字段名，触发重试 ({retry_count + 1}/{MAX_RETRY})")
                    # 递归调用，传入无效字段提示
                    async for chunk in self._llm_based_rewrite_stream(
                        user_query,
                        table_schema_json,
                        table_description,
                        chat_history=chat_history,
                        sample_rows=sample_rows,
                        invalid_columns_hint=e.invalid_columns,
                        retry_count=retry_count + 1,
                    ):
                        yield chunk
                else:
                    logger.error(f"❌ 重试{MAX_RETRY}次后仍有无效字段，放弃重试")
                    raise JSONParseError(str(e))
            except JSONParseError as e:
                logger.error(f"JSON解析失败: {e}")
                raise

    def _parse_rewrite_result(
        self, llm_output: str, original_query: str, valid_column_names: set = None
    ) -> Dict:
        """
        解析LLM输出的JSON结果

        如果解析失败或字段名不存在，抛出 JSONParseError 异常以触发重试机制
        
        Args:
            llm_output: LLM输出的文本
            original_query: 原始用户问题
            valid_column_names: 有效的字段名集合，用于校验
        """
        try:
            # 提取JSON部分
            start_idx = llm_output.find("{")
            end_idx = llm_output.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_output[start_idx:end_idx]
                result = json.loads(json_str)

                # 验证必要字段是否存在
                if "is_relevant" not in result:
                    logger.warning("JSON缺少 'is_relevant' 字段，默认设为 true")
                    result["is_relevant"] = True
                
                if not result.get("rewritten_query"):
                    logger.error("JSON缺少必要字段 'rewritten_query'")
                    raise JSONParseError("JSON缺少必要字段 'rewritten_query'")

                # 验证 relevant_columns 格式和字段名是否存在
                relevant_columns = result.get("relevant_columns", [])
                if relevant_columns:
                    invalid_columns = []
                    for idx, col in enumerate(relevant_columns):
                        if not isinstance(col, dict) or "column_name" not in col:
                            logger.error(f"relevant_columns[{idx}] 格式错误: {col}")
                            raise JSONParseError(
                                f"relevant_columns[{idx}] 缺少 'column_name' 字段"
                            )
                        
                        # 校验字段名是否存在于表中
                        col_name = col.get("column_name", "")
                        if valid_column_names and col_name not in valid_column_names:
                            invalid_columns.append(col_name)
                    
                    # 如果有无效字段名，抛出 InvalidColumnError 触发重试
                    if invalid_columns:
                        error_msg = f"以下字段名不存在于数据表中: {invalid_columns}"
                        logger.warning(f"⚠️ 字段校验失败: {error_msg}，将触发重试")
                        raise InvalidColumnError(error_msg, invalid_columns)

                # 添加原始问题
                result["original_query"] = original_query

                # 提取领域知识（如果有）
                domain_knowledge = result.get("domain_knowledge")
                if domain_knowledge and isinstance(domain_knowledge, dict):
                    column_name = domain_knowledge.get("column_name")
                    knowledge = domain_knowledge.get("knowledge")
                    if column_name and knowledge:
                        result["_extracted_knowledge"] = {
                            "column_name": column_name,
                            "knowledge": knowledge,
                        }

                logger.info("✅ JSON解析成功，字段校验通过")
                return result
            else:
                logger.error("无法从LLM输出中提取JSON")
                raise JSONParseError("无法从LLM输出中提取JSON内容")

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            raise JSONParseError(f"JSON解析失败: {e}")

    def _default_result(self, original_query: str) -> Dict:
        """
        返回默认结果（当LLM失败时）
        """
        return {
            "original_query": original_query,
            "is_relevant": True,
            "rewritten_query": original_query,
            "relevant_columns": [],
            "analysis_suggestions": ["请明确需要分析的数据维度和指标"],
            "analysis_logic": "基于用户问题进行数据分析",
        }
