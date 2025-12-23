"""
Query改写Agent - 参考format_sql/backend/agents/query_rewrite_assistant.py
负责根据数据理解信息，补充完善用户问题，明确相关列和分析建议
"""

import json
import logging
from typing import Dict, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """自定义异常：JSON解析失败"""

    pass


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

    def rewrite_query(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
    ) -> Dict[str, any]:
        """
        改写用户query

        优先使用LLM改写（带重试机制），如果LLM不可用或重试3次后仍失败则使用规则改写
        """
        logger.info(f"Query改写 - 原始问题: {user_query}")

        # 尝试使用LLM改写
        if self.llm_client:
            try:
                logger.info("尝试使用LLM进行Query改写（带重试机制，最多3次）...")
                result = self._llm_based_rewrite(
                    user_query,
                    table_schema_json,
                    table_description,
                    chat_history=chat_history,
                )
                logger.info(f"✅ LLM改写成功 - 改写结果: {result['rewritten_query']}")
                return result
            except JSONParseError as e:
                logger.error(f"❌ LLM改写失败（JSON解析错误，已重试3次）: {e}")
                logger.warning("⚠️ 使用规则改写作为fallback")
            except Exception as e:
                logger.warning(f"⚠️ LLM改写失败: {e}，使用规则改写作为fallback")
        else:
            logger.info("LLM客户端未配置，使用规则改写")

        # Fallback: 基于规则的简单改写
        result = self._rule_based_rewrite(user_query, table_schema_json)
        logger.info(f"规则改写 - 改写结果: {result['rewritten_query']}")

        return result

    def _rule_based_rewrite(self, user_query: str, table_schema_json: str) -> Dict:
        """
        基于规则的Query改写（不依赖LLM，快速可靠）
        """
        try:
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
                    analysis_suggestions.append(f"使用'区域'字段进行筛选或分组")

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

    def _build_rewrite_prompt(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
    ) -> str:
        """
        构建改写prompt
        """
        # 构建历史对话上下文
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n=== 历史对话上下文 ===\n"
            # 只保留最近4轮对话（8条消息），避免prompt过长
            recent_history = (
                chat_history[-8:] if len(chat_history) > 8 else chat_history
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

=== 数据表字段详细信息 ===
{table_schema_json}

**注意**：字段信息中可能包含 `domain_knowledge` 字段，这是之前从用户对话中学习到的业务知识，请优先参考使用。

=== 数据表描述 ===
{table_description}

{history_context}
=== 用户当前问题 ===
{user_query}

=== 输出格式（JSON） ===
请严格按照以下JSON格式输出：
{{
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
  ],
  "analysis_logic": "分析逻辑的详细说明，包括：1) 需要筛选哪些数据 2) 按什么维度分组 3) 计算哪些指标 4) 如何排序或对比",
  "domain_knowledge": {{
    "column_name": "字段名（如果用户纠正或补充了某个字段的使用方法）",
    "knowledge": "用户补充的业务知识或数据处理技巧（例如：'该字段格式为H1,H2，逗号前是H1绩效，逗号后是H2绩效，需要用SPLIT_PART函数分割'）"
  }}
}}

**关于 domain_knowledge 的说明**：
- 只有当用户明确纠正、补充或说明了某个字段的使用方法时才需要填写
- 如果用户只是普通提问，不需要填写此字段（可以省略或设为 null）
- 知识应该是可复用的、对未来分析有帮助的关键信息,比如业务规则、数据处理技巧等,这部分知识会作为领域知识保存到数据库中,用于后续的分析和推理,可以复用这些知识来回答用户的问题，如果上面已经记录了领域知识，请不要重复记录。

**重要 - 语言要求**：
- 用户的问题是**中文**
- 你必须用**中文**回复JSON中的所有字段
- "rewritten_query"、"usage"、"analysis_suggestions"、"analysis_logic" 等字段必须使用**中文**
- 即使表字段名是中英文混合，你的描述和分析也必须使用**中文**

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

=== Data Table Field Details ===
{table_schema_json}

**Note**: The field information may contain a `domain_knowledge` field, which is business knowledge learned from previous user conversations. Please prioritize using this.

=== Data Table Description ===
{table_description}

{history_context}
=== User's Current Question ===
{user_query}

=== Output Format (JSON) ===
Please strictly follow the following JSON format:
{{
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
  ],
  "analysis_logic": "Detailed explanation of analysis logic, including: 1) Which data to filter 2) What dimension to group by 3) Which indicators to calculate 4) How to sort or compare",
  "domain_knowledge": {{
    "column_name": "Field name (if the user has corrected or supplemented the usage method of a field)",
    "knowledge": "Business knowledge or data processing techniques supplemented by the user (e.g., 'This field format is H1,H2, before the comma is H1 performance, after the comma is H2 performance, need to use SPLIT_PART function to split')"
  }}
}}

**About domain_knowledge**:
- Only fill in when the user explicitly corrects, supplements, or explains the usage method of a field
- If the user is just asking a normal question, this field is not required (can be omitted or set to null)
- Knowledge should be reusable and helpful for future analysis, such as business rules, data processing techniques, etc. This knowledge will be saved to the database as domain knowledge for subsequent analysis and reasoning, and can be reused to answer user questions. If domain knowledge has already been recorded above, please do not repeat it.

**IMPORTANT - Language Requirement**:
- The user's question is in ENGLISH
- You MUST respond in ENGLISH for ALL fields in the JSON output
- The "rewritten_query", "usage", "analysis_suggestions", and "analysis_logic" fields MUST be in ENGLISH
- Even though the table field names are in Chinese, your descriptions and analysis MUST be in ENGLISH

Now please combine the historical context and the user's current question, analyze the user's real intent, enhance the current question and output JSON IN ENGLISH:
"""
        return prompt

    @retry(
        retry=retry_if_exception_type(JSONParseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"🔄 JSON解析失败，第 {retry_state.attempt_number} 次重试..."
        ),
    )
    def _llm_based_rewrite(
        self,
        user_query: str,
        table_schema_json: str,
        table_description: str,
        chat_history: list = None,
    ) -> Dict:
        """
        使用LLM进行Query改写（带重试机制）

        如果JSON解析失败，会自动重试最多3次：
        - 第1次重试：等待1秒
        - 第2次重试：等待2秒
        - 第3次重试：等待4秒
        """
        import asyncio
        import inspect
        from dbgpt.core import (
            ModelRequest,
            ModelMessage,
            ModelMessageRoleType,
            ModelRequestContext,
        )

        # 构建prompt
        prompt = self._build_rewrite_prompt(
            user_query, table_schema_json, table_description, chat_history=chat_history
        )
        print(f"🔍 query_rewrite_agent prompt: {prompt}")
        # 调用LLM（非流式）
        request_params = {
            "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            "temperature": 0.1,
            "max_new_tokens": 2000,
            "context": ModelRequestContext(stream=False),
        }

        # 如果有model_name，添加到请求中
        if self.model_name:
            request_params["model"] = self.model_name

        request = ModelRequest(**request_params)

        # 获取响应
        stream_response = self.llm_client.generate_stream(request)

        full_text = ""
        if inspect.isasyncgen(stream_response):

            async def collect_async():
                text = ""
                async for chunk in stream_response:
                    # 安全地获取文本内容，避免 "The content type is not text" 错误
                    try:
                        if hasattr(chunk, "has_text") and chunk.has_text:
                            text = chunk.text
                        elif hasattr(chunk, "text"):
                            # 尝试获取 text，如果失败则跳过
                            try:
                                text = chunk.text
                            except ValueError:
                                # 可能只有 thinking 内容，继续等待 text 内容
                                pass
                    except Exception as e:
                        logger.debug(f"获取chunk.text失败: {e}")
                        pass
                return text

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                full_text = loop.run_until_complete(collect_async())
            finally:
                loop.close()
        elif inspect.isgenerator(stream_response):
            for chunk in stream_response:
                # 安全地获取文本内容，避免 "The content type is not text" 错误
                try:
                    if hasattr(chunk, "has_text") and chunk.has_text:
                        full_text = chunk.text
                    elif hasattr(chunk, "text"):
                        # 尝试获取 text，如果失败则跳过
                        try:
                            full_text = chunk.text
                        except ValueError:
                            # 可能只有 thinking 内容，继续等待 text 内容
                            pass
                except Exception as e:
                    logger.debug(f"获取chunk.text失败: {e}")
                    pass
        else:
            raise Exception(f"Unexpected response type: {type(stream_response)}")

        # 解析结果
        result = self._parse_rewrite_result(full_text, user_query)

        return result

    def _parse_rewrite_result(self, llm_output: str, original_query: str) -> Dict:
        """
        解析LLM输出的JSON结果

        如果解析失败，抛出 JSONParseError 异常以触发重试机制
        """
        try:
            # 提取JSON部分
            start_idx = llm_output.find("{")
            end_idx = llm_output.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_output[start_idx:end_idx]
                result = json.loads(json_str)

                # 验证必要字段是否存在
                if not result.get("rewritten_query"):
                    logger.error("JSON缺少必要字段 'rewritten_query'")
                    raise JSONParseError("JSON缺少必要字段 'rewritten_query'")

                # 验证 relevant_columns 格式
                relevant_columns = result.get("relevant_columns", [])
                if relevant_columns:
                    for idx, col in enumerate(relevant_columns):
                        if not isinstance(col, dict) or "column_name" not in col:
                            logger.error(f"relevant_columns[{idx}] 格式错误: {col}")
                            raise JSONParseError(
                                f"relevant_columns[{idx}] 缺少 'column_name' 字段"
                            )

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

                logger.info("✅ JSON解析成功")
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
            "rewritten_query": original_query,
            "relevant_columns": [],
            "analysis_suggestions": ["请明确需要分析的数据维度和指标"],
            "analysis_logic": "基于用户问题进行数据分析",
        }
