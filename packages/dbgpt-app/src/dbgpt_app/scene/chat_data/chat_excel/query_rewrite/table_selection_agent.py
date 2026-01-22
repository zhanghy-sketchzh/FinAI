"""
表选择Agent - 在多表模式下，根据用户问题选择相关的表
"""
# ruff: noqa: E501

import json
import logging
from typing import AsyncIterator, Dict, List, Union

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """检测文本的主要语言"""
    if not text or not text.strip():
        return "en"

    import re

    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    chinese_chars = chinese_pattern.findall(text)
    chinese_count = sum(len(match) for match in chinese_chars)

    total_chars = len(re.sub(r'[\s\.,;:!?\'"\-_()\[\]{}]', "", text))

    if total_chars == 0:
        return "en"

    chinese_ratio = chinese_count / total_chars if total_chars > 0 else 0
    return "zh" if chinese_ratio > 0.3 else "en"


class TableSelectionAgent:
    """
    表选择Agent
    功能：
    1. 根据用户问题和各表的建表信息，判断需要使用哪些表
    2. 支持单表和多表选择
    3. 提供选择理由
    """

    def __init__(self, llm_client=None, model_name=None):
        """
        Args:
            llm_client: LLM客户端
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name

    async def select_tables_stream(
        self,
        user_query: str,
        tables_info: List[Dict],
        chat_history: list = None,
    ) -> AsyncIterator[Union[str, Dict]]:
        """
        流式选择相关表
        
        Args:
            user_query: 用户问题
            tables_info: 表信息列表，每个元素包含：
                - table_name: 表名
                - sheet_name: sheet名称
                - create_table_sql: 建表SQL
                - data_schema_json: 数据schema JSON（可选）
                - row_count: 行数
                - column_count: 列数
            chat_history: 聊天历史
            
        Yields:
            str: 流式输出的原始文本
            Dict: 最终选择结果，格式：
                {
                    "selected_tables": ["table1", "table2"],
                    "selection_reason": "选择理由"
                }
        """
        logger.info(f"表选择 - 用户问题: {user_query}, 可选表数量: {len(tables_info)}")

        # 如果只有一个表，直接返回
        if len(tables_info) == 1:
            result = {
                "selected_tables": [tables_info[0]["table_name"]],
                "selection_reason": "只有一个可用表",
            }
            yield result
            return

        # 使用LLM选择表
        if self.llm_client:
            try:
                async for chunk in self._llm_based_selection_stream(
                    user_query, tables_info, chat_history
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"LLM表选择失败: {e}，使用规则选择")

        # Fallback: 基于规则的选择
        result = self._rule_based_selection(user_query, tables_info)
        yield result

    def _rule_based_selection(
        self, user_query: str, tables_info: List[Dict]
    ) -> Dict:
        """
        基于规则的表选择（不依赖LLM）
        优先选择多个可能相关的表，宁可多选不要漏选
        """
        import re
        
        query_lower = user_query.lower()
        selected_tables = []
        selection_scores = {}  # 记录每个表的相关性分数
        
        # 提取问题中的关键词（过滤掉停用词）
        stop_words = {"的", "是", "在", "有", "和", "与", "或", "等", "中", "了", "吗", "呢", "啊", 
                      "the", "is", "in", "has", "and", "or", "with", "of", "to", "a", "an"}
        query_keywords = [w for w in re.findall(r'\w+', query_lower) if len(w) > 1 and w not in stop_words]
        
        for table in tables_info:
            table_name = table.get("table_name", "")
            sheet_name = table.get("sheet_name", "")
            create_sql = table.get("create_table_sql", "")
            score = 0
            
            # 1. 检查表名或sheet名是否在问题中被明确提及（高优先级）
            if table_name.lower() in query_lower or sheet_name.lower() in query_lower:
                score += 100
            
            # 2. 检查建表SQL中的列名是否与问题关键词匹配
            if create_sql:
                col_matches = re.findall(r'"([^"]+)"', create_sql)
                for col in col_matches:
                    col_lower = col.lower()
                    # 精确匹配列名
                    if col_lower in query_keywords:
                        score += 10
                    # 部分匹配（列名包含关键词或关键词包含列名）
                    for keyword in query_keywords:
                        if keyword in col_lower or col_lower in keyword:
                            score += 5
                            break
            
            # 3. 检查表描述是否相关
            table_description = ""
            if table.get("data_schema_json"):
                try:
                    schema = json.loads(table["data_schema_json"]) if isinstance(
                        table["data_schema_json"], str
                    ) else table["data_schema_json"]
                    table_description = schema.get("table_description", "")
                except Exception:
                    pass
            
            if table_description:
                desc_lower = table_description.lower()
                for keyword in query_keywords:
                    if keyword in desc_lower:
                        score += 3
            
            selection_scores[table_name] = score
            
            # 如果分数大于0，说明有一定相关性，加入候选
            if score > 0:
                selected_tables.append(table_name)

        # 如果没有匹配到任何表，返回所有表（宁可多选不要漏选）
        if not selected_tables:
            selected_tables = [t["table_name"] for t in tables_info]
            reason = "无法确定具体表，返回所有可用表以避免遗漏"
        else:
            # 按分数排序，显示选择理由
            sorted_tables = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)
            top_tables = [t for t, s in sorted_tables if s > 0]
            reason = f"根据问题关键词匹配选择了{len(selected_tables)}个相关表：{', '.join(top_tables)}"

        return {
            "selected_tables": list(set(selected_tables)),
            "selection_reason": reason,
        }

    def _build_selection_prompt(
        self,
        user_query: str,
        tables_info: List[Dict],
        chat_history: list = None,
    ) -> str:
        """构建表选择prompt"""
        
        # 构建表信息描述
        tables_desc = []
        for i, table in enumerate(tables_info, 1):
            table_name = table.get("table_name", f"table_{i}")
            sheet_name = table.get("sheet_name", "")
            create_sql = table.get("create_table_sql", "")
            row_count = table.get("row_count", 0)
            column_count = table.get("column_count", 0)
            
            # 尝试从schema中获取表描述和字段信息
            table_description = ""
            columns_info = []
            if table.get("data_schema_json"):
                try:
                    schema = json.loads(table["data_schema_json"]) if isinstance(
                        table["data_schema_json"], str
                    ) else table["data_schema_json"]
                    table_description = schema.get("table_description", "")
                    
                    # 提取字段及其唯一值信息
                    columns = schema.get("columns", [])
                    for col in columns:
                        col_name = col.get("column_name", "")
                        col_desc = col.get("description", "")
                        unique_values = col.get("unique_values_top20", [])
                        
                        if col_name:
                            col_info = f"    • {col_name}"
                            if col_desc:
                                col_info += f": {col_desc}"
                            
                            # 添加示例值（最多显示前10个）
                            if unique_values and isinstance(unique_values, list):
                                sample_values = unique_values[:10]
                                if len(sample_values) > 0:
                                    values_str = ", ".join(str(v) for v in sample_values)
                                    if len(unique_values) > 10:
                                        col_info += f" (示例值: {values_str}, ...)"
                                    else:
                                        col_info += f" (可选值: {values_str})"
                            
                            columns_info.append(col_info)
                except Exception as e:
                    logger.warning(f"解析表 {table_name} 的schema失败: {e}")
            
            # 构建表信息描述
            desc = f"""
### 表{i}: {table_name}
- Sheet名称: {sheet_name}
- 行数: {row_count}, 列数: {column_count}
- 表描述: {table_description if table_description else "无"}
- 建表SQL:
```sql
{create_sql}
```"""
            
            # 如果有字段详细信息，添加到描述中
            if columns_info:
                desc += f"\n- 字段详情（含示例值）:\n"
                desc += "\n".join(columns_info)
            
            tables_desc.append(desc)

        tables_info_str = "\n".join(tables_desc)

        # 构建历史对话上下文
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n=== 历史对话上下文 ===\n"
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                role = msg.get("role", "user") if isinstance(msg, dict) else "user"
                content = str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
                role_display = "用户" if "human" in role.lower() or "user" in role.lower() else "助手"
                history_context += f"\n{role_display}: {content}\n"

        # 检测用户语言
        user_language = detect_language(user_query)

        if user_language == "zh":
            prompt = f"""你是一个数据分析专家。用户提出了一个数据分析问题，你需要根据问题内容，从多个可用的数据表中选择最相关的表。

=== 可用的数据表 ===
{tables_info_str}

{history_context}

=== 用户问题 ===
{user_query}

=== 任务要求 ===
1. 分析用户问题，理解用户想要分析什么数据
2. 根据各表的建表SQL和描述，判断哪些表包含用户需要的数据
3. 选择一个或多个最相关的表

=== 输出格式（JSON） ===
请严格按照以下JSON格式输出：
{{
  "selected_tables": ["表名1", "表名2"],
  "selection_reason": "选择这些表的理由，说明为什么这些表与用户问题相关"
}}

**重要提示**：
- selected_tables 中的表名必须与上面提供的表名完全一致
- 如果用户问题明确提到了某个表名或关键词，优先选择包含该关键词的表
- 如果用户问题涉及"所有"、"全部"等词，可能需要选择多个表
- 如果无法确定，可以选择多个可能相关的表
- 尽量精确选择，避免选择不相关的表

请输出JSON：
"""
        else:
            prompt = f"""You are a data analysis expert. The user has asked a data analysis question, and you need to select the most relevant tables from multiple available data tables based on the question content.

=== Available Data Tables ===
{tables_info_str}

{history_context}

=== User Question ===
{user_query}

=== Task Requirements ===
1. Analyze the user's question to understand what data they want to analyze
2. Based on the CREATE TABLE SQL and descriptions of each table, determine which tables contain the data the user needs
3. Select one or more most relevant tables

=== Output Format (JSON) ===
Please strictly follow the following JSON format:
{{
  "selected_tables": ["table_name1", "table_name2"],
  "selection_reason": "The reason for selecting these tables, explaining why they are relevant to the user's question"
}}

**Important Notes**:
- The table names in selected_tables must exactly match the table names provided above
- If the user's question explicitly mentions a table name or keyword, prioritize tables containing that keyword
- If the user's question involves "all", "total", etc., multiple tables may need to be selected
- If uncertain, you can select multiple potentially relevant tables
- Try to be precise and avoid selecting irrelevant tables

Please output JSON:
"""
        return prompt

    async def _llm_based_selection_stream(
        self,
        user_query: str,
        tables_info: List[Dict],
        chat_history: list = None,
    ) -> AsyncIterator[Union[str, Dict]]:
        """使用LLM进行流式表选择"""
        import inspect

        from dbgpt.core import (
            ModelMessage,
            ModelMessageRoleType,
            ModelRequest,
            ModelRequestContext,
        )

        prompt = self._build_selection_prompt(user_query, tables_info, chat_history)
        
        logger.debug(f"表选择prompt: {prompt[:500]}...")

        request_params = {
            "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            "max_new_tokens": 1500,
            "context": ModelRequestContext(stream=True),
        }

        # 验证模型是否存在，如果不存在则使用默认模型（不指定model_name）
        actual_model_name = None
        if self.model_name and self.llm_client:
            try:
                # 获取所有可用模型
                available_models = await self.llm_client.models()
                model_names = [m.model for m in available_models]
                
                if self.model_name in model_names:
                    actual_model_name = self.model_name
                    logger.debug(f"✓ 使用指定模型: {self.model_name}")
                else:
                    logger.warning(
                        f"⚠️ 指定的模型 '{self.model_name}' 不存在，可用模型: {model_names}。"
                        f"将使用默认模型（不指定model_name，让LLM client自动选择）"
                    )
                    # 不设置 model_name，让 LLM client 使用默认模型
                    actual_model_name = None
            except Exception as e:
                logger.warning(
                    f"⚠️ 验证模型 '{self.model_name}' 时出错: {e}，"
                    f"将使用默认模型（不指定model_name）"
                )
                actual_model_name = None
        
        # 只有在模型存在时才添加到请求中
        if actual_model_name:
            request_params["model"] = actual_model_name

        request = ModelRequest(**request_params)
        stream_response = self.llm_client.generate_stream(request)

        full_text = ""
        if inspect.isasyncgen(stream_response):
            async for chunk in stream_response:
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
                        yield full_text
                except Exception as e:
                    logger.debug(f"获取chunk.text失败: {e}")
                    continue
        else:
            raise Exception(f"Unexpected response type: {type(stream_response)}")

        # 解析结果
        if full_text:
            result = self._parse_selection_result(full_text, tables_info)
            yield result

    def _parse_selection_result(
        self, llm_output: str, tables_info: List[Dict]
    ) -> Dict:
        """解析LLM输出的表选择结果"""
        try:
            # 提取JSON部分
            start_idx = llm_output.find("{")
            end_idx = llm_output.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_output[start_idx:end_idx]
                result = json.loads(json_str)

                # 验证选择的表名是否有效
                valid_table_names = {t["table_name"] for t in tables_info}
                selected = result.get("selected_tables", [])
                
                # 过滤无效的表名
                valid_selected = [t for t in selected if t in valid_table_names]
                
                if not valid_selected:
                    # 如果没有有效的表名，返回所有表
                    logger.warning("LLM选择的表名都无效，返回所有表")
                    valid_selected = list(valid_table_names)
                    result["selection_reason"] = "无法确定具体表，返回所有可用表"

                result["selected_tables"] = valid_selected
                
                logger.info(f"✅ 表选择完成: {valid_selected}")
                return result
            else:
                raise Exception("无法从LLM输出中提取JSON")

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 返回所有表
            return {
                "selected_tables": [t["table_name"] for t in tables_info],
                "selection_reason": "JSON解析失败，返回所有可用表",
            }
        except Exception as e:
            logger.error(f"解析表选择结果失败: {e}")
            return {
                "selected_tables": [t["table_name"] for t in tables_info],
                "selection_reason": f"解析失败: {e}，返回所有可用表",
            }

    def select_tables_sync(
        self,
        user_query: str,
        tables_info: List[Dict],
    ) -> Dict:
        """
        同步选择相关表（非流式版本）
        
        Args:
            user_query: 用户问题
            tables_info: 表信息列表
            
        Returns:
            选择结果字典
        """
        # 如果只有一个表，直接返回
        if len(tables_info) == 1:
            return {
                "selected_tables": [tables_info[0]["table_name"]],
                "selection_reason": "只有一个可用表",
            }

        # 使用LLM选择表
        if self.llm_client:
            try:
                return self._llm_based_selection_sync(user_query, tables_info)
            except Exception as e:
                logger.warning(f"LLM表选择失败: {e}，使用规则选择")

        # Fallback: 基于规则的选择
        return self._rule_based_selection(user_query, tables_info)

    def _llm_based_selection_sync(
        self,
        user_query: str,
        tables_info: List[Dict],
    ) -> Dict:
        """同步LLM表选择"""
        import asyncio
        import inspect

        from dbgpt.core import (
            ModelMessage,
            ModelMessageRoleType,
            ModelRequest,
            ModelRequestContext,
        )

        prompt = self._build_selection_prompt(user_query, tables_info)

        request_params = {
            "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            "max_new_tokens": 1500,
            "context": ModelRequestContext(stream=False),
        }

        # 验证模型是否存在，如果不存在则使用默认模型（不指定model_name）
        actual_model_name = None
        if self.model_name and self.llm_client:
            try:
                # 同步方法中使用 asyncio.run 来调用异步方法
                available_models = asyncio.run(self.llm_client.models())
                model_names = [m.model for m in available_models]
                
                if self.model_name in model_names:
                    actual_model_name = self.model_name
                    logger.debug(f"✓ 使用指定模型: {self.model_name}")
                else:
                    logger.warning(
                        f"⚠️ 指定的模型 '{self.model_name}' 不存在，可用模型: {model_names}。"
                        f"将使用默认模型（不指定model_name，让LLM client自动选择）"
                    )
                    # 不设置 model_name，让 LLM client 使用默认模型
                    actual_model_name = None
            except Exception as e:
                logger.warning(
                    f"⚠️ 验证模型 '{self.model_name}' 时出错: {e}，"
                    f"将使用默认模型（不指定model_name）"
                )
                actual_model_name = None
        
        # 只有在模型存在时才添加到请求中
        if actual_model_name:
            request_params["model"] = actual_model_name

        request = ModelRequest(**request_params)
        stream_response = self.llm_client.generate_stream(request)

        full_text = ""
        if inspect.isasyncgen(stream_response):
            async def collect_async():
                text = ""
                async for chunk in stream_response:
                    try:
                        if hasattr(chunk, "text"):
                            text = chunk.text
                    except Exception:
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
                try:
                    if hasattr(chunk, "text"):
                        full_text = chunk.text
                except Exception:
                    pass

        if full_text:
            return self._parse_selection_result(full_text, tables_info)
        
        # 如果LLM没有返回结果，使用规则选择
        return self._rule_based_selection(user_query, tables_info)

