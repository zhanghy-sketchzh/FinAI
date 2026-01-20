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
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        计算两个字符串之间的编辑距离（Levenshtein距离）
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 代替 j，因为 previous_row 和 current_row 的索引比 s2 多1
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _find_similar_columns(
        self, 
        invalid_column: str, 
        valid_columns: Dict[str, set], 
        table_name: str = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        为无效字段找到最相似的有效字段
        
        Args:
            invalid_column: 无效的字段名
            valid_columns: 有效字段字典 {table_name: set(column_names)}
            table_name: 指定的表名（如果有）
            top_k: 返回前k个最相似的字段
            
        Returns:
            相似字段列表，每个元素包含 {table_name, column_name, distance, similarity}
        """
        similar_columns = []
        
        # 如果指定了表名，只在该表中查找
        if table_name and table_name in valid_columns:
            search_tables = {table_name: valid_columns[table_name]}
        else:
            search_tables = valid_columns
        
        for tbl_name, columns in search_tables.items():
            for col_name in columns:
                # 计算编辑距离
                distance = self._calculate_edit_distance(
                    invalid_column.lower(), 
                    col_name.lower()
                )
                
                # 计算相似度（0-1之间，1表示完全相同）
                max_len = max(len(invalid_column), len(col_name))
                similarity = 1 - (distance / max_len) if max_len > 0 else 0
                
                similar_columns.append({
                    "table_name": tbl_name,
                    "column_name": col_name,
                    "distance": distance,
                    "similarity": similarity
                })
        
        # 按相似度排序（从高到低）
        similar_columns.sort(key=lambda x: (-x["similarity"], x["distance"]))
        
        # 返回前k个
        return similar_columns[:top_k]

    def _extract_valid_column_names(self, table_schema_json: str) -> Union[set, Dict[str, set]]:
        """
        从schema中提取所有有效的字段名
        支持单表和多表模式
        
        Returns:
            单表模式：返回字段名的集合 (set)
            多表模式：返回字典 {table_name: set(column_names)}
        """
        try:
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            if not isinstance(schema_obj, dict):
                return set()

            # 检查是否为多表模式
            if schema_obj.get("is_multi_table"):
                # 多表模式：返回 {table_name: set(columns)} 的字典
                table_columns_map = {}
                
                for table in schema_obj.get("tables", []):
                    table_name = table.get("table_name", "")
                    if not table_name:
                        continue
                    
                    table_columns = set()
                    
                    # 方式1：从 create_table_sql 中解析字段名
                    create_table_sql = table.get("create_table_sql", "")
                    if create_table_sql:
                        columns_from_sql = self._extract_columns_from_sql(create_table_sql)
                        table_columns.update(columns_from_sql)
                    
                    # 方式2：从 columns 列表中提取
                    columns = table.get("columns", [])
                    for col in columns:
                        if isinstance(col, dict):
                            col_name = col.get("column_name", "")
                            if col_name:
                                table_columns.add(col_name)
                        elif isinstance(col, str):
                            if col:
                                table_columns.add(col)
                    
                    table_columns_map[table_name] = table_columns
                
                return table_columns_map
            else:
                # 单表模式：返回字段名集合
                valid_columns = set()
                columns = schema_obj.get("columns", [])
                for col in columns:
                    if isinstance(col, dict):
                        col_name = col.get("column_name", "")
                        if col_name:
                            valid_columns.add(col_name)
                    elif isinstance(col, str):
                        if col:
                            valid_columns.add(col)

            if valid_columns:
                logger.debug(f"✅ 提取到的有效字段名数量: {len(valid_columns)}")
                logger.debug(f"字段名示例（前5个）: {list(valid_columns)[:5]}")
            else:
                logger.warning(f"⚠️ 未提取到任何有效字段名，schema格式可能不正确")
            return valid_columns
        except Exception as e:
            logger.warning(f"提取字段名失败: {e}")
            return set()
    
    def _extract_columns_from_sql(self, create_table_sql: str) -> set:
        """
        从 CREATE TABLE SQL 语句中提取字段名
        
        Args:
            create_table_sql: CREATE TABLE SQL 语句
            
        Returns:
            字段名的集合
        """
        import re
        columns = set()
        
        try:
            if not create_table_sql:
                return columns
            
            # 匹配 CREATE TABLE ... (字段定义) 中的字段
            # 模式1: "字段名" 类型
            # 模式2: "字段名" 类型,
            # 使用正则表达式匹配被双引号包围的字段名
            pattern = r'"([^"]+)"'
            matches = re.findall(pattern, create_table_sql)
            
            for match in matches:
                # 过滤掉可能是表名的情况（通常在 CREATE TABLE 后面）
                # 简单检查：如果 match 在 CREATE TABLE 语句的表名位置，跳过
                # 但对于大多数情况，所有匹配的双引号内容都可能是字段名
                columns.add(match)
            
            logger.debug(f"从SQL中提取到 {len(columns)} 个字段名")
        except Exception as e:
            logger.warning(f"从SQL提取字段名失败: {e}")
        
        return columns

    def _simplify_schema_for_rewrite(self, table_schema_json: str) -> str:
        """
        精简schema用于query改写，只保留必要字段信息
        移除 suggested_questions_zh、suggested_questions_en 等不必要字段
        支持多表 schema（is_multi_table=True）
        """
        try:
            if isinstance(table_schema_json, str):
                schema_obj = json.loads(table_schema_json)
            else:
                schema_obj = table_schema_json

            if not isinstance(schema_obj, dict):
                return table_schema_json

            # 检查是否为多表 schema
            if schema_obj.get("is_multi_table"):
                return self._simplify_multi_table_schema(schema_obj)

            # 单表模式：只保留columns字段，移除建议问题等
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

    def _simplify_multi_table_schema(self, schema_obj: dict) -> str:
        """
        精简多表 schema，为每个表保留必要的字段信息
        支持两种格式：
        1. 旧格式：包含 columns 列表
        2. 新格式：包含 create_table_sql 和 sample_rows
        """
        simplified = {
            "is_multi_table": True,
            "table_count": schema_obj.get("table_count", 0),
            "table_names": schema_obj.get("table_names", []),
            "table_description": schema_obj.get("table_description", ""),
            "tables": [],
        }
        
        for table in schema_obj.get("tables", []):
            simplified_table = {
                "table_name": table.get("table_name", ""),
                "table_description": table.get("table_description", ""),
            }
            
            # 新格式：直接使用 create_table_sql 和 sample_rows
            if table.get("create_table_sql"):
                simplified_table["create_table_sql"] = table.get("create_table_sql")
                simplified_table["sample_rows"] = table.get("sample_rows", [])
                simplified_table["columns"] = table.get("columns", [])  # 列名列表
            else:
                # 旧格式：处理 columns 列表
                simplified_table["columns"] = []
                for col in table.get("columns", []):
                    if isinstance(col, dict):
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
                            simplified_col["possible_values"] = values[:10] if len(values) > 10 else values
                        # 保留数值统计摘要
                        if col.get("statistics_summary"):
                            simplified_col["stats"] = col["statistics_summary"]
                        simplified_table["columns"].append(simplified_col)
                    else:
                        # 如果是字符串（列名），直接添加
                        simplified_table["columns"].append(col)
            
            simplified["tables"].append(simplified_table)
        
        return json.dumps(simplified, ensure_ascii=False, indent=2)

    def _format_multi_table_info_for_prompt(self, simplified_schema: str) -> str:
        """
        将多表 schema 格式化为 prompt 中易读的格式
        包含建表 SQL 和样本数据
        """
        try:
            schema_obj = json.loads(simplified_schema) if isinstance(simplified_schema, str) else simplified_schema
        except Exception:
            return simplified_schema
        
        if not schema_obj.get("is_multi_table"):
            return simplified_schema
        
        result_parts = []
        
        for table in schema_obj.get("tables", []):
            table_name = table.get("table_name", "未知表")
            table_desc = table.get("table_description", "")
            
            table_section = f"### 表名: {table_name}\n"
            if table_desc:
                table_section += f"**表描述**: {table_desc}\n\n"
            
            # 添加建表 SQL
            create_sql = table.get("create_table_sql", "")
            if create_sql:
                table_section += f"**建表SQL**:\n```sql\n{create_sql}\n```\n\n"
            else:
                # 如果没有建表SQL，从 columns 构建字段列表
                columns = table.get("columns", [])
                if columns:
                    if isinstance(columns[0], dict):
                        col_names = [col.get("column_name", "") for col in columns]
                    else:
                        col_names = columns
                    table_section += f"**字段列表**: {', '.join(col_names)}\n\n"
            
            # 添加样本数据
            sample_rows = table.get("sample_rows", [])
            columns = table.get("columns", [])
            if sample_rows:
                table_section += "**样本数据（前2行）**:\n"
                # 如果有列名，显示为表格形式
                if columns:
                    if isinstance(columns[0], dict):
                        col_names = [col.get("column_name", "") for col in columns]
                    else:
                        col_names = columns
                    table_section += f"列名: {col_names}\n"
                for i, row in enumerate(sample_rows[:2]):
                    table_section += f"行{i+1}: {row}\n"
                table_section += "\n"
            
            result_parts.append(table_section)
        
        return "\n---\n".join(result_parts)

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
        
        # 检测是否为多表模式
        is_multi_table = False
        try:
            schema_obj = json.loads(table_schema_json) if isinstance(table_schema_json, str) else table_schema_json
            is_multi_table = schema_obj.get("is_multi_table", False)
        except Exception:
            pass
        
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
                chat_history[-9:] if len(chat_history) > 9 else chat_history
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
        logger.info(f"🌐 Query改写 - 检测到用户输入语言: {user_language}, 多表模式: {is_multi_table}")

        # 根据语言和是否多表选择prompt
        if user_language == "zh":
            if is_multi_table:
                prompt = self._build_multi_table_rewrite_prompt_zh(
                    user_query, simplified_schema, table_description, 
                    sample_rows_str, history_context
                )
            else:
                prompt = self._build_single_table_rewrite_prompt_zh(
                    user_query, simplified_schema, table_description,
                    sample_rows_str, history_context
                )
        else:
            if is_multi_table:
                prompt = self._build_multi_table_rewrite_prompt_en(
                    user_query, simplified_schema, table_description,
                    sample_rows_str, history_context
                )
            else:
                prompt = self._build_single_table_rewrite_prompt_en(
                    user_query, simplified_schema, table_description,
                    sample_rows_str, history_context
                )
        return prompt

    def _build_single_table_rewrite_prompt_zh(
        self, user_query: str, simplified_schema: str, table_description: str,
        sample_rows_str: str, history_context: str
    ) -> str:
        """构建单表模式的中文改写prompt"""
        return f"""你是一个数据分析专家。用户提出了一个数据分析问题，你需要：
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
  "is_relevant": true,
  "rewritten_query": "改写后的完整问题，明确指出需要分析的维度和指标",
  "analysis_suggestions": [
    "建议1：具体的分析步骤或注意事项",
    "建议2：...",
    "建议3：..."
  ],
  "analysis_logic": "分析逻辑的详细说明",
  "relevant_columns": [
    {{
      "column_name": "列名",
      "usage": "用途说明（如：筛选条件/分组维度/聚合指标）"
    }}
  ],
  "domain_knowledge": null
}}

**严格要求 - 字段名必须来自提供的列表**：
- relevant_columns 中的 column_name 必须**严格从上面提供的字段信息中选择**
- **禁止推测、创造或编造任何不在字段列表中的字段名**

现在请结合历史上下文及用户当前问题，分析用户的真实意图，补充改写当前问题并用中文输出JSON：
"""

    def _build_multi_table_rewrite_prompt_zh(
        self, user_query: str, simplified_schema: str, table_description: str,
        sample_rows_str: str, history_context: str
    ) -> str:
        """构建多表模式的中文改写prompt"""
        # 解析 schema 以获取各表的建表SQL和样本数据
        tables_info_str = self._format_multi_table_info_for_prompt(simplified_schema)
        
        # 提取字段差异警告（如果有）
        schema_diff_warning = ""
        try:
            schema_obj = json.loads(simplified_schema) if isinstance(simplified_schema, str) else simplified_schema
            if schema_obj.get("schema_differences"):
                schema_diff_warning = f"\n\n{schema_obj['schema_differences']}\n"
        except Exception:
            pass
        
        return f"""你是一个数据分析专家。用户提出了一个需要**多表联合查询**的数据分析问题。

**重要：这是一个多表查询场景，你需要考虑如何联合使用多个表来回答用户的问题。**

=== 可用的数据表 ===
{tables_info_str}
{schema_diff_warning}
**注意**：
1. 上面包含多个表的建表SQL和样本数据
2. 你需要分析哪些表和字段与用户问题相关
3. 如果需要跨表查询，考虑使用 UNION ALL 合并相似结构的表，或使用 JOIN 关联不同表
4. **不同表的字段名可能不同，但含义相似**，请仔细对比各表的字段名和样本数据来理解字段对应关系
5. **⚠️ 关键：在使用UNION ALL时，必须确保SELECT的字段在所有表中都存在！如果某个字段只存在于部分表，请使用NULL或默认值填充不存在该字段的表**

=== 数据表描述 ===
{table_description}

{history_context}
=== 用户当前问题 ===
{user_query}

=== 输出格式（JSON） ===
请严格按照以下JSON格式输出：
{{
  "is_relevant": true,
  "rewritten_query": "改写后的完整问题，明确指出需要从哪些表查询、分析的维度和指标",
  ],
  "analysis_suggestions": [
    "建议1：说明如何联合多个表进行查询",
    "建议2：如果表结构相似，建议使用 UNION ALL 合并后再分析",
    "建议3：如果需要关联不同表，说明 JOIN 的方式",
    "建议4：..."
  ],
  "analysis_logic": "多表分析逻辑的详细说明，包括：1) 使用哪些表 2) 如何联合（UNION/JOIN）3) 筛选条件 4) 分组维度 5) 聚合指标",
  "relevant_columns": [
    {{
      "table_name": "表名（多表时必须指定）",
      "column_name": "列名（必须与建表SQL中的列名完全一致）",
      "usage": "用途说明（如：筛选条件/分组维度/聚合指标）"
    }}
  "multi_table_strategy": {{
    "strategy": "UNION_ALL 或 JOIN 或 SINGLE_TABLE",
    "tables_to_use": ["表名1", "表名2"],
    "join_condition": "如果是JOIN，说明关联条件；如果是UNION_ALL，说明字段映射关系"
  }},
  "domain_knowledge": null
}}

**多表查询策略说明**：
- **UNION_ALL**：当多个表结构相似，需要合并所有数据进行分析时使用
- **JOIN**：当需要关联不同表的数据时使用
- **SINGLE_TABLE**：如果只需要查询单个表

**严格要求**：
- relevant_columns 中的 column_name 必须**严格从上面提供的建表SQL中选择**
- 多表模式下，必须在 relevant_columns 中指定 table_name
- **不同表的相同含义字段名可能不同，必须分别列出**
- **禁止推测、创造或编造任何不在字段列表中的字段名**

现在请结合历史上下文及用户当前问题，分析用户的真实意图，考虑多表联合查询的方式，补充改写当前问题并用中文输出JSON：
"""

    def _build_single_table_rewrite_prompt_en(
        self, user_query: str, simplified_schema: str, table_description: str,
        sample_rows_str: str, history_context: str
    ) -> str:
        """构建单表模式的英文改写prompt"""
        return f"""You are a data analysis expert. The user has asked a data analysis question. You need to:
1. Understand the user's real intent based on historical questions and answers, and enhance the current question
2. Enhance the user's question based on the data table field information
3. Clearly identify the columns that may be used (including filter condition columns, grouping dimension columns, and aggregation indicator columns)
4. Provide 3-5 analysis suggestions explaining how to analyze this question
5. Give a clear analysis logic

=== Data Table Field Information ===
{simplified_schema}

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
  "is_relevant": true,
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
  "analysis_logic": "Detailed explanation of analysis logic",
  "domain_knowledge": null
}}

**STRICT REQUIREMENT - Column names must come from the provided list**:
- The column_name in relevant_columns MUST be **strictly selected from the field information provided above**
- **DO NOT guess, create, or fabricate any column names that are not in the field list**

Now please analyze the user's real intent and output JSON IN ENGLISH:
"""

    def _build_multi_table_rewrite_prompt_en(
        self, user_query: str, simplified_schema: str, table_description: str,
        sample_rows_str: str, history_context: str
    ) -> str:
        """构建多表模式的英文改写prompt"""
        # 解析 schema 以获取各表的建表SQL和样本数据
        tables_info_str = self._format_multi_table_info_for_prompt(simplified_schema)
        
        # 提取字段差异警告（如果有）
        schema_diff_warning = ""
        try:
            schema_obj = json.loads(simplified_schema) if isinstance(simplified_schema, str) else simplified_schema
            if schema_obj.get("schema_differences"):
                # 将中文警告转换为英文
                diff_text = schema_obj['schema_differences']
                # 简单处理：保留原文，因为警告中包含具体字段名
                schema_diff_warning = f"\n\n{diff_text}\n"
        except Exception:
            pass
        
        return f"""You are a data analysis expert. The user has asked a data analysis question that requires **multi-table query**.

**IMPORTANT: This is a multi-table query scenario. You need to consider how to combine multiple tables to answer the user's question.**

=== Available Data Tables ===
{tables_info_str}
{schema_diff_warning}
**Note**:
1. The above contains CREATE TABLE SQL and sample data for each table
2. You need to analyze which tables and fields are relevant to the user's question
3. If cross-table query is needed, consider using UNION ALL to merge similar tables, or JOIN to relate different tables
4. **Different tables may have different column names but similar meanings**, please carefully compare column names and sample data to understand the field mapping
5. **⚠️ CRITICAL: When using UNION ALL, ensure that the SELECTed fields exist in ALL tables! If a field only exists in some tables, use NULL or default values to fill in the tables that don't have that field**

=== Data Table Description ===
{table_description}

{history_context}
=== User's Current Question ===
{user_query}

=== Output Format (JSON) ===
Please strictly follow the following JSON format:
{{
  "is_relevant": true,
  "rewritten_query": "The enhanced complete question, clearly indicating which tables to query, dimensions and indicators to analyze",
  "relevant_columns": [
    {{
      "table_name": "Table name (required for multi-table)",
      "column_name": "Column name (must match exactly with CREATE TABLE SQL)",
      "usage": "Usage description (e.g., filter condition/grouping dimension/aggregation indicator)"
    }}
  ],
  "analysis_suggestions": [
    "Suggestion 1: Explain how to combine multiple tables for query",
    "Suggestion 2: If table structures are similar, suggest using UNION ALL to merge before analysis",
    "Suggestion 3: If different tables need to be related, explain the JOIN method",
    "Suggestion 4: ..."
  ],
  "analysis_logic": "Detailed explanation of multi-table analysis logic, including: 1) Which tables to use 2) How to combine (UNION/JOIN) 3) Filter conditions 4) Grouping dimensions 5) Aggregation indicators",
  "multi_table_strategy": {{
    "strategy": "UNION_ALL or JOIN or SINGLE_TABLE",
    "tables_to_use": ["table1", "table2"],
    "join_condition": "If JOIN, explain the join condition; if UNION_ALL, explain the field mapping"
  }},
  "domain_knowledge": null
}}

**Multi-table Query Strategy**:
- **UNION_ALL**: Use when multiple tables have similar structures and need to merge all data for analysis
- **JOIN**: Use when data from different tables needs to be related
- **SINGLE_TABLE**: If only one table needs to be queried

**STRICT REQUIREMENT**:
- The column_name in relevant_columns MUST be **strictly selected from the CREATE TABLE SQL provided above**
- In multi-table mode, table_name MUST be specified in relevant_columns
- **Different tables may have different column names for the same meaning - list them separately**

Now please analyze the user's real intent, consider multi-table query approach, and output JSON IN ENGLISH:
"""

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
        if valid_column_names:
            if isinstance(valid_column_names, dict):
                # 多表模式
                total_cols = sum(len(cols) for cols in valid_column_names.values())
                logger.info(f"✅ 多表模式：提取到 {len(valid_column_names)} 个表，共 {total_cols} 个字段用于校验")
                for tbl_name, tbl_cols in list(valid_column_names.items())[:3]:
                    logger.debug(f"  表 '{tbl_name}': {len(tbl_cols)} 个字段")
            else:
                # 单表模式
                logger.info(f"✅ 单表模式：提取到 {len(valid_column_names)} 个有效字段名用于校验")
                logger.debug(f"有效字段名（前10个）: {list(valid_column_names)[:10]}")
        else:
            logger.warning(f"⚠️ 未提取到有效字段名，将跳过字段名校验")

        # 检测用户语言
        user_language = detect_language(user_query)

        # 构建prompt
        prompt = self._build_rewrite_prompt(
            user_query, table_schema_json, table_description, 
            chat_history=chat_history, sample_rows=sample_rows
        )
        
        # 如果有无效字段提示，追加到prompt中（根据语言选择）
        if invalid_columns_hint:
            # 解析无效字段列表并找相似字段
            similar_suggestions = []
            import re
            
            # 处理不同格式的invalid_columns_hint
            if isinstance(invalid_columns_hint, list):
                invalid_fields = invalid_columns_hint
            elif isinstance(invalid_columns_hint, str):
                # 从字符串中提取字段名（格式：['字段1', '字段2']）
                invalid_fields = re.findall(r"'([^']+)'", invalid_columns_hint)
            else:
                invalid_fields = []
            
            if invalid_fields:
                for invalid_field in invalid_fields:
                    # 提取表名和字段名
                    table_match = re.search(r'\(字段不存在于表\s*[\'"]?([^\'"）]+)[\'"]?\s*中\)', invalid_field)
                    if table_match:
                        specified_table = table_match.group(1)
                        # 去掉表名后缀，获取纯字段名
                        pure_field = re.sub(r'\s*\(字段不存在于表.*\)', '', invalid_field)
                    else:
                        specified_table = None
                        pure_field = invalid_field
                    
                    # 查找相似字段
                    similar_cols = self._find_similar_columns(
                        pure_field, 
                        valid_column_names,
                        specified_table,
                        top_k=3
                    )
                    
                    if similar_cols:
                        # 过滤相似度>0.6的字段
                        good_matches = [c for c in similar_cols if c['similarity'] > 0.6]
                        if good_matches:
                            if user_language == "zh":
                                suggestion = f"  • 无效字段：{invalid_field}\n    相似字段推荐："
                                for match in good_matches:
                                    suggestion += f"\n      - {match['table_name']}.{match['column_name']} (相似度: {match['similarity']:.2f})"
                            else:
                                suggestion = f"  • Invalid field: {invalid_field}\n    Similar field suggestions:"
                                for match in good_matches:
                                    suggestion += f"\n      - {match['table_name']}.{match['column_name']} (similarity: {match['similarity']:.2f})"
                            similar_suggestions.append(suggestion)
            
            # 将invalid_columns_hint转换为显示格式
            if isinstance(invalid_columns_hint, list):
                invalid_columns_display = str(invalid_columns_hint)
            else:
                invalid_columns_display = invalid_columns_hint
            
            if user_language == "zh":
                correction_hint = f"""

**错误纠正**：
你上次返回的以下字段名在数据表中不存在，请勿使用：
{invalid_columns_display}
"""
                if similar_suggestions:
                    correction_hint += "\n**相似字段推荐**（基于编辑距离算法）：\n"
                    correction_hint += "\n".join(similar_suggestions)
                    correction_hint += "\n\n请参考上面推荐的相似字段，或从字段列表中选择其他合适的字段。"
                else:
                    correction_hint += "\n请仔细检查上面提供的字段列表，只使用实际存在的字段名重新生成。如果找不到对应的字段，请在analysis_suggestions中说明该分析可能无法完成。"
            else:
                correction_hint = f"""

**Error Correction**:
The following column names you returned do not exist in the data table, DO NOT use them:
{invalid_columns_display}
"""
                if similar_suggestions:
                    correction_hint += "\n**Similar Field Suggestions** (based on edit distance algorithm):\n"
                    correction_hint += "\n".join(similar_suggestions)
                    correction_hint += "\n\nPlease refer to the similar fields recommended above, or choose other appropriate fields from the field list."
                else:
                    correction_hint += "\nPlease carefully check the field list provided above and regenerate using only column names that actually exist. If you cannot find the corresponding field, please explain in analysis_suggestions that this analysis may not be possible."
            
            prompt += correction_hint
            logger.info(f"🔄 重试第{retry_count}次，添加无效字段提示及相似字段推荐")
        
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
        self, llm_output: str, original_query: str, valid_column_names: Union[set, Dict[str, set]] = None
    ) -> Dict:
        """
        解析LLM输出的JSON结果

        如果解析失败或字段名不存在，抛出 JSONParseError 异常以触发重试机制
        
        Args:
            llm_output: LLM输出的文本
            original_query: 原始用户问题
            valid_column_names: 有效的字段名，单表模式为set，多表模式为dict {table_name: set(columns)}
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
                if relevant_columns and valid_column_names:
                    invalid_columns = []
                    is_multi_table = isinstance(valid_column_names, dict)
                    
                    for idx, col in enumerate(relevant_columns):
                        if not isinstance(col, dict) or "column_name" not in col:
                            logger.error(f"relevant_columns[{idx}] 格式错误: {col}")
                            raise JSONParseError(
                                f"relevant_columns[{idx}] 缺少 'column_name' 字段"
                            )
                        
                        col_name = col.get("column_name", "")
                        table_name = col.get("table_name", "")
                        
                        if not col_name:
                            continue
                        
                        # 多表模式：需要校验字段是否存在于指定的表中
                        if is_multi_table:
                            if not table_name:
                                # 多表模式下必须指定表名
                                logger.warning(f"多表模式下字段 '{col_name}' 未指定 table_name")
                                # 检查字段是否存在于任意表中
                                found_in_any_table = False
                                for tbl_name, tbl_cols in valid_column_names.items():
                                    if col_name in tbl_cols:
                                        found_in_any_table = True
                                        logger.info(f"字段 '{col_name}' 存在于表 '{tbl_name}' 中")
                                        break
                                
                                if not found_in_any_table:
                                    invalid_columns.append(f"{col_name} (未指定表名且不存在于任何表中)")
                            else:
                                # 检查指定的表是否存在
                                if table_name not in valid_column_names:
                                    invalid_columns.append(f"{table_name}.{col_name} (表 '{table_name}' 不存在)")
                                    logger.warning(f"表 '{table_name}' 不存在于有效表列表中")
                                else:
                                    # 检查字段是否存在于指定的表中
                                    table_cols = valid_column_names[table_name]
                                    if col_name not in table_cols:
                                        invalid_columns.append(f"{col_name} (字段不存在于表 '{table_name}' 中)")
                                        logger.warning(f"字段 '{col_name}' 不存在于表 '{table_name}' 中")
                                        logger.debug(f"表 '{table_name}' 的有效字段（前10个）: {list(table_cols)[:10]}")
                        else:
                            # 单表模式：直接校验字段名
                            pure_col_name = col_name
                            if "." in col_name:
                                parts = col_name.split(".", 1)
                                if len(parts) == 2:
                                    pure_col_name = parts[1].strip('"').strip("'")
                            
                            if pure_col_name not in valid_column_names and col_name not in valid_column_names:
                                invalid_columns.append(col_name)
                                logger.warning(f"字段名校验失败: '{col_name}' 不在有效字段列表中")
                    
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
