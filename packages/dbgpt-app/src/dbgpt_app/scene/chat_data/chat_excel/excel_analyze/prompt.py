# ruff: noqa: E501
from dbgpt._private.config import Config
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)
from dbgpt_app.scene import AppScenePromptTemplateAdapter, ChatScene
from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.out_parser import (
    ChatExcelOutputParser,
)

CFG = Config()

_PROMPT_SCENE_DEFINE_EN = "You are a data analysis expert. "


# ===== SYSTEM PROMPT（简洁、稳定、不变化）=====
_SYSTEM_PROMPT_ZH = """你是一个专业的数据分析专家。

你的职责是理解用户的数据分析需求，并生成正确的 DuckDB SQL 查询。"""

_SYSTEM_PROMPT_EN = """You are a professional data analysis expert.

Your responsibility is to understand the user's data analysis requirements and generate correct DuckDB SQL queries."""  # noqa: E501

# ===== 可复用的 DuckDB 规则块 =====
_DUCKDB_RULES_ZH = """
### 【必须遵守】DuckDB 列名/表名引号规则：

**以下情况必须使用双引号包裹字段名或表名：**
1. 包含中文字符（如 "客户名称"、"地区"）
2. 以数字开头（如 "2022_销售额"、"2023_利润"）
3. 包含下划线以外的特殊字符或空格（如 "订单 ID"）

**严禁使用未加双引号的数字开头字段名！**

**【极其重要】列名必须完全匹配：**
- 必须使用表结构中提供的**完整列名**，包括所有括号、连字符等特殊字符
- 列名必须**逐字符精确匹配**表结构中的定义

### 【GROUP BY 关键规则】：
1. SELECT 中的非聚合列必须在 GROUP BY 中
2. ORDER BY 中的列必须在前面的 CTE 或查询中被正确选择
3. 多层 CTE 中各层之间的列引用必须一致，特别是排序和连接的列
**DuckDB的WHERE子句必须在SELECT列定义之前处理,不能引用SELECT中定义的别名**

### 【时间序列分析】：
- **同比分析**：需要至少连续两年的数据
- **环比分析**：需要足够的历史数据周期
- **使用 LAG 函数**：LAG(value, 1) 用于环比，LAG(value, 12) 用于同比
"""

_DUCKDB_RULES_EN = """
### 【MUST FOLLOW】DuckDB Column/Table Name Quoting Rules:

**The following cases MUST use double quotes around field/table names:**
1. Contains Chinese characters (e.g., "客户名称", "地区")
2. Starts with a digit (e.g., "2022_sales", "2023_profit")
3. Contains special characters other than underscore or spaces (e.g., "order ID")

**NEVER use digit-starting field names without double quotes!**

### 【GROUP BY Key Rules】：
1. Non-aggregate columns in SELECT must be in GROUP BY
2. Columns in ORDER BY must be properly selected in preceding CTE or query
3. Ensure column reference consistency between CTE layers, especially for columns used in sorting and joining

### 【Time Series Analysis】：
- **Year-over-year**: Requires at least 2 years of continuous data
- **Month-over-month**: Requires sufficient historical data periods
- **LAG Function**: LAG(value, 1) for month-over-month, LAG(value, 12) for year-over-year
"""

# ===== 可复用的约束条件块 =====
_ANALYSIS_CONSTRAINTS_ZH = """
表名：{table_name}
列名规则：中文/数字开头/特殊字符必须用双引号;不要使用 UNION / UNION ALL，如需多个结果请分别查询；时间戳处理使用 to_timestamp() 而非直接 CAST；注释行必须单独成行，不要放在 SQL 语句的同一行
图表优先：默认使用图表，分类对比用bar/pie，时序用line/area，仅明细记录用table
可用方式：{display_type}
展示顺序：数据摘要 → 图表可视化 → SQL查询
"""

_ANALYSIS_CONSTRAINTS_EN = """
Table: {table_name}
Column rules: Chinese/digit-starting/special chars need double quotes
Chart priority: Default to charts, categorical use bar/pie, time-series use line/area, only detailed records use table
Available types: {display_type}
Display order: Data summary → Chart visualization → SQL query
"""

# ===== 可复用的示例块 =====
_EXAMPLES_ZH = """
【示例 - 时间趋势，使用折线图】：
user: 看一下销售趋势
assistant: 为您展示销售趋势变化：
<api-call><name>response_line_chart</name><args><sql>
SELECT "日期", SUM("销售额") AS "销售额"
FROM data_analysis_table
WHERE "日期" IS NOT NULL
GROUP BY "日期"
ORDER BY "日期";
</sql></args></api-call>

"""

_EXAMPLES_EN = """
【Example  - Time trend, use line chart】：
user: Show sales trend
assistant: Displaying sales trend:
<api-call><name>response_line_chart</name><args><sql>
SELECT "date", SUM("sales") AS "sales"
FROM data_analysis_table
WHERE "date" IS NOT NULL
GROUP BY "date"
ORDER BY "date";
</sql></args></api-call>

"""

# ===== USER PROMPT TEMPLATE（动态生成，每次查询都变化）=====
_USER_PROMPT_TEMPLATE_ZH = """
请按照以下要求回答用户问题：

【输出要求】：
1. 按照<api-call><name>图表工具</name><args><sql>SQL语句</sql></args></api-call>的格式输出，不要输出其他内容。
2. 必须输出 <api-call></api-call> 代码块，并且SQL 不能为空
3. 优先使用图表类型(bar/line/pie/area)，避免过度使用table
4. 禁止：冗长说明、多个api-call、提前给结论、在api-call外输出SQL
5. 展示顺序：数据摘要 → 图表可视化 → SQL查询

【数据表结构】
{table_schema}

【数据样本】
{data_example}

{data_time_range}
{relevant_columns_info}

【DuckDB语法规则】
{duckdb_syntax_rules}

【约束条件】
{analysis_constraints}

【示例】
{examples}

用户问题：{query_rewrite_info}
"""

_USER_PROMPT_TEMPLATE_EN = """
【Output Requirements】:
1. According to the format <api-call><name>chart tool</name><args><sql>SQL statement</sql></args></api-call>, do not output other content.
2. Must output exactly ONE <api-call></api-call> block, SQL inside must NOT be empty
3. Prioritize chart types (bar/line/pie/area), avoid overusing table
4. Prohibited: lengthy explanations, multiple api-calls, premature conclusions, SQL outside api-call
5. Display order: Data summary → Chart visualization → SQL query

【Data Table Structure】
{table_schema}

【Data Sample】
{data_example}

{data_time_range}
{relevant_columns_info}

【DuckDB Syntax Rules】
{duckdb_syntax_rules}

【Constraints】
{analysis_constraints}

【Example】
{examples}

User Question: {query_rewrite_info}
"""

# ===== 默认选择合适的模板（保持向后兼容） =====
_SYSTEM_PROMPT = _SYSTEM_PROMPT_EN if CFG.LANGUAGE == "en" else _SYSTEM_PROMPT_ZH

_DUCKDB_RULES = _DUCKDB_RULES_EN if CFG.LANGUAGE == "en" else _DUCKDB_RULES_ZH

_ANALYSIS_CONSTRAINTS_TEMPLATE = (
    _ANALYSIS_CONSTRAINTS_EN if CFG.LANGUAGE == "en" else _ANALYSIS_CONSTRAINTS_ZH
)

_EXAMPLES = _EXAMPLES_EN if CFG.LANGUAGE == "en" else _EXAMPLES_ZH

_USER_PROMPT_TEMPLATE = (
    _USER_PROMPT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _USER_PROMPT_TEMPLATE_ZH
)


# ===== 动态语言选择工厂函数 =====
def get_prompt_templates_by_language(language: str = "zh"):
    """
    根据指定语言返回对应的 prompt 模板

    Args:
        language: "zh" 或 "en"

    Returns:
        包含所有模板的字典
    """
    is_english = language == "en"

    return {
        "system_prompt": _SYSTEM_PROMPT_EN if is_english else _SYSTEM_PROMPT_ZH,
        "duckdb_rules": _DUCKDB_RULES_EN if is_english else _DUCKDB_RULES_ZH,
        "analysis_constraints": (
            _ANALYSIS_CONSTRAINTS_EN if is_english else _ANALYSIS_CONSTRAINTS_ZH
        ),
        "examples": _EXAMPLES_EN if is_english else _EXAMPLES_ZH,
        "user_prompt_template": (
            _USER_PROMPT_TEMPLATE_EN if is_english else _USER_PROMPT_TEMPLATE_ZH
        ),
    }


PROMPT_NEED_STREAM_OUT = (
    True  # 启用流式输出，支持分阶段展示 Query改写 + SQL生成 + 最终结果
)

# Temperature is a configuration hyperparameter that controls the randomness of
# language model output.
# A high temperature produces more unpredictable and creative results, while a low
# temperature produces more common and conservative output.
# For example, if you adjust the temperature to 0.5, the model will usually generate
# text that is more predictable and less creative than if you set the temperature to
# 1.0.
PROMPT_TEMPERATURE = 0.3

prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanPromptTemplate.from_template(_USER_PROMPT_TEMPLATE),
    ]
)

prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ChatExcel.value(),
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=ChatExcelOutputParser(),
    temperature=PROMPT_TEMPERATURE,
)
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
