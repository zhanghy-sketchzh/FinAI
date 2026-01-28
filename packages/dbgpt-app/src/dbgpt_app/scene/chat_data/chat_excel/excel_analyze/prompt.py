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

**【极其重要】表名和列名必须完全匹配：**
- 表名必须使用约束条件中指定的**完整表名**，不能简化、缩写或修改
- 列名必须**逐字符精确匹配**表结构中的定义，包括括号、连字符等特殊字符
- 字符串条件必须完全匹配数据中的实际值，不能随意替换字符

### 【NULL值处理规则 - 极其重要】：
**在SQL中，NULL与任何值比较都返回NULL（不是TRUE或FALSE），会导致条件不匹配！**
- ❌ 错误：`WHERE "标签" != '剔除'` — 如果"标签"为NULL，此条件返回NULL，该行被过滤掉
- ✅ 正确：`WHERE ("标签" IS NULL OR "标签" != '剔除')` — 正确处理NULL值
- ✅ 正确：`WHERE COALESCE("标签", '') != '剔除'` — 使用COALESCE将NULL转为空字符串

**关键原则：排除某个值时，必须同时考虑NULL的情况**

### 【WHERE子句逻辑优先级】：
**混用AND和OR时必须用括号明确优先级：**
- ❌ 错误：`WHERE "部门"='A' OR "部门"='B' AND "年份"=2025`
- ✅ 正确：`WHERE ("部门"='A' OR "部门"='B') AND "年份"=2025`

### 【GROUP BY 关键规则】：
1. SELECT中的非聚合列必须在GROUP BY中
2. ORDER BY中的列必须在SELECT中存在
3. WHERE子句不能引用SELECT中定义的别名

### 【CTE和别名规则】：
- 同一SELECT中不能引用本层定义的别名
- CTE名称、表别名必须使用**英文或拼音**，禁止中文
  - ✅ 正确：`WITH sales_data AS (...)`
  - ❌ 错误：`WITH 销售数据 AS (...)`
- 避免除零：使用 `NULLIF(分母, 0)`，如 `ROUND(SUM("金额")/NULLIF(COUNT(*), 0), 2)`

### 【子查询使用规则】：
- 禁止在SELECT列表中使用返回多行的子查询
- 正确做法：使用JOIN或窗口函数替代
- 计算比例：`ROUND(SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)`

### 【数值格式化规则】：
所有数值列必须使用 `ROUND(column, 2)` 保留两位小数：
- `SELECT ROUND(SUM("金额"), 2) AS "总金额"`
- `SELECT ROUND("单价" * "数量", 2) AS "小计"`

### 【LIMIT使用规则】：
- 禁止自动添加LIMIT，除非用户明确要求（如"前10条"、"Top 5"）
- 默认展示所有符合条件的数据
"""

_DUCKDB_RULES_EN = """
### 【MUST FOLLOW】DuckDB Column/Table Name Quoting Rules:

**Use double quotes for field/table names when:**
1. Contains Chinese characters (e.g., "客户名称")
2. Starts with a digit (e.g., "2022_sales")
3. Contains special characters or spaces (e.g., "order ID")

**【CRITICAL】Table and Column Names Must Match Exactly:**
- Table name MUST use the **complete table name** from constraints, no simplification
- Column names must **match character-by-character** the definitions in table structure
- String conditions must exactly match actual values in data

### 【NULL Value Handling - CRITICAL】：
**In SQL, NULL compared with any value returns NULL (not TRUE or FALSE), causing rows to be filtered out!**
- ❌ Wrong: `WHERE "tag" != 'exclude'` — If "tag" is NULL, condition returns NULL, row filtered out
- ✅ Correct: `WHERE ("tag" IS NULL OR "tag" != 'exclude')` — Properly handles NULL
- ✅ Correct: `WHERE COALESCE("tag", '') != 'exclude'` — Use COALESCE to convert NULL to empty string

**Key principle: When excluding a value, always consider NULL cases**

### 【WHERE Clause Logic Priority】：
**Must use parentheses when mixing AND/OR:**
- ❌ Wrong: `WHERE "dept"='A' OR "dept"='B' AND "year"=2025`
- ✅ Correct: `WHERE ("dept"='A' OR "dept"='B') AND "year"=2025`

### 【GROUP BY Key Rules】：
1. Non-aggregate columns in SELECT must be in GROUP BY
2. ORDER BY columns must exist in SELECT
3. WHERE clause cannot reference aliases defined in SELECT

### 【CTE and Alias Rules】：
- Cannot reference aliases within same SELECT
- CTE names, table aliases MUST use **English or Pinyin**, no Chinese
  - ✅ Correct: `WITH sales_data AS (...)`
  - ❌ Wrong: `WITH 销售数据 AS (...)`
- Avoid division by zero: Use `NULLIF(denominator, 0)`

### 【Subquery Usage Rules】：
- NEVER use subqueries in SELECT list that return multiple rows
- Use JOIN or window functions instead
- Calculate percentage: `ROUND(SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)`

### 【Numeric Formatting Rules】：
All numeric columns must use `ROUND(column, 2)` to retain 2 decimal places:
- `SELECT ROUND(SUM("amount"), 2) AS "total"`
- `SELECT ROUND("price" * "quantity", 2) AS "subtotal"`

### 【LIMIT Usage Rules】：
- DO NOT automatically add LIMIT unless user explicitly requests (e.g., "top 10", "Top 5")
- Display all data that meets conditions by default
"""

# ===== 可复用的约束条件块 =====
_ANALYSIS_CONSTRAINTS_ZH = """
**【必须严格遵守】表名：{table_name}**

**核心规则**：
- 列名规则：中文/数字开头/特殊字符必须用双引号
- 字符串匹配：WHERE条件必须完全匹配数据实际值，不能随意替换字符
- **NULL处理：排除某值时必须考虑NULL，使用 (列 IS NULL OR 列 != 'xxx') 格式**
- WHERE逻辑：混用AND和OR时必须用括号明确优先级
- 别名规则：同一SELECT中不能引用本层别名；CTE名称必须用英文
- 数值格式：所有数值必须用 ROUND(column, 2) 保留两位小数
- LIMIT规则：禁止自动添加，除非用户明确要求

**展示规则**：
- 图表优先：分类对比用bar/pie，时序用line/area，仅明细用table
- 可用方式：{display_type}
"""

# ===== 多表模式的约束条件块 =====
_ANALYSIS_CONSTRAINTS_MULTI_TABLE_ZH = """
**【多表模式】可用的数据表：{table_names}**

**多表查询策略：**
1. **UNION ALL 合并查询**（结构相似的表）：
   - 用户询问"所有"、"全部"时使用
   - 不同表字段名可能不同，用AS统一别名
   - 缺失字段用NULL填充
   ```sql
   SELECT "字段A" AS "统一名", "数值" FROM "表1"
   UNION ALL
   SELECT "字段B" AS "统一名", NULL AS "数值" FROM "表2"
   ```

2. **单表查询**：用户明确指定某表时只查询对应表

3. **JOIN 关联查询**：需要关联不同类型的表时使用

**核心规则**：
- 表名/列名必须完全匹配，不能简化或修改
- **NULL处理：排除某值时必须考虑NULL，使用 (列 IS NULL OR 列 != 'xxx')**
- WHERE逻辑：混用AND和OR时必须用括号
- CTE名称必须用英文，禁止中文
- 数值格式：ROUND(column, 2) 保留两位小数
- LIMIT规则：禁止自动添加

**展示规则**：
- 图表优先：分类用bar/pie，时序用line/area
- 可用方式：{display_type}
"""

_ANALYSIS_CONSTRAINTS_MULTI_TABLE_EN = """
**【Multi-Table Mode】Available Tables: {table_names}**

**Multi-Table Query Strategies:**
1. **UNION ALL Merge Query** (similar-structured tables):
   - Use when user asks about "all", "total" data
   - Different tables may have different column names, use AS to unify
   - Fill missing fields with NULL
   ```sql
   SELECT "field_a" AS "unified", "value" FROM "table1"
   UNION ALL
   SELECT "field_b" AS "unified", NULL AS "value" FROM "table2"
   ```

2. **Single Table Query**: When user specifies a particular table

3. **JOIN Query**: When relating different types of tables

**Core Rules**:
- Table/Column names must match exactly, no simplification
- **NULL handling: When excluding a value, must consider NULL, use (col IS NULL OR col != 'xxx')**
- WHERE logic: Must use parentheses when mixing AND/OR
- CTE names must use English, no Chinese
- Numeric formatting: ROUND(column, 2) for 2 decimal places
- LIMIT rules: DO NOT add automatically

**Display Rules**:
- Chart priority: categorical use bar/pie, time-series use line/area
- Available types: {display_type}
"""

_ANALYSIS_CONSTRAINTS_EN = """
**【MUST STRICTLY FOLLOW】Table Name: {table_name}**

**Core Rules**:
- Column rules: Chinese/digit-starting/special chars need double quotes
- String matching: WHERE conditions must exactly match actual data values
- **NULL handling: When excluding a value, must consider NULL, use (col IS NULL OR col != 'xxx') format**
- WHERE logic: Must use parentheses when mixing AND/OR
- Alias rules: Cannot reference aliases within same SELECT; CTE names must use English
- Numeric formatting: All numbers must use ROUND(column, 2) for 2 decimal places
- LIMIT rules: DO NOT add automatically unless user explicitly requests

**Display Rules**:
- Chart priority: categorical use bar/pie, time-series use line/area, only details use table
- Available types: {display_type}
"""

# ===== 可复用的示例块 =====
_EXAMPLES_ZH = """
【示例1 - 基础聚合查询】：
user: 查看C部门2025年的销售总额
assistant: <api-call><name>response_table</name><args><sql>
SELECT "部门", "年份", ROUND(SUM("金额"), 2) AS "总额"
FROM "数据表"
WHERE "部门" = 'C' AND "年份" = 2025
GROUP BY "部门", "年份";
</sql></args></api-call>

【示例2 - NULL值处理（排除某个值时必须考虑NULL）】：
user: 查看所有非剔除标签的数据总和
assistant: <api-call><name>response_table</name><args><sql>
SELECT ROUND(SUM("金额"), 2) AS "总额"
FROM "数据表"
WHERE ("剔除标签" IS NULL OR "剔除标签" != '已剔除');
</sql></args></api-call>

【示例3 - 多条件组合（AND/OR必须用括号）】：
user: 查看A部门或B部门在2025年的数据
assistant: <api-call><name>response_table</name><args><sql>
SELECT "部门", ROUND(SUM("金额"), 2) AS "总额"
FROM "数据表"
WHERE ("部门" = 'A' OR "部门" = 'B') AND "年份" = 2025
GROUP BY "部门";
</sql></args></api-call>

"""

# ===== 多表模式的示例块 =====
_EXAMPLES_MULTI_TABLE_ZH = """
【示例1 - UNION ALL合并查询】（当需要合并多个结构相似的表时使用）：
user: 所有数据中哪个最大
assistant: <api-call><name>response_table</name><args><sql>
-- 使用 UNION ALL 合并多个表的数据，注意字段对齐
WITH all_data AS (
  SELECT "字段A" AS "统一字段名", "数值字段" AS "数值" FROM "表1"
  UNION ALL
  SELECT "字段B" AS "统一字段名", "数值字段2" AS "数值" FROM "表2"
)
SELECT "统一字段名", ROUND("数值", 2) AS "数值"
FROM all_data
ORDER BY "数值" DESC
LIMIT 1;
</sql></args></api-call>

【示例2 - 单表查询】（当用户明确指定某个表或某类数据时使用）：
user: 查询表A中的最大值
assistant: <api-call><name>response_table</name><args><sql>
SELECT "字段名", ROUND("数值字段", 2) AS "数值"
FROM "表A"
ORDER BY "数值字段" DESC
LIMIT 1;
</sql></args></api-call>

【示例3 - JOIN关联查询】（当需要关联不同类型的表时使用）：
user: 查询数据并关联参考表
assistant: <api-call><name>response_table</name><args><sql>
SELECT a."主字段", ROUND(a."数值" * b."系数", 2) AS "计算结果"
FROM "主表" a
JOIN "参考表" b ON a."关联字段" = b."关联字段";
</sql></args></api-call>

"""

_EXAMPLES_MULTI_TABLE_EN = """
【Example 1 - UNION ALL Merge Query】(Use when merging multiple tables with similar structure):
user: What is the maximum value across all data
assistant: <api-call><name>response_table</name><args><sql>
-- Use UNION ALL to merge data from multiple tables, ensure field alignment
WITH all_data AS (
  SELECT "field_a" AS "unified_field", "value_field" AS "value" FROM "table1"
  UNION ALL
  SELECT "field_b" AS "unified_field", "value_field2" AS "value" FROM "table2"
)
SELECT "unified_field", ROUND("value", 2) AS "value"
FROM all_data
ORDER BY "value" DESC
LIMIT 1;
</sql></args></api-call>

【Example 2 - Single Table Query】(Use when user explicitly specifies a table or data type):
user: Query the maximum value in table A
assistant: <api-call><name>response_table</name><args><sql>
SELECT "field_name", ROUND("value_field", 2) AS "value"
FROM "table_a"
ORDER BY "value_field" DESC
LIMIT 1;
</sql></args></api-call>

【Example 3 - JOIN Related Query】(Use when relating different types of tables):
user: Query data and join with reference table
assistant: <api-call><name>response_table</name><args><sql>
SELECT a."main_field", ROUND(a."value" * b."coefficient", 2) AS "calculated_result"
FROM "main_table" a
JOIN "reference_table" b ON a."join_field" = b."join_field";
</sql></args></api-call>

"""

_EXAMPLES_EN = """
【Example 1 - Basic Aggregation】:
user: Show total sales for department C in 2025
assistant: <api-call><name>response_table</name><args><sql>
SELECT "department", "year", ROUND(SUM("amount"), 2) AS "total"
FROM "data_table"
WHERE "department" = 'C' AND "year" = 2025
GROUP BY "department", "year";
</sql></args></api-call>

【Example 2 - NULL Handling (Must consider NULL when excluding values)】:
user: Show sum of all non-excluded data
assistant: <api-call><name>response_table</name><args><sql>
SELECT ROUND(SUM("amount"), 2) AS "total"
FROM "data_table"
WHERE ("exclude_tag" IS NULL OR "exclude_tag" != 'excluded');
</sql></args></api-call>

【Example 3 - Multi-condition (AND/OR must use parentheses)】:
user: Show data for department A or B in 2025
assistant: <api-call><name>response_table</name><args><sql>
SELECT "department", ROUND(SUM("amount"), 2) AS "total"
FROM "data_table"
WHERE ("department" = 'A' OR "department" = 'B') AND "year" = 2025
GROUP BY "department";
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
def get_prompt_templates_by_language(language: str = "zh", is_multi_table_mode: bool = False):
    """
    根据指定语言返回对应的 prompt 模板

    Args:
        language: "zh" 或 "en"
        is_multi_table_mode: 是否为多表模式

    Returns:
        包含所有模板的字典
    """
    is_english = language == "en"

    # 根据是否多表模式选择约束条件和示例
    if is_multi_table_mode:
        analysis_constraints = (
            _ANALYSIS_CONSTRAINTS_MULTI_TABLE_EN if is_english else _ANALYSIS_CONSTRAINTS_MULTI_TABLE_ZH
        )
        examples = (
            _EXAMPLES_MULTI_TABLE_EN if is_english else _EXAMPLES_MULTI_TABLE_ZH
        )
    else:
        analysis_constraints = (
            _ANALYSIS_CONSTRAINTS_EN if is_english else _ANALYSIS_CONSTRAINTS_ZH
        )
        examples = _EXAMPLES_EN if is_english else _EXAMPLES_ZH

    return {
        "system_prompt": _SYSTEM_PROMPT_EN if is_english else _SYSTEM_PROMPT_ZH,
        "duckdb_rules": _DUCKDB_RULES_EN if is_english else _DUCKDB_RULES_ZH,
        "analysis_constraints": analysis_constraints,
        "examples": examples,
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
