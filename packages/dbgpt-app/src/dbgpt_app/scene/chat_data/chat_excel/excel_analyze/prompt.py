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

**【极其重要】表名必须完全匹配：**
- **表名必须使用约束条件中指定的完整表名，不能简化、缩写或修改**
- **在所有SQL语句的FROM子句中，必须使用约束条件中明确指定的表名**

**【极其重要】列名必须完全匹配：**
- 必须使用表结构中提供的**完整列名**，包括所有括号、连字符等特殊字符
- 列名必须**逐字符精确匹配**表结构中的定义
- **字符串匹配必须精确**：在WHERE子句中使用字符串条件时，必须完全匹配数据中的实际值，不能随意替换字符（如"和"不能替换为"与"，"部门"不能替换为"部"等）

### 【字符串匹配精确规则】：
- **必须精确匹配**：在WHERE子句中使用字符串条件时，必须完全匹配数据中的实际值
- **禁止字符替换**：不能随意替换字符
- **建议做法**：如果不确定精确值，可以使用LIKE或IN操作符，但必须基于实际数据中的值

### 【WHERE子句逻辑优先级】：
**【极其重要】混用AND和OR时必须用括号明确优先级：**
- ❌ 错误：`WHERE "部门"='A' OR "部门"='B' AND "年份"=2025`（逻辑不清晰，易误解）
- ✅ 正确：`WHERE ("部门"='A' OR "部门"='B') AND "年份"=2025`（先判断部门，再判断年份）
- ✅ 正确：`WHERE "部门"='A' OR ("部门"='B' AND "年份"=2025)`（A部门全部 或 B部门2025年）
**关键原则：根据用户意图和对话逻辑，用括号明确表达条件的优先级和组合关系**

### 【GROUP BY 关键规则】：
1. SELECT 中的非聚合列必须在 GROUP BY 中
2. ORDER BY 中的列必须在前面的 CTE 或查询中被正确选择
3. 多层 CTE 中各层之间的列引用必须一致，特别是排序和连接的列
**DuckDB的WHERE子句必须在SELECT列定义之前处理,不能引用SELECT中定义的别名**

### 【CTE和别名规则】：
**【极其重要】同一SELECT中不能引用本SELECT定义的别名：**
**CTE字段传递：后续需要的字段必须在CTE的SELECT中明确列出**
**⚠️ 关键：CTE名称、表别名、子查询别名等SQL标识符必须使用英文或拼音，禁止使用中文！**
  - ✅ 正确示例：`WITH haiwai_data AS (...)`、`AS hj_table`
  - ❌ 错误示例：`WITH海外数据 AS (...)`、`AS 合计表`
**人均计算：使用 NULLIF(分母, 0) 避免除零，如 ROUND(SUM("金额")/NULLIF(COUNT(*), 0), 2)**

### 【子查询使用规则】：
- **禁止在 SELECT 列表中使用返回多行的子查询**（会导致"More than one row returned"错误）
- **正确做法**：使用 JOIN 或窗口函数替代，如需子查询必须与主查询关联（WHERE 条件关联）
- **示例**：计算比例应使用 `ROUND(SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)` 而非子查询

### 【时间序列分析】：
- **同比分析**：需要至少连续两年的数据
- **环比分析**：需要足够的历史数据周期
- **使用 LAG 函数**：LAG(value, 1) 用于环比，LAG(value, 12) 用于同比

### 【数值格式化规则】：
- **所有数值列必须保留两位小数**：在 SELECT 语句中，所有数值类型的列（包括聚合函数结果）都必须使用 ROUND() 函数保留两位小数
- **格式示例**：
  - 原始列：`SELECT "销售额"` → 应改为：`SELECT ROUND("销售额", 2) AS "销售额"`
  - 聚合函数：`SELECT SUM("金额") AS "总金额"` → 应改为：`SELECT ROUND(SUM("金额"), 2) AS "总金额"`
  - 计算列：`SELECT "单价" * "数量" AS "小计"` → 应改为：`SELECT ROUND("单价" * "数量", 2) AS "小计"`
- **必须对所有数值结果应用 ROUND(column, 2)**，确保输出结果统一保留两位小数

### 【LIMIT 使用规则】：
- **禁止自动添加 LIMIT**：除非用户明确要求限制返回行数（如"只显示前10条"、"显示前5名"等），否则**严禁**在 SQL 查询中使用 LIMIT 子句
- **默认展示所有数据**：应该尽可能展示所有符合查询条件的分析数据，让用户看到完整的结果
- **仅在用户明确要求时使用**：只有当用户明确表达需要限制结果数量时（如"前N条"、"只显示N个"、"限制为N条"等），才可以使用 LIMIT
"""

_DUCKDB_RULES_EN = """
### 【MUST FOLLOW】DuckDB Column/Table Name Quoting Rules:

**The following cases MUST use double quotes around field/table names:**
1. Contains Chinese characters (e.g., "客户名称", "地区")
2. Starts with a digit (e.g., "2022_sales", "2023_profit")
3. Contains special characters other than underscore or spaces (e.g., "order ID")

**NEVER use digit-starting field names without double quotes!**

**【CRITICAL】Table Name Must Match Exactly:**
- **Table name MUST use the complete table name specified in the constraints, cannot be simplified, abbreviated, or modified**
- **In all SQL FROM clauses, you MUST use the exact table name specified in the constraints**

**【CRITICAL】Column Name Must Match Exactly:**
- Must use **complete column names** from table structure, including all parentheses, hyphens and special characters
- Column names must **match character-by-character** the definitions in table structure
- **String matching must be exact**: When using string conditions in WHERE clause, must completely match actual values in data, cannot arbitrarily replace characters (e.g., "and" cannot be replaced with "&", "dept" cannot be abbreviated)

### 【String Matching Precision Rules】：
- **Must match exactly**: When using string conditions in WHERE clause, must exactly match the actual values in the data
- **No character substitution**: Cannot arbitrarily replace characters
- **Recommended approach**: If unsure of exact value, use LIKE or IN operators, but must be based on actual values in the data

### 【WHERE Clause Logic Priority】：
**【CRITICAL】Must use parentheses to clarify priority when mixing AND/OR:**
**Key principle: Use parentheses to clearly express condition priority and combination based on user intent and conversation logic**

### 【GROUP BY Key Rules】：
1. Non-aggregate columns in SELECT must be in GROUP BY
2. Columns in ORDER BY must be properly selected in preceding CTE or query
3. Ensure column reference consistency between CTE layers, especially for columns used in sorting and joining
**DuckDB WHERE clause is processed before SELECT column definitions, cannot reference aliases defined in SELECT**

### 【CTE and Alias Rules】：
**【CRITICAL】Cannot reference aliases within same SELECT:**
**CTE field passing: Fields needed subsequently must be explicitly listed in CTE SELECT**
**⚠️ Important: CTE names, table aliases, subquery aliases and other SQL identifiers MUST use English or Pinyin, Chinese characters are FORBIDDEN!**
  - ✅ Correct: `WITH haiwai_data AS (...)`、`AS summary_table`
  - ❌ Wrong: `WITH海外数据 AS (...)`、`AS 合计表`
**Per-capita calculation: Use NULLIF(denominator, 0) to avoid division by zero, e.g. ROUND(SUM("amt")/NULLIF(COUNT(*), 0), 2)**

### 【Subquery Usage Rules】：
- **NEVER use subqueries in SELECT list that return multiple rows** (causes "More than one row returned" error)
- **Correct approach**: Use JOIN or window functions instead, if subquery needed must correlate with main query (WHERE condition)
- **Example**: Calculate percentage use `ROUND(SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)` instead of subquery

### 【Time Series Analysis】：
- **Year-over-year**: Requires at least 2 years of continuous data
- **Month-over-month**: Requires sufficient historical data periods
- **LAG Function**: LAG(value, 1) for month-over-month, LAG(value, 12) for year-over-year

### 【Numeric Formatting Rules】：
- **All numeric columns must retain 2 decimal places**: In SELECT statements, all numeric columns (including aggregate function results) must use ROUND() function to retain 2 decimal places
- **Format examples**:
  - Original column: `SELECT "sales"` → Should be: `SELECT ROUND("sales", 2) AS "sales"`
  - Aggregate function: `SELECT SUM("amount") AS "total"` → Should be: `SELECT ROUND(SUM("amount"), 2) AS "total"`
  - Calculated column: `SELECT "price" * "quantity" AS "subtotal"` → Should be: `SELECT ROUND("price" * "quantity", 2) AS "subtotal"`
- **Must apply ROUND(column, 2) to all numeric results** to ensure output consistently retains 2 decimal places

### 【LIMIT Usage Rules】：
- **DO NOT automatically add LIMIT**: Unless the user explicitly requests to limit the number of returned rows (e.g., "show only top 10", "display top 5", etc.), you **MUST NOT** use LIMIT clause in SQL queries
- **Display all data by default**: Should display all analysis data that meets the query conditions as much as possible, allowing users to see complete results
- **Use LIMIT only when user explicitly requests**: Only use LIMIT when the user clearly expresses the need to limit result count (e.g., "top N", "only show N", "limit to N rows", etc.)
"""

# ===== 可复用的约束条件块 =====
_ANALYSIS_CONSTRAINTS_ZH = """
**【必须严格遵守】表名：{table_name}**
**重要提示：在所有SQL语句的FROM子句中，必须使用上面指定的完整表名，不能简化、缩写或修改！**

列名规则：中文/数字开头/特殊字符必须用双引号;不要使用 UNION / UNION ALL，如需多个结果请分别查询；时间戳处理使用 to_timestamp() 而非直接 CAST；注释行必须单独成行，不要放在 SQL 语句的同一行
字符串匹配规则：WHERE子句中的字符串条件必须完全匹配数据中的实际值，不能随意替换字符（如"和"不能替换为"与"），建议使用LIKE或IN操作符基于实际数据值
**WHERE逻辑规则：混用AND和OR时必须用括号明确优先级，根据用户意图正确组合条件**
子查询规则：禁止在 SELECT 列表中使用返回多行的子查询（会导致"More than one row returned"错误），应使用 JOIN 或窗口函数替代
**别名规则：同一SELECT中不能引用本层定义的别名，使用完整表达式或多层CTE；CTE中必须列出所有后续需要的字段**
数值格式化：所有数值列和聚合结果必须使用 ROUND(column, 2) 保留两位小数
**LIMIT规则：禁止自动添加LIMIT，除非用户明确要求限制返回行数，否则应展示所有符合条件的数据**
图表优先：默认使用图表，分类对比用bar/pie，时序用line/area，仅明细记录用table
可用方式：{display_type}
展示顺序：数据摘要 → 图表可视化 → SQL查询
"""

# ===== 多表模式的约束条件块 =====
_ANALYSIS_CONSTRAINTS_MULTI_TABLE_ZH = """
**【多表模式】可用的数据表：{table_names}**

**【重要】多表查询策略：**
1. **UNION ALL 合并查询**（推荐用于结构相似的表）：
   - 当用户询问"所有"、"全部"、"总共"等需要合并多个表数据时
   - 使用 UNION ALL 将多个表的数据合并后再进行聚合分析
   - ⚠️ **关键注意事项**：
     * 不同表的字段名可能不同，需要使用 AS 统一别名
     * **如果某个字段只存在于部分表中，对于不存在该字段的表，必须使用 NULL 或 0 填充**
     * 必须确保 UNION ALL 的每个 SELECT 语句返回相同数量和类型的列
   - 示例（包含NULL填充）：
     ```sql
     SELECT "字段A" AS "统一名", "数值字段" AS "数值" FROM "表1"
     UNION ALL
     SELECT "字段B" AS "统一名", NULL AS "数值" FROM "表2"  -- 表2没有"数值字段"，用NULL填充
     ```

2. **单表查询**：
   - 如果用户明确指定了某个表或某类数据，只查询对应的表
   - 根据表名和表描述判断用户想查询哪个表

3. **JOIN 关联查询**：
   - 当需要关联不同类型的表时
   - 使用适当的 JOIN 条件连接表

**表名必须使用上面列出的完整表名，不能简化、缩写或修改**
**字段名必须严格使用建表SQL中的字段名，不能使用不存在的字段**

列名规则：中文/数字开头/特殊字符必须用双引号;时间戳处理使用 to_timestamp() 而非直接 CAST；注释行必须单独成行，不要放在 SQL 语句的同一行
字符串匹配规则：WHERE子句中的字符串条件必须完全匹配数据中的实际值，不能随意替换字符（如"和"不能替换为"与"），建议使用LIKE或IN操作符基于实际数据值
**WHERE逻辑规则：混用AND和OR时必须用括号明确优先级，根据用户意图正确组合条件**
子查询规则：禁止在 SELECT 列表中使用返回多行的子查询（会导致"More than one row returned"错误），应使用 JOIN 或窗口函数替代
**别名规则：同一SELECT中不能引用本层定义的别名，使用完整表达式或多层CTE；CTE中必须列出所有后续需要的字段**
**⚠️ SQL标识符规则：CTE名称、表别名、子查询别名等必须使用英文或拼音，禁止使用中文（会导致语法错误）**
数值格式化：所有数值列和聚合结果必须使用 ROUND(column, 2) 保留两位小数
**LIMIT规则：禁止自动添加LIMIT，除非用户明确要求限制返回行数，否则应展示所有符合条件的数据**
图表优先：默认使用图表，分类对比用bar/pie，时序用line/area，仅明细记录用table
可用方式：{display_type}
展示顺序：数据摘要 → 图表可视化 → SQL查询
"""

_ANALYSIS_CONSTRAINTS_MULTI_TABLE_EN = """
**【Multi-Table Mode】Available Tables: {table_names}**

**【IMPORTANT】Multi-Table Query Strategies:**
1. **UNION ALL Merge Query** (Recommended for similar-structured tables):
   - When user asks about "all", "total", "overall" data requiring data from multiple tables
   - Use UNION ALL to merge data from multiple tables before aggregation analysis
   - ⚠️ **Critical Notes**:
     * Different tables may have different column names, use AS to unify aliases
     * **If a field exists only in some tables, you MUST use NULL or 0 to fill in for tables that don't have that field**
     * Ensure each SELECT in UNION ALL returns the same number and types of columns
   - Example (with NULL filling):
     ```sql
     SELECT "field_a" AS "unified_name", "value_field" AS "value" FROM "table1"
     UNION ALL
     SELECT "field_b" AS "unified_name", NULL AS "value" FROM "table2"  -- table2 doesn't have "value_field", use NULL
     ```

2. **Single Table Query**:
   - If user explicitly specifies a particular table or data type, only query the corresponding table
   - Determine which table to query based on table name and description

3. **JOIN Related Query**:
   - When relating different types of tables
   - Use appropriate JOIN conditions to connect tables

**Table names MUST use the complete names listed above, cannot simplify, abbreviate, or modify**
**Column names MUST strictly use the field names from CREATE TABLE SQL, cannot use non-existent fields**

Column rules: Chinese/digit-starting/special chars need double quotes
String matching rules: String conditions in WHERE clause must exactly match actual values in data
**WHERE logic rules: Must use parentheses to clarify priority when mixing AND/OR**
Subquery rules: NEVER use subqueries in SELECT list that return multiple rows
**Alias rules: Cannot reference aliases within same SELECT, use full expressions or multi-layer CTEs**
Numeric formatting: All numeric columns and aggregate results must use ROUND(column, 2)
**LIMIT rules: DO NOT automatically add LIMIT unless user explicitly requests**
Chart priority: Default to charts, categorical use bar/pie, time-series use line/area
Available types: {display_type}
Display order: Data summary → Chart visualization → SQL query
"""

_ANALYSIS_CONSTRAINTS_EN = """
**【MUST STRICTLY FOLLOW】Table Name: {table_name}**
**IMPORTANT: In all SQL FROM clauses, you MUST use the complete table name specified above, cannot simplify, abbreviate, or modify it!**

Column rules: Chinese/digit-starting/special chars need double quotes
String matching rules: String conditions in WHERE clause must exactly match actual values in data, cannot arbitrarily replace characters (e.g., "和" cannot be replaced with "与"), recommend using LIKE or IN operators based on actual data values
**WHERE logic rules: Must use parentheses to clarify priority when mixing AND/OR, correctly combine conditions based on user intent**
Subquery rules: NEVER use subqueries in SELECT list that return multiple rows (causes "More than one row returned" error), use JOIN or window functions instead
**Alias rules: Cannot reference aliases within same SELECT, use full expressions or multi-layer CTEs; CTE must list all fields needed subsequently**
**⚠️ SQL identifier rules: CTE names, table aliases, subquery aliases MUST use English or Pinyin, Chinese characters are FORBIDDEN (will cause syntax errors)**
Numeric formatting: All numeric columns and aggregate results must use ROUND(column, 2) to retain 2 decimal places
**LIMIT rules: DO NOT automatically add LIMIT unless user explicitly requests to limit returned rows, otherwise display all data that meets conditions**
Chart priority: Default to charts, categorical use bar/pie, time-series use line/area, only detailed records use table
Available types: {display_type}
Display order: Data summary → Chart visualization → SQL query
"""

# ===== 可复用的示例块 =====
_EXAMPLES_ZH = """
【示例】：
user: 查看非空部门和C部门2025年的数据
assistant: <api-call><name>response_table</name><args><sql>
SELECT "部门", "年份", ROUND(SUM("金额"), 2) AS "总额"
FROM data_analysis_table
WHERE (("部门"=null OR "部门"='') AND  "部门"='C'
GROUP BY "部门", "年份"
ORDER BY "部门", "年份";
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
【Example】：
user: Show data for non-empty departments and C departments in 2025
assistant: <api-call><name>response_table</name><args><sql>
SELECT "部门", "年份", ROUND(SUM("金额"), 2) AS "总额"
FROM data_analysis_table
WHERE (("部门"=null OR "部门"='') AND  "部门"='C'
GROUP BY "部门", "年份"
ORDER BY "部门", "年份";
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
