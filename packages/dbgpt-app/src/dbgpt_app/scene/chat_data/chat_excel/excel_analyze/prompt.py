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

_DEFAULT_TEMPLATE_EN = """
The user has a table file data to be analyzed, which has already been imported into a DuckDB table.

A sample of the data is as follows:
``````json
{data_example}
``````

The DuckDB table structure information is as follows:
{table_schema}
{data_time_range}
{query_rewrite_info}
{relevant_columns_info}

For DuckDB, please pay special attention to the following DuckDB syntax rules:
``````markdown
### [MUST FOLLOW] DuckDB Column/Table Name Quoting Rules:

**The following cases MUST use double quotes around field/table names:**
1. Contains Chinese characters (e.g., "客户名称", "地区")
2. Starts with a digit (e.g., "2022_sales", "2023_profit")
3. Contains special characters other than underscore or spaces (e.g., "order ID")

**NEVER use digit-starting field names without double quotes!**

**Correct Examples:**
```sql
-- ✅ Correct: All columns that need quotes are quoted
SELECT "category", SUM("2022_sales") AS "total_sales"
FROM data_analysis_table
GROUP BY "category"
ORDER BY "total_sales" DESC;

-- ✅ Correct: Column names in function arguments are also quoted
SELECT "region", 
       SUM("2022_sales") AS "sales",
       SUM("2022_profit") AS "profit"
FROM data_analysis_table
WHERE "order_date" >= '2022-01-01'
GROUP BY "region";
```

**Wrong Examples:**
```sql
-- ❌ Wrong: Digit-starting column name without quotes
SELECT category, SUM(2022_sales) FROM data_analysis_table;

-- ❌ Wrong: Chinese column name without quotes
SELECT 地区, 客户名称 FROM data_analysis_table;
```

### When using GROUP BY in DuckDB SQL queries, note these key points:

1. Any non-aggregate columns that appear in the SELECT clause must also appear in the GROUP BY clause
2. When referencing a column in ORDER BY or window functions, ensure that column has been properly selected in the preceding CTE or query
3. When building multi-layer CTEs, ensure column reference consistency between layers, especially for columns used in sorting and joining
4. If a column doesn't need an exact value, you can use the ANY_VALUE() function as an alternative

### Key considerations for time series analysis:

1. **Year-over-year analysis**: Requires comparison with the same period last year, SQL must include at least 2 consecutive years of data

2. **Month-over-month analysis**: Requires comparison with the previous period, ensure sufficient time range
   - Month-over-month: Requires at least 2 consecutive months of data
   - Quarter-over-quarter: Requires at least 2 consecutive quarters of data

3. **When using window functions LAG/LEAD**:
   - LAG(value, 1) for month-over-month (previous period)
   - LAG(value, 12) for year-over-year (same month last year)
   - Must ensure data range includes required historical periods
``````

Please answer the user's questions through DuckDB SQL data analysis based on the data structure information provided, while meeting the following constraints.

Constraints:
	1. Please fully understand the user's question and analyze it using DuckDB SQL. Return the analysis content according to the output format required below, with the SQL output in the corresponding SQL parameter
	2. [MANDATORY] All column/table names containing Chinese characters, starting with digits, or containing special characters MUST be wrapped in double quotes! Examples: "category", "2022_sales", "customer_name", etc.
	3. If Query rewrite results are provided above, please prioritize using the rewritten question and strictly follow the "relevant fields" and "analysis suggestions" to generate SQL
	4. [IMPORTANT] Please select the most optimal way from the display methods given below for data rendering, and put the type name in the name parameter value of the required return format.
	**Chart Selection Rules:**
	- If the user explicitly requests "chart", "visualization", "graph", etc., you MUST prioritize chart types (such as response_bar_chart, response_line_chart, response_pie_chart) over response_table
	- For categorical data comparison (e.g., regions, categories), prefer response_bar_chart (bar chart) or response_pie_chart (pie chart)
	- For time series data, prefer response_line_chart (line chart) or response_area_chart (area chart)
	- For proportion/distribution data, prefer response_pie_chart (pie chart) or response_donut_chart (donut chart)
	- Only use response_table when there are many columns (>5) or many non-numeric columns
	Available data display methods are: {display_type}
	5. The table name to be used in the SQL is: {table_name}. Please check your generated SQL and do not use column names that are not in the data structure
	6. Prioritize using data analysis methods to answer. If the user's question does not involve data analysis content, you can answer based on your understanding
    7. DuckDB processes timestamps using dedicated functions (like to_timestamp()) instead of direct CAST
    8. Please note that comment lines should be on a separate line and not on the same line as SQL statements
    9. Do not use UNION or UNION ALL complex SQL syntax. If multiple results are needed, query them separately
	10. Convert the SQL part in the output content to:
	<api-call><name>[display method]</name><args><sql>
	[correct duckdb data analysis sql]</sql></args></api-call>
	format, refer to the return format requirements

**IMPORTANT**: Do not provide specific answers or conclusions before SQL execution! Only provide guiding text and let query results speak for themselves.

Answer format is as follows:
Brief guiding text (no specific answers)<api-call><name>[display method]</name><args>
<sql>[correct duckdb data analysis sql]</sql></args></api-call>

You can refer to the examples below:

Example 1 (Table display):
user: Analyze sales and profit by region, showing region name, total sales, total profit, and average profit margin (profit/sales).
assistant: Analyzing sales performance by region:
<api-call><name>response_table</name><args><sql>
SELECT "region",
       SUM("sales") AS "total_sales",
       SUM("profit") AS "total_profit",
       SUM("profit")/NULLIF(SUM("sales"),0) AS "profit_margin"
FROM data_analysis_table
WHERE "region" IS NOT NULL
GROUP BY "region"
ORDER BY "total_sales" DESC;
</sql></args></api-call>

Example 2 (Chart display - user explicitly requests chart):
user: Analyze sales and profit by region, grouped by region dimension, calculate total sales, total profit and average profit margin (profit/sales) for each region, and display the results in chart form.
assistant: Analyzing sales performance by region:
<api-call><name>response_bar_chart</name><args><sql>
SELECT 
    "region" AS "region_name",
    SUM(CAST("sales" AS FLOAT)) AS "total_sales",
    SUM(CAST("profit" AS FLOAT)) AS "total_profit",
    SUM(CAST("profit" AS FLOAT)) / NULLIF(SUM(CAST("sales" AS FLOAT)), 0) AS "average_profit_margin"
FROM data_analysis_table
WHERE "region" IS NOT NULL
GROUP BY "region"
ORDER BY "total_sales" DESC;
</sql></args></api-call>

Example 3 (Categorical comparison chart):
user: Show sales proportion by product category.
assistant: Analyzing sales proportion by product category:
<api-call><name>response_pie_chart</name><args><sql>
SELECT 
    "category" AS "category",
    SUM(CAST("sales" AS FLOAT)) AS "sales"
FROM data_analysis_table
WHERE "category" IS NOT NULL
GROUP BY "category"
ORDER BY "sales" DESC;
</sql></args></api-call>

Note that the answer must conform to the <api-call> format! Please answer in the same language as the user's question!
**Remember**: Do not provide specific data results in advance, because you haven't executed SQL yet and don't know the actual results!
User question: {user_input}
"""

# ===== SYSTEM PROMPT（简洁、稳定、不变化）=====
_SYSTEM_PROMPT_ZH = """你是一个专业的数据分析专家。

你的职责是理解用户的数据分析需求，并生成正确的 DuckDB SQL 查询。"""

_SYSTEM_PROMPT_EN = """You are a professional data analysis expert.

Your responsibility is to understand the user's data analysis requirements and generate correct DuckDB SQL queries."""

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

### 【CTE Alias and WHERE Rules - Critical】：
**DuckDB WHERE clause must be processed before SELECT column definitions, cannot reference aliases defined in SELECT**

Wrong 1:
```sql
WITH cte AS (
    SELECT "original" AS "new"
    FROM table
    WHERE "original" IS NOT NULL  -- Wrong!
)
```

Wrong 2:
```sql
WITH cte1 AS (SELECT "col" AS "new_col" FROM t),
cte2 AS (
    SELECT *
    FROM cte1
    WHERE "new_col" IS NOT NULL  -- Wrong! WHERE before alias definition
)
```

Correct:
```sql
WITH cte1 AS (SELECT "col" AS "new_col" FROM t)
SELECT * FROM cte1 WHERE "new_col" IS NOT NULL  -- WHERE in outermost query
```

Or use subquery:
```sql
SELECT * FROM (
    SELECT "original" AS "new" FROM table
) WHERE "new" IS NOT NULL
```

### 【Time Series Analysis】：
- **Year-over-year**: Requires at least 2 years of continuous data
- **Month-over-month**: Requires sufficient historical data periods
- **LAG Function**: LAG(value, 1) for month-over-month, LAG(value, 12) for year-over-year
"""

# ===== 可复用的约束条件块 =====
_ANALYSIS_CONSTRAINTS_ZH = """
表名：{table_name}
列名规则：中文/数字开头/特殊字符必须用双引号
图表选择：分类对比用bar/pie，时序用line/area，多列(>5)用table
可用方式：{display_type}
"""

_ANALYSIS_CONSTRAINTS_EN = """
Table: {table_name}
Column rules: Chinese/digit-starting/special chars need double quotes
Chart selection: categorical use bar/pie, time-series use line/area, many columns(>5) use table
Available types: {display_type}
"""

# ===== 可复用的示例块 =====
_EXAMPLES_ZH = """
【示例】：
user: 分析各地区销售
assistant: 为您分析各地区销售：
<api-call><name>response_bar_chart</name><args><sql>
SELECT "地区", SUM("销售额") AS "总销售额"
FROM data_analysis_table
WHERE "地区" IS NOT NULL
GROUP BY "地区"
ORDER BY "总销售额" DESC;
</sql></args></api-call>
"""

_EXAMPLES_EN = """
【Example】：
user: Analyze sales by region
assistant: Analyzing sales by region:
<api-call><name>response_bar_chart</name><args><sql>
SELECT "region", SUM("sales") AS "total_sales"
FROM data_analysis_table
WHERE "region" IS NOT NULL
GROUP BY "region"
ORDER BY "total_sales" DESC;
</sql></args></api-call>
"""

# ===== USER PROMPT TEMPLATE（动态生成，每次查询都变化）=====
_USER_PROMPT_TEMPLATE_ZH = """
请按照以下要求回答用户问题：

【输出要求】：
1. 简短引导文本(1-2句话)
2. <api-call></api-call>代码块(包含SQL)
3. 禁止：冗长说明、多个api-call、提前给结论、在api-call外输出SQL

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
1. Brief guide text (1-2 sentences)
2. <api-call></api-call> block (with SQL)
3. Prohibited: lengthy explanations, multiple api-calls, premature conclusions, SQL outside api-call

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

# ===== 选择合适的模板 =====
_SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_EN if CFG.LANGUAGE == "en" else _SYSTEM_PROMPT_ZH
)

_DUCKDB_RULES = (
    _DUCKDB_RULES_EN if CFG.LANGUAGE == "en" else _DUCKDB_RULES_ZH
)

_ANALYSIS_CONSTRAINTS_TEMPLATE = (
    _ANALYSIS_CONSTRAINTS_EN if CFG.LANGUAGE == "en" else _ANALYSIS_CONSTRAINTS_ZH
)

_EXAMPLES = (
    _EXAMPLES_EN if CFG.LANGUAGE == "en" else _EXAMPLES_ZH
)

_USER_PROMPT_TEMPLATE = (
    _USER_PROMPT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _USER_PROMPT_TEMPLATE_ZH
)

# ===== 保留旧的模板变量以保持兼容 =====
_DEFAULT_TEMPLATE_ZH = """
用户有一份待分析表格文件数据，目前已经导入到 DuckDB 表中，\

一部分采样数据如下:
``````json
{data_example}
``````

DuckDB 表结构信息如下：
{table_schema}
{data_time_range}
{query_rewrite_info}
{relevant_columns_info}

DuckDB 中，需要特别注意的 DuckDB 语法规则：
``````markdown
### 【必须遵守】DuckDB 列名/表名引号规则：

**以下情况必须使用双引号包裹字段名或表名：**
1. 包含中文字符（如 "客户名称"、"地区"）
2. 以数字开头（如 "2022_销售额"、"2023_利润"）
3. 包含下划线以外的特殊字符或空格（如 "订单 ID"）

**严禁使用未加双引号的数字开头字段名！**

**正确示例：**
```sql
SELECT "细分", SUM("2022_销售额") AS "总销售额"
FROM data_analysis_table
GROUP BY "细分"
ORDER BY "总销售额" DESC;

### 在 DuckDB SQL 查询中使用 GROUP BY 时需要注意以下关键点：

1. 任何出现在 SELECT 子句中的非聚合列，必须同时出现在 GROUP BY 子句中
2. 当在 ORDER BY 或窗口函数中引用某个列时，确保该列已在前面的 CTE 或查询中被正确选择
3. 在构建多层 CTE 时，需要确保各层之间的列引用一致性，特别是用于排序和连接的列
4. 如果某列不需要精确值，可以使用 ANY_VALUE() 函数作为替代方案

### 时间序列分析的关键注意事项：

1. **同比分析**：需要对比去年同期数据，SQL必须包含至少连续两年的数据

2. **环比分析**：需要对比上一个周期数据，确保时间范围足够
   - 月环比：至少需要连续两个月数据
   - 季度环比：至少需要连续两个季度数据

3. **使用窗口函数LAG/LEAD时**：
   - LAG(value, 1) 用于环比（上一期）
   - LAG(value, 12) 用于同比（去年同月）
   - 必须确保数据范围包含所需的历史期间
``````

请基于给你的数据结构信息，在满足下面约束条件下通过\
DuckDB SQL数据分析回答用户的问题。
约束条件:
	1.请充分理解用户的问题，使用 DuckDB SQL 的方式进行分析，\
	分析内容按下面要求的输出格式返回，SQL 请输出在对应的 SQL 参数中
	2.【强制要求】所有包含中文、以数字开头、或包含特殊字符的列名/表名，必须使用双引号包裹！\
	3.如果上面提供了Query改写结果，请优先使用改写后的问题，并严格按照"相关字段"和"分析建议"来生成SQL
	4.【重要】请从如下给出的展示方式种选择最优的一种用以进行数据渲染，\
	将类型名称放入返回要求格式的name参数值中。\
	可用数据展示方式如下: {display_type}
	5.SQL中需要使用的表名是: {table_name},请检查你生成的sql，\
	不要使用没在数据结构中的列名
	6.优先使用数据分析的方式回答，如果用户问题不涉及数据分析内容，你可以按你的理解进行回答
    7.DuckDB 处理时间戳需通过专用函数（如 to_timestamp()）而非直接 CAST
    8.请注意，注释行要单独一行，不要放在 SQL 语句的同一行中
    9.不要使用UNION、UNION ALL等复杂SQL语法，如果需要查询多个结果，请分别查询
	
	
**重要**：不要在SQL执行前给出具体的答案或结论！只提供引导性文本，让查询结果自己说话。输出内容中sql部分转换为：
	<api-call><name>[数据显示方式]</name><args><sql>\
	[正确的duckdb数据分析sql]</sql></args></api-call> \
	这样的格式，参考返回格式要求

回答格式如下:
简短的引导性文本（不包含具体答案）<api-call><name>[数据展示方式]</name><args>\
<sql>[正确的duckdb数据分析sql]</sql></args></api-call>

你可以参考下面的样例:

例子1（表格展示）：
user: 分析各地区的销售额和利润，需要显示地区名称、总销售额、\
总利润以及平均利润率（利润/销售额）。
assistant: 为您分析各地区的销售表现：
<api-call><name>response_table</name><args><sql>
SELECT "地区",
       SUM("销售额") AS "总销售额",
       SUM("利润") AS "总利润",
       SUM("利润")/NULLIF(SUM("销售额"),0) AS "利润率"
FROM data_analysis_table
WHERE "地区" IS NOT NULL
GROUP BY "地区"
ORDER BY "总销售额" DESC;
</sql></args></api-call>

例子2（图表展示 - 用户明确要求图表）：
user: 分析各地区的销售额和利润，按区域维度分组，计算每个地区的总销售额、总利润以及平均利润率（利润/销售额），并以图表形式展示结果。
assistant: 为您分析各地区的销售表现：
<api-call><name>response_bar_chart</name><args><sql>
SELECT 
    "区域_区域" AS "地区名称",
    SUM(CAST("指标信息_销售额" AS FLOAT)) AS "总销售额",
    SUM(CAST("指标信息_利润" AS FLOAT)) AS "总利润",
    SUM(CAST("指标信息_利润" AS FLOAT)) / NULLIF(SUM(CAST("指标信息_销售额" AS FLOAT)), 0) AS "平均利润率"
FROM data_analysis_table
WHERE "区域_区域" IS NOT NULL
GROUP BY "区域_区域"
ORDER BY "总销售额" DESC;
</sql></args></api-call>

例子3（分类对比图表）：
user: 展示各产品类别的销售额占比。
assistant: 为您分析各产品类别的销售额占比：
<api-call><name>response_pie_chart</name><args><sql>
SELECT 
    "产品信息_类别" AS "类别",
    SUM(CAST("指标信息_销售额" AS FLOAT)) AS "销售额"
FROM data_analysis_table
WHERE "产品信息_类别" IS NOT NULL
GROUP BY "产品信息_类别"
ORDER BY "销售额" DESC;
</sql></args></api-call>

注意，回答一定要符合 <api-call> 的格式! 请使用和用户问题相同的语言回答！
**切记**：不要提前给出具体的数据结果，因为你还没有执行SQL，不知道真实结果！
"""


_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

PROMPT_NEED_STREAM_OUT = False

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
