import io
import logging
import os
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import chardet
import duckdb
import numpy as np
import pandas as pd
import sqlparse

from dbgpt.util.file_client import FileClient
from dbgpt.util.pd_utils import csv_colunm_foramt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


class TransformedExcelResponse(NamedTuple):
    description: str
    columns: List[Dict[str, str]]
    plans: List[str]


def excel_colunm_format(old_name: str) -> str:
    new_column = old_name.strip()
    new_column = new_column.replace(" ", "_")
    return new_column


def add_quotes_to_chinese_columns(sql, column_names=[]):
    """
    为SQL中需要加引号的列名添加引号
    
    通用规则：
    1. 包含中文字符
    2. 以数字开头
    3. 是SQL关键字但被用作列名
    
    使用正则表达式方式，更可靠和通用
    """
    import re
    
    # SQL关键字列表（常见的）
    SQL_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'BY', 'AS', 'JOIN', 
        'ON', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL',
        'INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS', 'UNION', 'ALL',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
        'SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'ROW_NUMBER', 'PARTITION', 'OVER', 'DISTINCT', 'HAVING', 'LIMIT', 'OFFSET',
        'WITH', 'CTE', 'DESC', 'ASC', 'RN'
    }
    
    def needs_quotes(identifier):
        """判断标识符是否需要加引号"""
        if not identifier or identifier.startswith('"'):
            return False
        
        # 已经加了引号的，跳过
        if identifier.startswith('"') and identifier.endswith('"'):
            return False
        
        # 数字、字符串字面量、函数调用，跳过
        if identifier.isdigit() or identifier.startswith("'"):
            return False
        
        # SQL关键字（大小写不敏感），跳过
        if identifier.upper() in SQL_KEYWORDS:
            return False
        
        # 规则1: 包含中文
        if any('\u4e00' <= char <= '\u9fa5' for char in identifier):
            return True
        
        # 规则2: 以数字开头
        if identifier[0].isdigit():
            return True
        
        return False
    
    # 1. 先处理 AS 别名（特殊处理，因为它们可能以数字开头）
    as_pattern = r'\bAS\s+([a-zA-Z_\u4e00-\u9fa5][a-zA-Z0-9_\u4e00-\u9fa5]*|\d[a-zA-Z0-9_\u4e00-\u9fa5]+)'
    sql = re.sub(as_pattern, lambda m: f'AS "{m.group(1)}"' if needs_quotes(m.group(1)) else m.group(0), sql, flags=re.IGNORECASE)
    
    # 2. 处理点号前后的标识符（表名.列名）
    dot_pattern = r'([a-zA-Z_\u4e00-\u9fa5][a-zA-Z0-9_\u4e00-\u9fa5]*|\d[a-zA-Z0-9_\u4e00-\u9fa5]+)\.([a-zA-Z_\u4e00-\u9fa5][a-zA-Z0-9_\u4e00-\u9fa5]*|\d[a-zA-Z0-9_\u4e00-\u9fa5]+)'
    
    def replace_dot_notation(match):
        table_or_alias = match.group(1)
        column = match.group(2)
        
        # 表名/别名加引号
        if needs_quotes(table_or_alias):
            table_or_alias = f'"{table_or_alias}"'
        
        # 列名加引号
        if needs_quotes(column):
            column = f'"{column}"'
        
        return f'{table_or_alias}.{column}'
    
    sql = re.sub(dot_pattern, replace_dot_notation, sql)
    
    # 3. ✅ 关键修复：处理所有独立的标识符（包括函数参数内的列名）
    # 匹配标识符：字母/数字/下划线/中文组成，后面不是左括号（避免匹配函数名）
    # 同时支持以数字开头的列名（如 2022_销售额）
    identifier_pattern = r'(?<!")(?<!\w)([a-zA-Z_\u4e00-\u9fa5][\w\u4e00-\u9fa5]*|\d[\w\u4e00-\u9fa5]+)(?!\()(?!")'
    
    def replace_identifier(match):
        identifier = match.group(1)
        if needs_quotes(identifier):
            return f'"{identifier}"'
        return identifier
    
    # 分段处理SQL，避免在字符串内部替换
    result_parts = []
    in_string = False
    string_char = None
    i = 0
    current_segment = []
    
    while i < len(sql):
        char = sql[i]
        
        # 检测字符串边界
        if char in ("'", '"') and (i == 0 or sql[i-1] != '\\'):
            if not in_string:
                # 进入字符串前，处理之前积累的非字符串部分
                if current_segment:
                    segment_sql = ''.join(current_segment)
                    segment_sql = re.sub(identifier_pattern, replace_identifier, segment_sql)
                    result_parts.append(segment_sql)
                    current_segment = []
                
                in_string = True
                string_char = char
                result_parts.append(char)
            elif char == string_char:
                # 退出字符串
                in_string = False
                string_char = None
                result_parts.append(char)
            else:
                result_parts.append(char)
        elif in_string:
            result_parts.append(char)
        else:
            current_segment.append(char)
        
        i += 1
    
    # 处理最后的非字符串部分
    if current_segment:
        segment_sql = ''.join(current_segment)
        segment_sql = re.sub(identifier_pattern, replace_identifier, segment_sql)
        result_parts.append(segment_sql)
    
    result_sql = ''.join(result_parts)
    
    return result_sql


def read_from_df(
    db: "DuckDBPyConnection",
    file_path,
    file_name: str,
    table_name: str,
):
    file_client = FileClient()
    file_info = file_client.read_file(conv_uid=None, file_key=file_path)

    result = chardet.detect(file_info)
    encoding = result["encoding"]
    confidence = result["confidence"]

    logger.info(
        f"File Info:{len(file_info)},Detected Encoding: {encoding} "
        f"(Confidence: {confidence})"
    )
    # read excel file
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df_tmp = pd.read_excel(file_info, index_col=False)
        df = pd.read_excel(
            file_info,
            index_col=False,
            converters={i: csv_colunm_foramt for i in range(df_tmp.shape[1])},
        )
    elif file_name.endswith(".csv"):
        df_tmp = pd.read_csv(
            file_info if isinstance(file_info, str) else io.BytesIO(file_info),
            index_col=False,
            encoding=encoding,
        )
        df = pd.read_csv(
            file_info if isinstance(file_info, str) else io.BytesIO(file_info),
            index_col=False,
            encoding=encoding,
            converters={i: csv_colunm_foramt for i in range(df_tmp.shape[1])},
        )
    else:
        raise ValueError("Unsupported file format.")

    df.replace("", np.nan, inplace=True)

    unnamed_columns_tmp = [
        col
        for col in df_tmp.columns
        if col.startswith("Unnamed") and df_tmp[col].isnull().all()
    ]
    df_tmp.drop(columns=unnamed_columns_tmp, inplace=True)

    df = df[df_tmp.columns.values]

    columns_map = {}
    for column_name in df_tmp.columns:
        df[column_name] = df[column_name].astype(str)
        columns_map.update({column_name: excel_colunm_format(column_name)})
        try:
            df[column_name] = pd.to_datetime(df[column_name]).dt.strftime("%Y-%m-%d")
        except ValueError:
            try:
                df[column_name] = pd.to_numeric(df[column_name])
            except ValueError:
                try:
                    df[column_name] = df[column_name].astype(str)
                except Exception:
                    print("Can't transform column: " + column_name)

    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    # write data in duckdb
    db.register("temp_df_table", df)
    # The table is explicitly created due to the issue at
    # https://github.com/eosphoros-ai/DB-GPT/issues/2437.
    db.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df_table")
    return table_name


def read_direct(
    db: "DuckDBPyConnection",
    file_path: str,
    file_name: str,
    table_name: str,
):
    try:
        # Try to import data automatically, It will automatically detect from the file
        # extension
        db.sql(f"create table {table_name} as SELECT * FROM '{file_path}'")
        return
    except Exception as e:
        logger.warning(f"Error while reading file: {str(e)}")
    file_extension = os.path.splitext(file_path)[1]
    load_params = {}
    if file_extension == ".csv":
        load_func = "read_csv"
        load_params = {}
    elif file_extension == ".xlsx":
        # 尝试使用DuckDB的read_xlsx，如果失败则fallback到pandas
        load_func = "read_xlsx"
        load_params["empty_as_varchar"] = "true"
        load_params["ignore_errors"] = "true"
    elif file_extension == ".xls":
        return read_from_df(db, file_path, file_name, table_name)
    elif file_extension == ".json":
        load_func = "read_json_auto"
    elif file_extension == ".parquet":
        load_func = "read_parquet"
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    func_args = ", ".join([f"{k}={v}" for k, v in load_params.items()])
    if func_args:
        from_exp = f"FROM {load_func}('{file_path}', {func_args})"
    else:
        from_exp = f"FROM {load_func}('{file_path}')"
    load_sql = f"create table {table_name} as SELECT * {from_exp}"
    try:
        db.sql(load_sql)
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error while reading file with {load_func}: {error_msg}")
        # 如果是类型转换错误或解析错误，直接fallback到pandas
        if "Could not convert" in error_msg or "Failed to parse" in error_msg or "Invalid Input" in error_msg:
            logger.info("检测到类型转换错误，使用pandas读取（支持混合类型数据）")
        return read_from_df(db, file_path, file_name, table_name)


class ExcelReader:
    def __init__(
        self,
        conv_uid: str,
        file_path: str,
        file_name: Optional[str] = None,
        read_type: str = "df",
        database_name: str = ":memory:",
        table_name: str = "data_analysis_table",
        duckdb_extensions_dir: Optional[List[str]] = None,
        force_install: bool = False,
        show_columns: bool = False,
        use_existing_db: bool = False,  # 新增：是否使用已存在的数据库
    ):
        if not file_name:
            file_name = os.path.basename(file_path)
        self.conv_uid = conv_uid
        # connect DuckDB

        db_exists = os.path.exists(database_name) if database_name != ":memory:" else False

        self.db = duckdb.connect(database=database_name, read_only=False)

        self.temp_table_name = "temp_table"
        self.table_name = table_name

        self.excel_file_name = file_name

        if duckdb_extensions_dir:
            self.install_extension(duckdb_extensions_dir, force_install)

        # 如果use_existing_db=True且数据库已存在，则跳过数据导入
        if use_existing_db and db_exists:
            logger.info(f"✅ 使用已存在的数据库: {database_name}，表: {table_name}")
            curr_table = self.table_name
        elif not db_exists:
            curr_table = self.temp_table_name
            if read_type == "df":
                read_from_df(self.db, file_path, file_name, curr_table)
            else:
                read_direct(self.db, file_path, file_name, curr_table)
        else:
            curr_table = self.table_name

        if show_columns:
            # Print table schema
            result = self.db.sql(f"DESCRIBE {curr_table}")
            columns = result.fetchall()
            for column in columns:
                print(column)

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def __del__(self):
        self.close()

    def run(self, sql, table_name: str, df_res: bool = False, transform: bool = True):
        try:
            if f'"{table_name}"' in sql:
                sql = sql.replace(f'"{table_name}"', table_name)
            if transform:
                sql = add_quotes_to_chinese_columns(sql)
            logger.info(f"To be executed SQL: {sql}")
            if df_res:
                return self.db.sql(sql).df()
            return self._run_sql(sql)
        except Exception as e:
            logger.error(f"excel sql run error!, {str(e)}")
            raise ValueError(f"Data Query Exception!\\nSQL[{sql}].\\nError:{str(e)}")

    def _run_sql(self, sql: str):
        results = self.db.sql(sql)
        columns = []
        for desc in results.description:
            columns.append(desc[0])
        return columns, results.fetchall()

    def get_df_by_sql_ex(self, sql: str, table_name: Optional[str] = None):
        table_name = table_name or self.table_name
        return self.run(sql, table_name, df_res=True)

    def get_sample_data(self, table_name: str, sample_size: int = 2):
        """
        获取高质量的采样数据
        
        优化策略：使用LIMIT代替USING SAMPLE，确保：
        1. 数据有效性（不会采到空行或表头）
        2. 数据代表性（按顺序取前N行）
        3. 可重复性（每次结果一致）
        """
        columns, datas = self.run(
            f"SELECT * FROM {table_name} LIMIT {sample_size};",
            table_name=table_name,
            transform=False,
        )
        return columns, datas

    def get_columns(self, table_name: str):
        sql = f"""
        SELECT 
    dc.column_name,
    dc.data_type AS column_type,
    CASE WHEN dc.is_nullable THEN 'YES' ELSE 'NO' END AS "null",
    '' AS key,
    '' AS default,
    '' AS "extra",
    dc.comment
FROM duckdb_columns() dc
WHERE dc.table_name = '{table_name}'
AND dc.schema_name = 'main';
"""
        columns, datas = self.run(sql, table_name, transform=False)
        return columns, datas

    def get_create_table_sql(self, table_name: str) -> str:
        sql = f"""SELECT comment, table_name, database_name FROM duckdb_tables() \
        where table_name = '{table_name}'"""

        columns, datas = self.run(sql, table_name, transform=False)
        table_comment = datas[0][0] if datas and len(datas) > 0 else ""
        cl_columns, cl_datas = self.get_columns(table_name)
        ddl_sql = f"CREATE TABLE {table_name} (\n"
        column_strs = []
        for cl_data in cl_datas:
            column_name = cl_data[0]
            column_type = cl_data[1]
            nullable = cl_data[2]
            column_key = cl_data[3]
            column_default = cl_data[4]
            column_comment = cl_data[6]
            curr_sql = f"    {column_name} {column_type}"
            if column_key and column_key == "PRI":
                curr_sql += " PRIMARY KEY"
            elif nullable and str(nullable).lower() == "no":
                curr_sql += " NOT NULL"
            elif column_default:
                curr_sql += f" DEFAULT {column_default}"
            elif column_comment:
                curr_sql += f" COMMENT '{column_comment}'"
            column_strs.append(curr_sql)
        ddl_sql += ",\n".join(column_strs)
        if table_comment:
            ddl_sql += f"\n) COMMENT '{table_comment}';"
        else:
            ddl_sql += "\n);"

        return ddl_sql

    def get_summary(self, table_name: str) -> str:
        data = self.run(
            f"SUMMARIZE {table_name}", table_name, transform=False, df_res=True
        ).to_json(force_ascii=False)
        return data

    def transform_table(
        self,
        old_table_name: str,
        new_table_name: str,
        transform: TransformedExcelResponse,
    ):
        table_comment = transform.description
        select_sql_list = []
        new_table = new_table_name

        _, cl_datas = self.get_columns(old_table_name)
        old_col_name_to_type = {cl_data[0]: cl_data[1] for cl_data in cl_datas}

        create_columns = []
        for col_transform in transform.columns:
            old_column_name = col_transform["old_column_name"]
            new_column_name = col_transform["new_column_name"]
            new_column_type = old_col_name_to_type[old_column_name]
            old_column_name = f'"{old_column_name}"'  # 使用双引号括起列名
            select_sql_list.append(f"{old_column_name} AS {new_column_name}")
            create_columns.append(f"{new_column_name} {new_column_type}")

        select_sql = ", ".join(select_sql_list)
        create_columns_str = ", ".join(create_columns)
        create_table_str = f"CREATE TABLE {new_table}(\n{create_columns_str}\n);"
        sql = f"""
    {create_table_str}
    INSERT INTO {new_table} SELECT {select_sql}
    from {old_table_name};
    """
        logger.info("Begin to transform table, SQL: \n" + sql)
        self.db.sql(sql)

        # Transform single quotes in table comments, then execute separately
        escaped_table_comment = table_comment.replace("'", "''")
        table_comment_sql = ""
        try:
            table_comment_sql = (
                f"COMMENT ON TABLE {new_table} IS '{escaped_table_comment}';"
            )
            self.db.sql(table_comment_sql)
            logger.info(f"Added comment to table {new_table}")
        except Exception as e:
            logger.warning(
                f"Error while adding table comment: {str(e)}\nSQL: {table_comment_sql}"
            )

        for col_transform in transform.columns:
            column_comment_sql = ""
            new_column_name = ""
            try:
                new_column_name = col_transform["new_column_name"]
                column_description = col_transform["column_description"]
                # In SQL, single quotes within single quotes need to be escaped with
                # two single quotes
                escaped_description = column_description.replace("'", "''")
                column_comment_sql = (
                    f"COMMENT ON COLUMN {new_table}.{new_column_name}"
                    f" IS '{escaped_description}';"
                )
                self.db.sql(column_comment_sql)
                logger.debug(f"Added comment to column {new_table}.{new_column_name}")
            except Exception as e:
                logger.warning(
                    f"Error while adding comment to column {new_column_name}:"
                    f" {str(e)}\nSQL: {column_comment_sql}"
                )

        return new_table

    def install_extension(
        self, duckdb_extensions_dir: Optional[List[str]], force_install: bool = False
    ) -> int:
        if not duckdb_extensions_dir:
            return 0
        cnt = 0
        for extension_dir in duckdb_extensions_dir:
            if not os.path.exists(extension_dir):
                logger.warning(f"Extension directory not exists: {extension_dir}")
                continue
            extension_files = [
                os.path.join(extension_dir, f)
                for f in os.listdir(extension_dir)
                if f.endswith(".duckdb_extension.gz") or f.endswith(".duckdb_extension")
            ]
            _, extensions = self._query_extension()
            installed_extensions = [ext[0] for ext in extensions if ext[1]]
            for extension_file in extension_files:
                try:
                    extension_name = os.path.basename(extension_file).split(".")[0]
                    if not force_install and extension_name in installed_extensions:
                        logger.info(
                            f"Extension {extension_name} has been installed, skip"
                        )
                        continue
                    self.db.install_extension(
                        extension_file, force_install=force_install
                    )
                    self.db.load_extension(extension_name)
                    cnt += 1
                    logger.info(f"Installed extension {extension_name} for DuckDB")
                except Exception as e:
                    logger.warning(
                        f"Error while installing extension {extension_file}: {str(e)}"
                    )
        logger.debug(f"Installed extensions: {cnt}")
        self.list_extensions()
        return cnt

    def list_extensions(self, stdout=False):
        from prettytable import PrettyTable

        table = PrettyTable()
        columns, datas = self._query_extension()
        table.field_names = columns
        for data in datas:
            table.add_row(data)
        show_str = "DuckDB Extensions:\n"
        show_str += table.get_formatted_string()
        if stdout:
            print(show_str)
        else:
            logger.info(show_str)

    def _query_extension(self):
        return self._run_sql(
            "SELECT extension_name, installed, description FROM duckdb_extensions();"
        )
    
    def get_table_size(self, table_name: str) -> dict:
        """
        获取表的行数和列数
        参考 database_model.py 的 get_table_size 方法
        """
        try:
            # 获取列数
            columns_result = self.db.sql(f"DESCRIBE {table_name}").fetchall()
            column_count = len(columns_result)
            
            # 获取行数
            row_count = self.db.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            return {"row_count": row_count, "column_count": column_count}
        except Exception as e:
            logger.error(f"Error getting table size: {e}")
            return {"row_count": 0, "column_count": 0}
    
    def get_unique_values(self, table_name: str, max_unique_values: int = 100) -> dict:
        """
        获取表中列的唯一值（仅非数值列，且唯一值数量<=max_unique_values）
        参考 database_model.py 的 get_unique_values 方法
        这对LLM理解分类数据非常有帮助
        """
        unique_values = {}
        
        try:
            # 获取所有列及其类型
            columns_result = self.db.sql(f"DESCRIBE {table_name}").fetchall()
            
            for column_info in columns_result:
                column_name = column_info[0]
                column_type = column_info[1].upper()
                
                # 只对非数值类型的列获取唯一值
                if not any(numeric_type in column_type for numeric_type in ['INT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'BIGINT']):
                    try:
                        # 获取唯一值数量
                        count_query = f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"'
                        unique_count = self.db.sql(count_query).fetchone()[0]
                        
                        # 只有唯一值数量不超过阈值时才获取
                        if unique_count <= max_unique_values and unique_count > 0:
                            # 获取所有唯一值
                            values_query = f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL ORDER BY "{column_name}" LIMIT {max_unique_values}'
                            result = self.db.sql(values_query).fetchall()
                            unique_values[column_name] = [row[0] for row in result]
                    except Exception as e:
                        logger.warning(f"Error getting unique values for column {column_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in get_unique_values: {e}")
        
        return unique_values
    
    def get_table_info_description(self, table_name: str) -> str:
        """
        生成表的详细描述信息，包括表大小、列信息和唯一值
        参考 database_model.py 的 describe_table_info 方法
        """
        try:
            # 获取表大小
            table_size = self.get_table_size(table_name)
            
            # 获取列信息
            columns_result = self.db.sql(f"DESCRIBE {table_name}").fetchall()
            
            # 获取唯一值
            unique_values = self.get_unique_values(table_name)
            
            # 构建描述
            description = f"表名: {table_name}\n"
            description += f"数据规模: {table_size['row_count']} 行 × {table_size['column_count']} 列\n\n"
            description += "列信息:\n"
            
            for column_info in columns_result:
                column_name = column_info[0]
                column_type = column_info[1]
                is_nullable = "YES" if column_info[2] == "YES" else "NO"
                
                column_desc = f"  • {column_name} ({column_type})"
                if is_nullable == "NO":
                    column_desc += " [NOT NULL]"
                description += column_desc + "\n"
            
            # 添加唯一值信息
            if unique_values:
                description += "\n分类列的可选值（对数据分析很重要）:\n"
                for column, values in unique_values.items():
                    # 限制显示的值数量
                    if len(values) <= 20:
                        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in values])
                    else:
                        preview_values = values[:20]
                        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in preview_values])
                        values_str += f" ... (共 {len(values)} 个值)"
                    
                    description += f"  • {column}: {values_str}\n"
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            return f"Error: {str(e)}"
