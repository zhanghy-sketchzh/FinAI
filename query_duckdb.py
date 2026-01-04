#!/usr/bin/env python3
"""
DuckDB查询工具
用法: python query_duckdb.py <数据库路径> <SQL查询>
示例: python query_duckdb.py packages/pilot/meta_data/excel_dbs/excel_d4cf6997.duckdb "SELECT * FROM \"奖金数据模版4\" LIMIT 5"
"""

import sys
import os
import json
import duckdb
from pathlib import Path


def query_duckdb(db_path: str, sql: str, output_format: str = "table"):
    """
    执行DuckDB查询
    
    Args:
        db_path: DuckDB数据库文件路径
        sql: SQL查询语句
        output_format: 输出格式，可选: "table", "json", "csv"
    
    Returns:
        查询结果
    """
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    # 连接数据库（只读模式）
    conn = duckdb.connect(database=db_path, read_only=True)
    
    try:
        # 执行查询
        result = conn.execute(sql)
        
        # 获取列名
        columns = [desc[0] for desc in result.description]
        
        # 获取所有行
        rows = result.fetchall()
        
        # 根据输出格式返回结果
        if output_format == "json":
            # 返回JSON格式
            data = [dict(zip(columns, row)) for row in rows]
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif output_format == "csv":
            # 返回CSV格式
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)
            return output.getvalue()
        else:
            # 默认返回表格格式
            return format_table(columns, rows)
    
    except Exception as e:
        raise Exception(f"SQL执行失败: {str(e)}")
    
    finally:
        conn.close()


def format_table(columns, rows):
    """格式化输出为表格"""
    if not rows:
        return "查询结果为空"
    
    # 计算每列的最大宽度
    col_widths = {}
    for col in columns:
        col_widths[col] = max(len(str(col)), max(len(str(row[columns.index(col)])) for row in rows))
    
    # 构建表格
    lines = []
    
    # 表头
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))
    
    # 数据行
    for row in rows:
        line = " | ".join(str(row[columns.index(col)]).ljust(col_widths[col]) for col in columns)
        lines.append(line)
    
    return "\n".join(lines)


def list_tables(db_path: str):
    """列出数据库中的所有表"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = duckdb.connect(database=db_path, read_only=True)
    
    try:
        result = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        
        tables = [row[0] for row in result]
        return tables
    finally:
        conn.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n用法:")
        print("  python query_duckdb.py <数据库路径> <SQL查询> [输出格式]")
        print("\n示例:")
        print('  python query_duckdb.py packages/pilot/meta_data/excel_dbs/excel_d4cf6997.duckdb "SELECT * FROM \\"奖金数据模版4\\" LIMIT 5"')
        print('  python query_duckdb.py packages/pilot/meta_data/excel_dbs/excel_d4cf6997.duckdb "SELECT * FROM \\"奖金数据模版4\\" LIMIT 5" json')
        print("\n列出所有表:")
        print("  python query_duckdb.py <数据库路径> --list-tables")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # 如果是列出表
    if sys.argv[2] == "--list-tables":
        try:
            tables = list_tables(db_path)
            print(f"\n数据库 {db_path} 中的表:")
            for table in tables:
                print(f"  - {table}")
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    sql = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else "table"
    
    try:
        result = query_duckdb(db_path, sql, output_format)
        print(result)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

