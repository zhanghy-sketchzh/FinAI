#!/usr/bin/env python3
"""
简单的 DuckDB SQL 执行脚本
直接在代码中修改数据库路径和SQL语句
"""
import duckdb
from pathlib import Path

# ========== 配置区域 - 在这里修改 ==========

# 数据库文件路径
DB_PATH = "/home/adminstrator/work/FinAI/packages/pilot/meta_data/excel_dbs/excel_7a8cb57b.duckdb"

# 表名
TABLE_NAME = "奖金数据模版_final"

# 要执行的SQL (支持多条,用列表)
SQL_QUERIES = [
    # 示例1: 查看前5行数据
    f'''
    WITH
  department_summary AS (
    SELECT
      "基本信息(BasicInfo)-部门" AS "部门",
      COUNT("基本信息(BasicInfo)-员工ID") AS "员工人数",
      ROUND(
        SUM("本次合计(TotalofThisYear)-合计金额(CNY)") / COUNT("基本信息(BasicInfo)-员工ID"),
        2
      ) AS "2025年人均奖金",
      ROUND(
        SUM("上年度参考(ReferenceInfo)-2024合计金额(CNY)") / COUNT("基本信息(BasicInfo)-员工ID"),
        2
      ) AS "2024年人均奖金"
    FROM
      "奖金数据模版_final"
    WHERE
      "本次合计(TotalofThisYear)-合计金额(CNY)" IS NOT NULL
      AND "上年度参考(ReferenceInfo)-2024合计金额(CNY)" IS NOT NULL
    GROUP BY
      "基本信息(BasicInfo)-部门"
  ),
  yoy_calculation AS (
    SELECT
      "部门",
      "员工人数",
      "2025年人均奖金",
      "2024年人均奖金",
      ROUND(
        (("2025年人均奖金" - "2024年人均奖金") / "2024年人均奖金") * 100,
        2
      ) AS "YOY增长率(%)",
      ROUND(("2025年人均奖金" - "2024年人均奖金"), 2) AS "变化金额"
    FROM
      department_summary
  )
SELECT
  "部门",
  "员工人数",
  "2025年人均奖金",
  "2024年人均奖金",
  "YOY增长率(%)",
  "变化金额"
FROM
  yoy_calculation
ORDER BY
  "部门";
    ''',
    
]

# ========== 执行区域 - 一般不需要修改 ==========

def main():
    # 检查数据库文件
    if not Path(DB_PATH).exists():
        print(f"❌ 错误: 数据库文件不存在: {DB_PATH}")
        return
    
    print(f"📦 连接数据库: {DB_PATH}")
    print(f"📋 表名: {TABLE_NAME}")
    print("=" * 80)
    
    try:
        # 连接数据库
        conn = duckdb.connect(DB_PATH)
        
        # 执行所有SQL
        for i, sql in enumerate(SQL_QUERIES, 1):
            sql = sql.strip()
            if not sql or sql.startswith('#'):
                continue
                
            print(f"\n【查询 {i}】")
            print("-" * 80)
            print(sql)
            print("-" * 80)
            
            try:
                # 执行SQL
                result = conn.execute(sql)
                df = result.df()
                
                print(f"\n✅ 返回 {len(df)} 行数据:\n")
                print(df.to_string())
                print("\n" + "=" * 80)
                
            except Exception as e:
                print(f"\n❌ SQL执行失败:")
                print(f"错误: {e}")
                print("=" * 80)
        
        # 关闭连接
        conn.close()
        print("\n✅ 完成!")
        
    except Exception as e:
        print(f"\n❌ 连接数据库失败: {e}")


if __name__ == "__main__":
    main()
