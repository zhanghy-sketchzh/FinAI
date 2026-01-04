#!/usr/bin/env python3
"""
简单的 DuckDB SQL 执行脚本
直接在代码中修改数据库路径和SQL语句
"""
import duckdb
from pathlib import Path

# ========== 配置区域 - 在这里修改 ==========

# 数据库文件路径
DB_PATH = "packages/pilot/meta_data/excel_dbs/excel_4368df0b.db"

# 表名
TABLE_NAME = "奖金数据模版4"

# 要执行的SQL (支持多条,用列表)
SQL_QUERIES = [
    # 示例1: 查看前5行数据
    f'''
    WITH base_2023 AS ( SELECT "上年度参考信息(Reference Info of Last Year)-2023人才梯队" AS "2023人才梯队", COUNT(DISTINCT "基本信息(BasicInfo)-员工ID") AS "2023人数", AVG( CAST("上年度参考信息(Reference Info of Last Year)-2023合计金额(CNY)" AS DOUBLE) ) AS "2023平均奖金(CNY)", SUM( CAST("上年度参考信息(Reference Info of Last Year)-2023合计金额(CNY)" AS DOUBLE) ) AS "2023奖金总额(CNY)" FROM "奖金数据模版3" WHERE "上年度参考信息(Reference Info of Last Year)-2023人才梯队" IS NOT NULL GROUP BY "上年度参考信息(Reference Info of Last Year)-2023人才梯队" ), base_2023_with_ratio AS ( SELECT *, ROUND( "2023人数" * 100.0 / SUM("2023人数") OVER (), 2 ) AS "2023人数占比(%)" FROM base_2023 ), base_2024 AS ( SELECT "基本信息(BasicInfo)-人才梯队" AS "人才梯队", COUNT(DISTINCT "基本信息(BasicInfo)-员工ID") AS "2024人数" FROM "奖金数据模版3" WHERE "基本信息(BasicInfo)-人才梯队" IS NOT NULL GROUP BY "基本信息(BasicInfo)-人才梯队" ) SELECT b23."2023人才梯队", b23."2023人数", b23."2023人数占比(%)", ROUND(b23."2023平均奖金(CNY)", 2) AS "2023平均奖金(CNY)", b23."2023奖金总额(CNY)", COALESCE(b24."2024人数", 0) AS "2024人数", COALESCE(b24."2024人数", 0) - b23."2023人数" AS "人数变化", ROUND( (COALESCE(b24."2024人数", 0) - b23."2023人数") * 100.0 / NULLIF(b23."2023人数", 0), 2 ) AS "人数变化率(%)" FROM base_2023_with_ratio b23 LEFT JOIN base_2024 b24 ON b23."2023人才梯队" = b24."人才梯队" ORDER BY b23."2023人数" DESC;
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
