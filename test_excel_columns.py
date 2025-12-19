#!/usr/bin/env python3
"""测试Excel列名处理"""
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "dbgpt-serve" / "src"))
sys.path.insert(0, str(project_root / "packages" / "dbgpt-core" / "src"))

from dbgpt_serve.datasource.service.excel_auto_register import ExcelAutoRegisterService

def test_excel_columns():
    """测试Excel列名处理"""
    excel_path = "/Users/luchun/Desktop/work/Finai/奖金数据模版3.xlsx"
    
    # 初始化服务（不需要LLM）
    service = ExcelAutoRegisterService(llm_client=None, model_name=None)
    
    # 处理Excel
    print("=" * 80)
    print(f"正在处理Excel文件: {excel_path}")
    print("=" * 80)
    
    try:
        result = service.process_excel(
            excel_file_path=excel_path,
            force_reimport=True
        )
        
        print(f"\n处理结果:")
        print(f"  状态: {result['status']}")
        print(f"  表名: {result['table_name']}")
        print(f"  行数: {result['row_count']}")
        print(f"  列数: {result['column_count']}")
        
        print(f"\n最终列名列表 ({result['column_count']}列):")
        print("=" * 80)
        for i, col_info in enumerate(result['columns_info'], 1):
            print(f"{i:3d}. {col_info['name']}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_excel_columns()
