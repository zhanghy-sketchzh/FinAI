#!/usr/bin/env python3
"""
测试 Excel 信息获取功能

功能：
1. 上传 Excel 文件
2. 获取 Excel 基本信息（前10行、行列数、数据描述、推荐问题）
"""
import requests
import json
import sys
import os
from pathlib import Path

# API 配置
API_BASE = "http://localhost:5670"
API_KEY = ""  # 如果需要的话

def upload_excel(excel_file_path: str, conv_uid: str = "test_conv_001"):
    """
    上传 Excel 文件
    
    Args:
        excel_file_path: Excel 文件路径
        conv_uid: 会话ID
    
    Returns:
        上传结果
    """
    url = f"{API_BASE}/api/v1/resource/file/upload"
    
    # 准备文件
    files = {
        'doc_files': open(excel_file_path, 'rb')
    }
    
    # 准备参数
    params = {
        'chat_mode': 'chat_excel',
        'conv_uid': conv_uid,
        'sys_code': 'test'
    }
    
    headers = {}
    if API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'
    
    print(f"📤 正在上传 Excel 文件: {excel_file_path}")
    print(f"   会话ID: {conv_uid}")
    
    try:
        response = requests.post(url, files=files, params=params, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success'):
            data = result.get('data', {})
            print(f"\n✅ 上传成功！")
            print(f"   文件名: {data.get('file_name')}")
            print(f"   数据库: {data.get('db_name')}")
            print(f"   表名: {data.get('table_name')}")
            print(f"   数据规模: {data.get('row_count')} 行 × {data.get('column_count')} 列")
            print(f"   注册状态: {data.get('register_status')}")
            
            # 显示推荐问题
            suggested_questions = data.get('suggested_questions', [])
            if suggested_questions:
                print(f"\n💡 推荐问题 ({len(suggested_questions)} 个):")
                for i, question in enumerate(suggested_questions, 1):
                    print(f"   {i}. {question}")
            
            return data
        else:
            print(f"❌ 上传失败: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        files['doc_files'].close()


def get_excel_info(conv_uid: str):
    """
    获取 Excel 基本信息
    
    Args:
        conv_uid: 会话ID
    
    Returns:
        Excel 信息
    """
    url = f"{API_BASE}/api/v1/resource/excel/info"
    
    params = {
        'conv_uid': conv_uid
    }
    
    headers = {}
    if API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'
    
    print(f"\n📊 正在获取 Excel 信息...")
    print(f"   会话ID: {conv_uid}")
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success'):
            data = result.get('data', {})
            print(f"\n✅ 获取成功！")
            print(f"\n=== Excel 基本信息 ===")
            print(f"文件名: {data.get('file_name')}")
            print(f"表名: {data.get('table_name')}")
            print(f"数据规模: {data.get('row_count')} 行 × {data.get('column_count')} 列")
            print(f"上传时间: {data.get('gmt_created')}")
            
            # 显示前10行数据
            top_10_rows = data.get('top_10_rows', [])
            if top_10_rows:
                print(f"\n=== 前 {len(top_10_rows)} 行数据 ===")
                for i, row in enumerate(top_10_rows[:3], 1):  # 只显示前3行
                    print(f"第 {i} 行: {row}")
                if len(top_10_rows) > 3:
                    print(f"... (共 {len(top_10_rows)} 行)")
            
            # 显示数据描述
            data_description = data.get('data_description')
            if data_description:
                print(f"\n=== 数据描述 ===")
                # 只显示前500字符
                desc_preview = data_description[:500]
                print(desc_preview)
                if len(data_description) > 500:
                    print(f"... (共 {len(data_description)} 字符)")
            
            # 显示推荐问题
            suggested_questions = data.get('suggested_questions', [])
            if suggested_questions:
                print(f"\n=== 推荐问题 ({len(suggested_questions)} 个) ===")
                for i, question in enumerate(suggested_questions, 1):
                    print(f"{i}. {question}")
            
            return data
        else:
            print(f"❌ 获取失败: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 Excel 信息获取功能")
    parser.add_argument("--excel", type=str, help="Excel 文件路径")
    parser.add_argument("--conv-uid", type=str, default="test_conv_001", help="会话ID")
    parser.add_argument("--api-base", type=str, default="http://localhost:5670", help="API 基础URL")
    parser.add_argument("--api-key", type=str, default="", help="API Key")
    parser.add_argument("--only-get", action="store_true", help="只获取信息，不上传")
    
    args = parser.parse_args()
    
    global API_BASE, API_KEY
    API_BASE = args.api_base
    API_KEY = args.api_key
    
    print("=" * 80)
    print("🧪 Excel 信息获取功能测试")
    print("=" * 80)
    
    if not args.only_get:
        # 上传 Excel 文件
        if not args.excel:
            print("❌ 请指定 Excel 文件路径 (--excel)")
            sys.exit(1)
        
        if not os.path.exists(args.excel):
            print(f"❌ 文件不存在: {args.excel}")
            sys.exit(1)
        
        upload_result = upload_excel(args.excel, args.conv_uid)
        
        if not upload_result:
            print("\n❌ 上传失败，测试终止")
            sys.exit(1)
    
    # 获取 Excel 信息
    excel_info = get_excel_info(args.conv_uid)
    
    if not excel_info:
        print("\n❌ 获取信息失败，测试终止")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

