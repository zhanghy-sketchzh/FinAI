#!/usr/bin/env python3
"""
通过 API 删除 ad_data 数据库连接配置的脚本

使用方法:
    python scripts/delete_ad_data_connection_api.py
    
注意: 需要先启动 DB-GPT 服务
"""

import requests
import sys
import os

# 默认 API 地址
DEFAULT_API_BASE = "http://localhost:5670"
DEFAULT_API_KEY = None  # 如果设置了 api_keys，需要提供


def delete_ad_data_connection(api_base: str = DEFAULT_API_BASE, api_key: str = None):
    """通过 API 删除 ad_data 数据库连接配置"""
    
    # 首先获取数据源列表，找到 ad_data 的 ID
    list_url = f"{api_base}/api/v2/serve/datasources"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        print(f"正在获取数据源列表...")
        response = requests.get(list_url, headers=headers)
        response.raise_for_status()
        datasources = response.json().get("data", [])
        
        # 查找 ad_data
        ad_data_source = None
        for ds in datasources:
            if ds.get("db_name") == "ad_data":
                ad_data_source = ds
                break
        
        if not ad_data_source:
            print("未找到 ad_data 数据库连接配置")
            return False
        
        datasource_id = ad_data_source.get("id")
        print(f"找到 ad_data 数据库连接配置 (ID: {datasource_id})")
        print(f"  - 数据库类型: {ad_data_source.get('db_type')}")
        print(f"  - 数据库名称: {ad_data_source.get('db_name')}")
        
        # 确认删除
        confirm = input("\n确认要删除 ad_data 数据库连接配置吗? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("取消删除操作")
            return False
        
        # 删除数据源
        delete_url = f"{api_base}/api/v2/serve/datasources/{datasource_id}"
        print(f"\n正在删除 ad_data 数据库连接配置...")
        response = requests.delete(delete_url, headers=headers)
        response.raise_for_status()
        
        print("✓ 成功删除 ad_data 数据库连接配置（包括 db_profile）")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API 请求失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  响应内容: {e.response.text}")
        return False
    except Exception as e:
        print(f"✗ 删除失败: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="删除 ad_data 数据库连接配置")
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"API 基础地址 (默认: {DEFAULT_API_BASE})"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API 密钥（如果配置了 api_keys）"
    )
    
    args = parser.parse_args()
    delete_ad_data_connection(args.api_base, args.api_key)

