#!/usr/bin/env python3
"""
删除 ad_data 数据库连接配置的脚本

使用方法:
    python scripts/delete_ad_data_connection.py [--config config_file.toml]
"""

import sys
import os
import argparse

# 添加项目路径到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "packages", "dbgpt-app", "src"))
sys.path.insert(0, os.path.join(project_root, "packages", "dbgpt-serve", "src"))
sys.path.insert(0, os.path.join(project_root, "packages", "dbgpt-core", "src"))
sys.path.insert(0, os.path.join(project_root, "packages", "dbgpt-ext", "src"))


def delete_ad_data_connection(config_file: str = None):
    """删除 ad_data 数据库连接配置"""
    from dbgpt import SystemApp
    from dbgpt_app.dbgpt_server import load_config
    from dbgpt_app.base import _initialize_db_storage
    from dbgpt_serve.datasource.manages.connect_config_db import ConnectConfigDao
    
    # 加载配置
    if config_file is None:
        # 尝试使用默认配置文件
        default_configs = [
            "configs/dbgpt-app-config.example.toml",
            "configs/dbgpt-proxy-siliconflow.toml",
        ]
        config_file = None
        for cfg in default_configs:
            cfg_path = os.path.join(project_root, cfg)
            if os.path.exists(cfg_path):
                config_file = cfg_path
                break
        
        if config_file is None:
            print("错误: 未找到配置文件，请使用 --config 参数指定配置文件")
            return False
    
    if not os.path.isabs(config_file):
        config_file = os.path.join(project_root, config_file)
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        return False
    
    print(f"正在加载配置文件: {config_file}")
    
    try:
        # 加载配置并初始化 SystemApp
        app_config = load_config(config_file)
        system_app = SystemApp()
        
        # 初始化数据库存储
        _initialize_db_storage(app_config.service, system_app)
        
        # 创建 DAO 实例
        dao = ConnectConfigDao()
        
        # 检查 ad_data 是否存在
        db_config = dao.get_by_names("ad_data")
        if db_config:
            print(f"\n找到 ad_data 数据库连接配置:")
            print(f"  - ID: {db_config.id}")
            print(f"  - 数据库类型: {db_config.db_type}")
            print(f"  - 数据库名称: {db_config.db_name}")
            if db_config.db_host:
                print(f"  - 主机: {db_config.db_host}:{db_config.db_port}")
            
            # 确认删除
            confirm = input("\n确认要删除 ad_data 数据库连接配置吗? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                try:
                    # 注意：这里只删除数据库连接配置，不删除 db_profile
                    # 如果需要删除 db_profile，需要使用 Service.delete_by_db_name 方法
                    dao.delete_db("ad_data")
                    print("✓ 成功删除 ad_data 数据库连接配置")
                    print("⚠️  注意: 如果存在 db_profile，需要手动删除或使用 API 删除")
                    return True
                except Exception as e:
                    print(f"✗ 删除失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print("取消删除操作")
                return False
        else:
            print("未找到 ad_data 数据库连接配置")
            return False
    except Exception as e:
        print(f"✗ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除 ad_data 数据库连接配置")
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径 (默认: 自动查找)"
    )
    args = parser.parse_args()
    delete_ad_data_connection(args.config)

