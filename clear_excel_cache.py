#!/usr/bin/env python3
"""
清除Excel缓存和会话聊天缓存工具
用于重新生成Schema理解和清除会话数据
"""
import sqlite3
import sys
import shutil
from pathlib import Path
from typing import List


def _get_path_candidates(relative_path: str) -> List[Path]:
    """
    获取可能的路径候选列表
    
    Args:
        relative_path: 相对路径，如 "pilot/data/excel_cache/excel_metadata.db"
    
    Returns:
        路径候选列表
    """
    base_dir = Path(__file__).parent
    candidates = [
        base_dir / "packages" / relative_path,  # packages/pilot/...
        base_dir / relative_path,  # pilot/...
    ]
    return [p for p in candidates if p.exists()]


def _find_first_path(relative_path: str) -> Path:
    """
    查找第一个存在的路径
    
    Args:
        relative_path: 相对路径
    
    Returns:
        存在的路径，如果都不存在则返回第一个候选路径
    """
    candidates = _get_path_candidates(relative_path)
    if candidates:
        return candidates[0]
    # 如果都不存在，返回默认的第一个候选路径
    base_dir = Path(__file__).parent
    return base_dir / "packages" / relative_path


def clear_cache_by_filename(filename: str = None):
    """
    清除指定文件的缓存
    
    Args:
        filename: Excel文件名，如果为None则清除所有缓存
    """
    # 缓存数据库路径
    cache_db = _find_first_path("pilot/data/excel_cache/excel_metadata.db")
    
    if not cache_db.exists():
        print(f"❌ 缓存数据库不存在: {cache_db}")
        return
    
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    
    # 查看当前缓存
    cursor.execute("SELECT id, original_filename, table_name, db_name, access_count FROM excel_metadata")
    records = cursor.fetchall()
    
    if not records:
        print("📭 当前没有缓存记录")
        conn.close()
        return
    
    print(f"\n📊 当前缓存记录 ({len(records)}条):")
    print("-" * 80)
    for record in records:
        print(f"ID: {record[0]}, 文件: {record[1]}, 表名: {record[2]}, 数据库: {record[3]}, 访问次数: {record[4]}")
    print("-" * 80)
    
    if filename:
        # 删除指定文件的缓存
        cursor.execute("DELETE FROM excel_metadata WHERE original_filename = ?", (filename,))
        deleted = cursor.rowcount
        conn.commit()
        
        if deleted > 0:
            print(f"\n✅ 已删除 '{filename}' 的缓存记录 ({deleted}条)")
            print("💡 下次上传相同文件时，将重新生成Schema理解")
        else:
            print(f"\n⚠️ 未找到文件 '{filename}' 的缓存记录")
    else:
        # 清除所有缓存
        choice = input("\n⚠️  确认要清除所有缓存吗？(yes/no): ")
        if choice.lower() == 'yes':
            cursor.execute("DELETE FROM excel_metadata")
            deleted = cursor.rowcount
            conn.commit()
            print(f"\n✅ 已清除所有缓存记录 ({deleted}条)")
        else:
            print("\n❌ 取消操作")
    
    conn.close()


def list_cache():
    """列出所有缓存记录"""
    cache_db = _find_first_path("pilot/data/excel_cache/excel_metadata.db")
    
    if not cache_db.exists():
        print(f"❌ 缓存数据库不存在: {cache_db}")
        return
    
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, 
            original_filename, 
            table_name, 
            db_name, 
            row_count, 
            column_count,
            access_count,
            created_at,
            last_accessed
        FROM excel_metadata
        ORDER BY last_accessed DESC
    """)
    
    records = cursor.fetchall()
    
    if not records:
        print("📭 当前没有缓存记录")
    else:
        print(f"\n📊 缓存记录详情 (共{len(records)}条):\n")
        for record in records:
            print(f"{'='*80}")
            print(f"ID: {record[0]}")
            print(f"文件名: {record[1]}")
            print(f"表名: {record[2]}")
            print(f"数据库: {record[3]}")
            print(f"数据规模: {record[4]}行 × {record[5]}列")
            print(f"访问次数: {record[6]}")
            print(f"创建时间: {record[7]}")
            print(f"最后访问: {record[8]}")
    
    conn.close()


def clear_cache_by_id(cache_id: int):
    """根据ID删除缓存"""
    cache_db = _find_first_path("pilot/data/excel_cache/excel_metadata.db")
    
    if not cache_db.exists():
        print(f"❌ 缓存数据库不存在: {cache_db}")
        return
    
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM excel_metadata WHERE id = ?", (cache_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    if deleted > 0:
        print(f"✅ 已删除ID={cache_id}的缓存记录")
    else:
        print(f"⚠️ 未找到ID={cache_id}的缓存记录")


def clear_chat_excel_tmp(auto_confirm: bool = False):
    """清除Excel聊天临时数据库目录
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    tmp_dirs = _get_path_candidates("pilot/data/_chat_excel_tmp")
    
    if not tmp_dirs:
        print(f"📭 Excel聊天临时目录不存在")
        return
    
    total_files = 0
    for tmp_dir in tmp_dirs:
        if not tmp_dir.exists():
            continue
            
        files = list(tmp_dir.glob("*"))
        file_count = len(files)
        total_files += file_count
    
    if total_files == 0:
        print(f"📭 Excel聊天临时目录为空")
        return
    
    print(f"\n📊 发现 {total_files} 个临时文件/目录")
    if not auto_confirm:
        choice = input("⚠️  确认要清除Excel聊天临时数据库吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_total = 0
    for tmp_dir in tmp_dirs:
        if not tmp_dir.exists():
            continue
        
        files = list(tmp_dir.glob("*"))
        file_count = len(files)
        
        try:
            shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            deleted_total += file_count
            print(f"✅ 已清除 {tmp_dir} ({file_count}个文件/目录)")
        except Exception as e:
            print(f"❌ 清除 {tmp_dir} 失败: {e}")
    
    if deleted_total > 0:
        print(f"✅ 总计清除 {deleted_total} 个临时文件/目录")


def clear_chat_history(conv_uid: str = None, auto_confirm: bool = False):
    """清除会话历史记录
    
    Args:
        conv_uid: 会话ID，如果为None则清除所有会话
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    db_paths = _get_path_candidates("pilot/meta_data/dbgpt.db")
    
    if not db_paths:
        print(f"❌ 数据库不存在")
        return
    
    total_conv_deleted = 0
    total_msg_deleted = 0
    
    for db_path in db_paths:
        print(f"\n处理数据库: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        try:
            if conv_uid:
                # 删除指定会话的消息
                cursor.execute("DELETE FROM chat_history_message WHERE conv_uid = ?", (conv_uid,))
                msg_deleted = cursor.rowcount
                
                # 删除指定会话的历史记录
                cursor.execute("DELETE FROM chat_history WHERE conv_uid = ?", (conv_uid,))
                conv_deleted = cursor.rowcount
                
                conn.commit()
                
                total_conv_deleted += conv_deleted
                total_msg_deleted += msg_deleted
                
                if conv_deleted > 0 or msg_deleted > 0:
                    print(f"  ✅ 已删除会话 '{conv_uid}' 的记录 (历史: {conv_deleted}条, 消息: {msg_deleted}条)")
            else:
                # 统计记录数
                cursor.execute("SELECT COUNT(*) FROM chat_history")
                conv_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM chat_history_message")
                msg_count = cursor.fetchone()[0]
                
                if conv_count == 0 and msg_count == 0:
                    print(f"  📭 当前没有会话记录")
                    continue
                
                print(f"  📊 当前会话记录: 历史 {conv_count}条, 消息 {msg_count}条")
                
                if not auto_confirm and total_conv_deleted == 0:  # 只在第一次询问
                    choice = input("\n⚠️  确认要清除所有会话记录吗？(yes/no): ")
                    if choice.lower() != 'yes':
                        print("❌ 取消操作")
                        return
                
                cursor.execute("DELETE FROM chat_history_message")
                msg_deleted = cursor.rowcount
                cursor.execute("DELETE FROM chat_history")
                conv_deleted = cursor.rowcount
                conn.commit()
                
                total_conv_deleted += conv_deleted
                total_msg_deleted += msg_deleted
                
                print(f"  ✅ 已清除会话记录 (历史: {conv_deleted}条, 消息: {msg_deleted}条)")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                print(f"  ⚠️ 数据库表不存在: {e}")
            else:
                raise
        finally:
            conn.close()
    
    if total_conv_deleted > 0 or total_msg_deleted > 0:
        print(f"\n✅ 总计: 历史 {total_conv_deleted}条, 消息 {total_msg_deleted}条")


def clear_excel_dbs(auto_confirm: bool = False):
    """清除Excel数据库文件目录（DuckDB格式）
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    excel_dbs_dirs = _get_path_candidates("pilot/meta_data/excel_dbs")
    
    if not excel_dbs_dirs:
        print(f"📭 Excel数据库目录不存在")
        return
    
    total_files = 0
    all_db_files = []
    
    for excel_dbs_dir in excel_dbs_dirs:
        if not excel_dbs_dir.exists():
            continue
        # 同时支持 .duckdb（新格式）和 .db（旧格式，兼容性）
        duckdb_files = list(excel_dbs_dir.glob("*.duckdb"))
        db_files = list(excel_dbs_dir.glob("*.db"))
        all_db_files.extend(duckdb_files)
        all_db_files.extend(db_files)
        total_files += len(duckdb_files) + len(db_files)
    
    if total_files == 0:
        print(f"📭 Excel数据库目录为空")
        return
    
    print(f"\n📊 发现 {total_files} 个Excel数据库文件（.duckdb 和 .db）")
    if not auto_confirm:
        choice = input("⚠️  确认要清除所有Excel数据库文件吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_count = 0
    for db_file in all_db_files:
        try:
            db_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ 删除文件失败 {db_file.name}: {e}")
    print(f"✅ 已清除 {deleted_count}/{total_files} 个Excel数据库文件")


def clear_uploaded_excel_files(auto_confirm: bool = False):
    """清除上传的Excel文件（在pilot/data/目录下的.xlsx文件）
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    data_dirs = _get_path_candidates("pilot/data")
    
    if not data_dirs:
        print(f"📭 数据目录不存在")
        return
    
    total_files = 0
    all_excel_files = []
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        # 只查找目录下的.xlsx文件，不包括子目录
        excel_files = [f for f in data_dir.glob("*.xlsx") if f.is_file()]
        all_excel_files.extend(excel_files)
        total_files += len(excel_files)
    
    if total_files == 0:
        print(f"📭 没有找到上传的Excel文件")
        return
    
    print(f"\n📊 发现 {total_files} 个上传的Excel文件")
    if not auto_confirm:
        choice = input("⚠️  确认要清除所有上传的Excel文件吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_count = 0
    for excel_file in all_excel_files:
        try:
            excel_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ 删除文件失败 {excel_file.name}: {e}")
    print(f"✅ 已清除 {deleted_count}/{total_files} 个上传的Excel文件")


def clear_file_server_storage(auto_confirm: bool = False):
    """清除文件服务器存储目录
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    file_server_dirs = _get_path_candidates("pilot/data/file_server")
    
    if not file_server_dirs:
        print(f"📭 文件服务器存储目录不存在")
        return
    
    total_files = 0
    for file_server_dir in file_server_dirs:
        if not file_server_dir.exists():
            continue
        files = list(file_server_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        total_files += len(files)
    
    if total_files == 0:
        print(f"📭 文件服务器存储目录为空")
        return
    
    print(f"\n📊 发现 {total_files} 个文件")
    if not auto_confirm:
        choice = input("⚠️  确认要清除文件服务器存储吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_total = 0
    for file_server_dir in file_server_dirs:
        if not file_server_dir.exists():
            continue
        
        files = list(file_server_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        file_count = len(files)
        
        try:
            # 删除目录下所有内容但保留目录
            for item in file_server_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            deleted_total += file_count
            print(f"✅ 已清除 {file_server_dir} ({file_count}个文件)")
        except Exception as e:
            print(f"❌ 清除 {file_server_dir} 失败: {e}")
    
    if deleted_total > 0:
        print(f"✅ 总计清除 {deleted_total} 个文件")


def clear_model_cache(auto_confirm: bool = False):
    """清除模型缓存目录
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    # 支持多个路径
    model_cache_dirs = _get_path_candidates("pilot/data/model_cache")
    
    if not model_cache_dirs:
        print(f"📭 模型缓存目录不存在")
        return
    
    total_files = 0
    for model_cache_dir in model_cache_dirs:
        if not model_cache_dir.exists():
            continue
        files = list(model_cache_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        total_files += len(files)
    
    if total_files == 0:
        print(f"📭 模型缓存目录为空")
        return
    
    print(f"\n📊 发现 {total_files} 个缓存文件")
    if not auto_confirm:
        choice = input("⚠️  确认要清除模型缓存吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_total = 0
    for model_cache_dir in model_cache_dirs:
        if not model_cache_dir.exists():
            continue
        
        files = list(model_cache_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        file_count = len(files)
        
        try:
            # 删除目录下所有内容但保留目录
            for item in model_cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            deleted_total += file_count
            print(f"✅ 已清除 {model_cache_dir} ({file_count}个文件)")
        except Exception as e:
            print(f"❌ 清除 {model_cache_dir} 失败: {e}")
    
    if deleted_total > 0:
        print(f"✅ 总计清除 {deleted_total} 个缓存文件")


def list_chat_history():
    """列出会话历史记录"""
    # 支持多个路径
    db_paths = _get_path_candidates("pilot/meta_data/dbgpt.db")
    
    if not db_paths:
        print(f"❌ 数据库不存在")
        return
    
    for db_path in db_paths:
        print(f"\n数据库: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    conv_uid,
                    chat_mode,
                    user_name,
                    summary,
                    gmt_created,
                    gmt_modified
                FROM chat_history
                ORDER BY gmt_modified DESC
                LIMIT 50
            """)
            
            records = cursor.fetchall()
            
            if not records:
                print("  📭 当前没有会话记录")
            else:
                print(f"\n  📊 会话历史记录 (最近50条):\n")
                for record in records:
                    print(f"  {'='*78}")
                    print(f"  会话ID: {record[0]}")
                    print(f"  聊天模式: {record[1]}")
                    print(f"  用户: {record[2]}")
                    print(f"  摘要: {record[3][:100] if record[3] else '无'}...")
                    print(f"  创建时间: {record[4]}")
                    print(f"  修改时间: {record[5]}")
                
                # 统计总数
                cursor.execute("SELECT COUNT(*) FROM chat_history")
                total = cursor.fetchone()[0]
                print(f"\n  📊 总计: {total} 条会话记录")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                print(f"  ⚠️ 数据库表不存在: {e}")
            else:
                raise
        finally:
            conn.close()


def clear_all_caches(skip_confirm: bool = False):
    """清除所有缓存（Excel缓存、会话记录、临时文件等）
    
    Args:
        skip_confirm: 是否跳过确认提示（用于API调用）
    """
    print("\n⚠️  警告: 此操作将清除以下所有缓存:")
    print("  1. Excel缓存数据库（excel_metadata.db）")
    print("  2. Excel数据库文件（.duckdb 和 .db）")
    print("  3. 上传的Excel文件")
    print("  4. Excel聊天临时数据库")
    print("  5. 会话历史记录")
    print("  6. 文件服务器存储")
    print("  7. 模型缓存")
    
    if not skip_confirm:
        choice = input("\n⚠️  确认要清除所有缓存吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    # 清除Excel缓存
    print("\n1️⃣ 清除Excel缓存...")
    clear_cache_by_filename(None)
    
    # 清除Excel数据库文件
    print("\n2️⃣ 清除Excel数据库文件...")
    clear_excel_dbs(auto_confirm=True)
    
    # 清除上传的Excel文件
    print("\n3️⃣ 清除上传的Excel文件...")
    clear_uploaded_excel_files(auto_confirm=True)
    
    # 清除Excel聊天临时数据库
    print("\n4️⃣ 清除Excel聊天临时数据库...")
    clear_chat_excel_tmp(auto_confirm=True)
    
    # 清除会话历史
    print("\n5️⃣ 清除会话历史记录...")
    clear_chat_history(None, auto_confirm=True)
    
    # 清除文件服务器存储
    print("\n6️⃣ 清除文件服务器存储...")
    clear_file_server_storage(auto_confirm=True)
    
    # 清除模型缓存
    print("\n7️⃣ 清除模型缓存...")
    clear_model_cache(auto_confirm=True)
    
    print("\n✅ 所有缓存清除完成！")


if __name__ == "__main__":
    print("🗑️  缓存清理工具 (Excel缓存 + 会话聊天缓存)\n")
    
    if len(sys.argv) == 1:
        # 无参数：显示帮助信息
        print("使用方法:")
        print("\n📋 Excel缓存相关:")
        print("  python clear_excel_cache.py excel-list              # 列出Excel缓存")
        print("  python clear_excel_cache.py excel-clear <文件名>     # 删除指定文件的Excel缓存")
        print("  python clear_excel_cache.py excel-clear-id <ID>     # 根据ID删除Excel缓存")
        print("  python clear_excel_cache.py excel-clear-all         # 清除所有Excel缓存")
        print("  python clear_excel_cache.py excel-dbs-clear         # 清除Excel数据库文件")
        print("  python clear_excel_cache.py excel-files-clear       # 清除上传的Excel文件")
        print("  python clear_excel_cache.py excel-tmp-clear         # 清除Excel聊天临时数据库")
        
        print("\n💬 会话聊天相关:")
        print("  python clear_excel_cache.py chat-list                # 列出会话历史记录")
        print("  python clear_excel_cache.py chat-clear <会话ID>      # 删除指定会话记录")
        print("  python clear_excel_cache.py chat-clear-all           # 清除所有会话记录")
        
        print("\n📁 其他缓存:")
        print("  python clear_excel_cache.py file-server-clear        # 清除文件服务器存储")
        print("  python clear_excel_cache.py model-cache-clear        # 清除模型缓存")
        
        print("\n🗑️  全部清除:")
        print("  python clear_excel_cache.py clear-all                # 清除所有缓存")
    
    elif len(sys.argv) >= 2:
        command = sys.argv[1]
        
        # Excel缓存相关命令
        if command == "excel-list":
            list_cache()
        
        elif command == "excel-clear" and len(sys.argv) == 3:
            filename = sys.argv[2]
            clear_cache_by_filename(filename)
        
        elif command == "excel-clear-id" and len(sys.argv) == 3:
            cache_id = int(sys.argv[2])
            clear_cache_by_id(cache_id)
        
        elif command == "excel-clear-all":
            clear_cache_by_filename(None)
        
        elif command == "excel-dbs-clear":
            clear_excel_dbs()
        
        elif command == "excel-files-clear":
            clear_uploaded_excel_files()
        
        elif command == "excel-tmp-clear":
            clear_chat_excel_tmp()
        
        # 会话聊天相关命令
        elif command == "chat-list":
            list_chat_history()
        
        elif command == "chat-clear" and len(sys.argv) == 3:
            conv_uid = sys.argv[2]
            clear_chat_history(conv_uid)
        
        elif command == "chat-clear-all":
            clear_chat_history(None)
        
        # 其他缓存命令
        elif command == "file-server-clear":
            clear_file_server_storage()
        
        elif command == "model-cache-clear":
            clear_model_cache()
        
        # 全部清除
        elif command == "clear-all":
            # 检查是否从stdin读取到'yes'（API调用场景）
            skip_confirm = sys.stdin.readable() and not sys.stdin.isatty()
            clear_all_caches(skip_confirm=skip_confirm)
        
        # 兼容旧命令
        elif command == "list":
            list_cache()
        
        elif command == "clear" and len(sys.argv) == 3:
            filename = sys.argv[2]
            clear_cache_by_filename(filename)
        
        else:
            print("❌ 无效的命令")
            print("\n使用方法:")
            print("  python clear_excel_cache.py excel-list              # 列出Excel缓存")
            print("  python clear_excel_cache.py chat-list                # 列出会话历史")
            print("  python clear_excel_cache.py clear-all                # 清除所有缓存")
            print("\n使用 'python clear_excel_cache.py' 查看完整帮助")



