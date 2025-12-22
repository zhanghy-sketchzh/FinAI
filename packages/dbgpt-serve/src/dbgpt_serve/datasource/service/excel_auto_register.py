#!/usr/bin/env python3
"""
Excel 自动注册到数据源服务
支持自动缓存和增量导入
"""
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill

logger = logging.getLogger(__name__)


class ExcelCacheManager:
    """Excel 缓存管理器"""

    def __init__(self, cache_dir: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录，默认为 pilot/data/excel_cache
        """
        if cache_dir is None:
            # 使用相对路径
            current_dir = Path(__file__).parent
            cache_dir = current_dir.parent.parent.parent.parent.parent / "pilot" / "data" / "excel_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据数据库路径
        self.meta_db_path = self.cache_dir / "excel_metadata.db"
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """初始化元数据数据库"""
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS excel_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash VARCHAR(64) UNIQUE NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                table_name VARCHAR(255) NOT NULL,
                db_name VARCHAR(255) NOT NULL,
                db_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                columns_info TEXT NOT NULL,
                summary_prompt TEXT,
                data_schema_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        try:
            cursor.execute("SELECT data_schema_json FROM excel_metadata LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE excel_metadata ADD COLUMN data_schema_json TEXT")
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def calculate_excel_hash(df: pd.DataFrame, filename: str) -> str:
        """
        计算 Excel 的内容哈希值
        
        Args:
            df: DataFrame 对象
            filename: 文件名（包含在哈希计算中）
        
        Returns:
            SHA256 哈希值
        """
        # 组合文件名、列名和数据内容
        content_parts = [
            filename,
            ",".join(df.columns.tolist()),
            df.to_csv(index=False)
        ]
        content = "\n".join(content_parts)
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_cached_info(self, content_hash: str) -> Optional[Dict]:
        """
        根据内容哈希获取缓存信息
        
        Args:
            content_hash: 内容哈希值
        
        Returns:
            缓存信息字典，如果不存在则返回 None
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                content_hash, original_filename, table_name, db_name, db_path,
                row_count, column_count, columns_info, summary_prompt, data_schema_json,
                created_at, last_accessed, access_count
            FROM excel_metadata
            WHERE content_hash = ?
        """, (content_hash,))
        
        row = cursor.fetchone()
        
        if row:
            # 更新访问统计
            cursor.execute("""
                UPDATE excel_metadata
                SET last_accessed = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE content_hash = ?
            """, (content_hash,))
            conn.commit()
            
            result = {
                "content_hash": row[0],
                "original_filename": row[1],
                "table_name": row[2],
                "db_name": row[3],
                "db_path": row[4],
                "row_count": row[5],
                "column_count": row[6],
                "columns_info": json.loads(row[7]),
                "summary_prompt": row[8],
                "data_schema_json": row[9],
                "created_at": row[10],
                "last_accessed": row[11],
                "access_count": row[12]
            }
            conn.close()
            return result
        
        conn.close()
        return None
    
    def save_cache_info(
        self,
        content_hash: str,
        original_filename: str,
        table_name: str,
        db_name: str,
        db_path: str,
        df: pd.DataFrame,
        summary_prompt: str = None,
        data_schema_json: str = None
    ):
        """
        保存缓存信息
        
        Args:
            content_hash: 内容哈希值
            original_filename: 原始文件名
            table_name: 表名
            db_name: 数据库名
            db_path: 数据库路径
            df: DataFrame 对象
            summary_prompt: 数据理解提示词
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        columns_info = [
            {"name": col, "dtype": str(df[col].dtype)} 
            for col in df.columns
        ]
        
        cursor.execute("""
            INSERT OR REPLACE INTO excel_metadata
            (content_hash, original_filename, table_name, db_name, db_path,
             row_count, column_count, columns_info, summary_prompt, data_schema_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_hash,
            original_filename,
            table_name,
            db_name,
            db_path,
            len(df),
            len(df.columns),
            json.dumps(columns_info, ensure_ascii=False),
            summary_prompt,
            data_schema_json
        ))
        
        conn.commit()
        conn.close()
    
    def update_summary_prompt(self, content_hash: str, summary_prompt: str):
        """
        更新数据理解提示词
        
        Args:
            content_hash: 内容哈希值
            summary_prompt: 数据理解提示词
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE excel_metadata
            SET summary_prompt = ?
            WHERE content_hash = ?
        """, (summary_prompt, content_hash))
        
        conn.commit()
        conn.close()
    
    def delete_cache_by_hash(self, content_hash: str):
        """
        根据内容哈希删除缓存
        
        Args:
            content_hash: 内容哈希值
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM excel_metadata WHERE content_hash = ?", (content_hash,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted > 0
    
    def delete_cache_by_filename(self, filename: str):
        """
        根据文件名删除缓存
        
        Args:
            filename: 原始文件名
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM excel_metadata WHERE original_filename = ?", (filename,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted > 0
    
    def list_all_cache(self) -> List[Dict]:
        """
        列出所有缓存记录
        
        Returns:
            缓存记录列表
        """
        conn = sqlite3.connect(str(self.meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                content_hash, original_filename, table_name, db_name,
                row_count, column_count, created_at, last_accessed, access_count
            FROM excel_metadata
            ORDER BY last_accessed DESC
        """)
        
        records = cursor.fetchall()
        conn.close()
        
        result = []
        for row in records:
            result.append({
                "content_hash": row[0],
                "original_filename": row[1],
                "table_name": row[2],
                "db_name": row[3],
                "row_count": row[4],
                "column_count": row[5],
                "created_at": row[6],
                "last_accessed": row[7],
                "access_count": row[8]
            })
        
        return result


class ExcelAutoRegisterService:
    """Excel 自动注册到数据源服务"""

    _instance = None
    _lock = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            # 创建锁（如果还没有）
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()
            
            with cls._lock:
                # 双重检查
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, llm_client=None, model_name=None):
        """初始化服务"""
        if not hasattr(self, '_initialized'):
            self.cache_manager = ExcelCacheManager()
            self.llm_client = llm_client
            self.model_name = model_name
            
            current_dir = Path(__file__).parent
            base_dir = current_dir.parent.parent.parent.parent.parent / "pilot" / "meta_data"
            self.db_storage_dir = base_dir / "excel_dbs"
            self.db_storage_dir.mkdir(parents=True, exist_ok=True)
            
            self._initialized = True
        else:
            if llm_client is not None:
                self.llm_client = llm_client
            if model_name is not None:
                self.model_name = model_name
    
    def _get_cell_value(self, cell) -> Optional[str]:
        """获取单元格值，处理公式"""
        if cell.value is None:
            return None
        
        if cell.data_type == 'f':
            try:
                if isinstance(cell.value, str) and cell.value.startswith('='):
                    cleaned = self._clean_excel_formula(cell.value)
                    return cleaned if cleaned else None
            except Exception as e:
                logger.warning(f"获取公式计算结果失败: {e}")
        
        value_str = str(cell.value)
        return value_str.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
    
    def _get_cell_bg_color(self, cell) -> Optional[str]:
        """获取单元格背景色"""
        fill = cell.fill
        if fill.fill_type == 'solid' or fill.fill_type == 'patternFill':
            fg_color = fill.fgColor
            if fg_color.type == 'rgb':
                rgb = fg_color.rgb
                if rgb and len(rgb) == 8:
                    return rgb[2:]
                return rgb
            elif fg_color.type == 'indexed':
                return f"indexed_{fg_color.indexed}"
            elif fg_color.type == 'theme':
                tint = fg_color.tint if fg_color.tint else 0
                return f"theme_{fg_color.theme}_{tint:.2f}"
        return None
    
    def _detect_header_rows_with_color(self, excel_file_path: str) -> Tuple[List[int], Dict]:
        """使用颜色信息和LLM检测表头行"""
        wb = openpyxl.load_workbook(excel_file_path)
        ws = wb.active
        
        max_check_rows = min(20, ws.max_row)
        max_cols = ws.max_column
        
        rows_data = []
        rows_colors = []
        
        for row_idx in range(1, max_check_rows + 1):
            row_values = []
            row_colors = []
            for col_idx in range(1, max_cols + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell_value = self._get_cell_value(cell)
                row_values.append(cell_value if cell_value is not None else "")
                row_colors.append(self._get_cell_bg_color(cell))
            rows_data.append(row_values)
            rows_colors.append(row_colors)
        
        if self.llm_client:
            try:
                header_rows = self._detect_header_rows_with_llm_and_color(rows_data, rows_colors)
                if header_rows:
                    color_info = {
                        'rows_data': rows_data,
                        'rows_colors': rows_colors,
                        'header_rows': header_rows,
                        'max_cols': max_cols
                    }
                    return header_rows, color_info
            except Exception as e:
                logger.warning(f"LLM检测失败: {e}")
        
        df_raw = pd.DataFrame(rows_data)
        header_rows = self._detect_header_rows_rule_based(df_raw)
        color_info = {
            'rows_data': rows_data,
            'rows_colors': rows_colors,
            'header_rows': header_rows,
            'max_cols': max_cols
        }
        return header_rows, color_info
    
    def _detect_header_rows_with_llm_and_color(self, rows_data: List[List], rows_colors: List[List]) -> List[int]:
        """
        使用LLM和颜色信息检测表头行
        
        Args:
            rows_data: 前20行的数据
            rows_colors: 前20行的颜色信息
        
        Returns:
            表头行的索引列表（从0开始）
        """
        import asyncio
        import inspect
        from dbgpt.core import ModelRequest, ModelMessage, ModelMessageRoleType, ModelRequestContext
        
        # 构建表格文本表示（包含颜色信息）
        table_text = "行号\t列1\t列2\t列3\t...\t颜色信息\n"
        for idx, (row_data, row_colors) in enumerate(zip(rows_data[:20], rows_colors[:20])):
            # 只显示前10列数据
            row_values = [str(val) if val else "" for val in row_data[:10]]
            # 统计颜色分布
            color_counts = {}
            for color in row_colors:
                if color:
                    color_counts[color] = color_counts.get(color, 0) + 1
            color_info = ", ".join([f"{color[:8]}({count}列)" for color, count in color_counts.items()]) if color_counts else "无背景色"
            table_text += f"{idx}\t" + "\t".join(row_values) + f"\t[{color_info}]\n"
        
        # 构建prompt
        prompt = f"""你是一个Excel数据分析专家。请分析以下Excel文件的前20行数据，判断哪些行是表头行（列名行）。

注意：
1. 表头行通常包含列名，如"ID"、"日期"、"名称"、"金额"等
2. 可能有多级表头（多行表头），最后一行是最具体的列名
3. 表头行通常有特殊的背景色，同一级的表头通常使用相同或相似的背景色
4. 表头行之前可能有汇总信息行、说明行（如"请勿删除"、公式等），这些不是表头
5. 表头行之后是数据行，数据行通常没有背景色或使用不同的背景色
6. **如果发现中英文对照的标题行（如"Name"和"中英文名"），只保留中文标题行的索引，跳过英文标题行**
7. **忽略包含"@@"、"="等公式标记的行，这些是Excel内部标记行**
8. **优先选择包含中文列名的行作为表头，而不是英文列名的行**

请仔细分析数据内容和颜色信息，返回JSON格式：
{{
  "reason": "判断理由，说明为什么选择这些行作为表头，以及如何识别和过滤重复的中英文标题行",
  "header_rows": [行索引列表，从0开始],
}}

示例1：
如果第0行是"订单信息"（有蓝色背景），第1行是"行 ID, 订单 ID, 订单日期..."（有浅蓝色背景），第2行开始是数据（无背景色），则返回：
{{
  "header_rows": [0, 1],
  "reason": "第0-1行有背景色且包含表头关键词，第0行是分类标签，第1行是具体列名。第2行开始无背景色且内容为数据值"
}}

示例2：
如果第0-2行是汇总信息（无背景色或不同背景色），第3行是表头（有背景色且包含ID、日期等关键词），则返回：
{{
  "header_rows": [3],
  "reason": "第3行有特殊背景色且包含ID、日期等典型表头关键词，前3行是汇总信息"
}}

示例3（重复标题行）：
如果第2行是"Onboarding_Date, Staff_ID, Department..."（英文标题），第3行是"入职日期, 员工ID, 部门..."（中文标题），这两行表示相同的列，则只返回：
{{
  "header_rows": [3],
  "reason": "第2行是英文标题，第3行是中文标题，它们表示相同的列。根据规则，只保留中文标题行（第3行），跳过英文标题行（第2行）"
}}

现在请分析以下数据（右侧显示了每行的颜色分布）：

{table_text}

请返回JSON格式的结果："""
        
        # 调用LLM（非流式）
        request_params = {
            "messages": [
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)
            ],
            "temperature": 0.1,
            "max_new_tokens": 1000,
            "context": ModelRequestContext(stream=False)
        }
        
        if self.model_name:
            request_params["model"] = self.model_name
        
        request = ModelRequest(**request_params)
        
        # 获取响应
        stream_response = self.llm_client.generate_stream(request)
        
        full_text = ""
        if inspect.isasyncgen(stream_response):
            async def collect_async():
                text = ""
                async for chunk in stream_response:
                    chunk_text = self._extract_chunk_text(chunk)
                    if chunk_text:
                        text = chunk_text
                return text
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                full_text = loop.run_until_complete(collect_async())
            finally:
                loop.close()
        elif inspect.isgenerator(stream_response):
            for chunk in stream_response:
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_text = chunk_text
        else:
            raise Exception(f"Unexpected response type: {type(stream_response)}")
        
        # 解析JSON结果
        try:
            # 提取JSON部分
            start_idx = full_text.find('{')
            end_idx = full_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = full_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                header_rows = result.get('header_rows', [])
                reason = result.get('reason', '')
                
                logger.info(f"LLM判断理由: {reason}")
                
                # 验证结果
                if isinstance(header_rows, list) and all(isinstance(x, int) for x in header_rows):
                    # 确保索引在有效范围内
                    valid_rows = [r for r in header_rows if 0 <= r < len(rows_data)]
                    if valid_rows:
                        return sorted(valid_rows)
                    else:
                        logger.warning(f"LLM返回的表头行索引无效: {header_rows}")
                        return None
                else:
                    logger.warning(f"LLM返回的表头行格式无效: {header_rows}")
                    return None
            else:
                logger.warning(f"无法从LLM输出中提取JSON: {full_text[:200]}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"LLM输出JSON解析失败: {e}, 输出: {full_text[:200]}")
            return None
    
    def _merge_headers_by_color(self, color_info: Dict) -> List[str]:
        """
        基于颜色信息合并表头
        
        策略：
        1. 对于每个表头行，识别同一颜色的列
        2. 如果同一颜色的列中只有一个单元格有值，用该值填充其他列
        3. 将多行表头合并为单行，用"-"连接
        
        Args:
            color_info: 颜色信息字典
        
        Returns:
            合并后的表头列表
        """
        rows_data = color_info['rows_data']
        rows_colors = color_info['rows_colors']
        header_rows = color_info['header_rows']
        max_cols = color_info['max_cols']
        
        # 提取表头行的数据和颜色
        header_data = [rows_data[i] for i in header_rows]
        header_colors = [rows_colors[i] for i in header_rows]
        
        # 对每一行表头，按颜色和位置连续性分组
        filled_headers = []
        for row_idx, (row_data, row_colors) in enumerate(zip(header_data, header_colors)):
            color_position_groups = []
            current_group = None
            current_color = None
            
            for col_idx, (value, color) in enumerate(zip(row_data, row_colors)):
                if color:
                    if (color == current_color and 
                        current_group and 
                        col_idx == current_group[-1][0] + 1):
                        current_group.append((col_idx, value))
                    else:
                        if current_group:
                            color_position_groups.append((current_color, current_group))
                        current_group = [(col_idx, value)]
                        current_color = color
                else:
                    if current_group:
                        color_position_groups.append((current_color, current_group))
                        current_group = None
                        current_color = None
            
            if current_group:
                color_position_groups.append((current_color, current_group))
            
            filled_row = list(row_data)
            for color, cells in color_position_groups:
                non_empty_values = [(idx, val) for idx, val in cells if val and str(val).strip()]
                
                if len(non_empty_values) == 1:
                    fill_value = non_empty_values[0][1]
                    for col_idx, _ in cells:
                        filled_row[col_idx] = fill_value
                elif len(non_empty_values) > 1:
                    non_empty_indices = sorted([idx for idx, _ in non_empty_values])
                    
                    for i, (val_idx, val) in enumerate(sorted(non_empty_values, key=lambda x: x[0])):
                        start_idx = val_idx
                        if i < len(non_empty_indices) - 1:
                            end_idx = non_empty_indices[i + 1]
                        else:
                            end_idx = max([idx for idx, _ in cells]) + 1
                        
                        cells_to_fill = [(idx, v) for idx, v in cells if start_idx <= idx < end_idx]
                        for col_idx, _ in cells_to_fill:
                            filled_row[col_idx] = val
            
            filled_headers.append(filled_row)
        
        # 合并多级表头
        combined_headers = []
        for col_idx in range(max_cols):
            # 收集该列的所有非空值（从上层到底层）
            col_values = []
            for row_idx in range(len(filled_headers)):
                val = filled_headers[row_idx][col_idx]
                if val and str(val).strip():
                    val_str = str(val).strip()
                    # 去除换行符、普通空格和不间断空格（在合并前就清理）
                    val_str = val_str.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace('\u00A0', '')
                    # 避免重复值
                    if not col_values or val_str != col_values[-1]:
                        col_values.append(val_str)
            
            # 用"-"连接多级表头
            if col_values:
                combined = "-".join(col_values)
            else:
                combined = f'Column_{col_idx}'
            
            combined_headers.append(combined)
        
        return combined_headers
    
    def _process_multi_level_header(self, df_raw: pd.DataFrame, excel_file_path: str) -> pd.DataFrame:
        """处理多级表头"""
        import numpy as np
        
        header_rows, color_info = self._detect_header_rows_with_color(excel_file_path)
        
        if not header_rows:
            header_rows = [0]
        
        combined_headers = self._merge_headers_by_color(color_info)
        
        cleaned_headers = []
        for header in combined_headers:
            cleaned = str(header)
            parts = cleaned.split('-')
            valid_parts = [p for p in parts if '=' not in p and '@@' not in p and '@' not in p 
                          and p.strip() and p.strip() not in ['-', '_', '']]
            
            if valid_parts:
                cleaned = '-'.join(valid_parts)
            else:
                cleaned = f'Column_{len(cleaned_headers)}'
            
            cleaned = cleaned.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace('\u00A0', '')
            cleaned = cleaned.replace('--', '-').replace('__', '_').strip('-_')
            
            if not cleaned or cleaned in ['-', '_']:
                cleaned = f'Column_{len(cleaned_headers)}'
            
            cleaned_headers.append(cleaned)
        
        final_headers = []
        seen = {}
        for header in cleaned_headers:
            if header in seen:
                seen[header] += 1
                final_headers.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                final_headers.append(header)
        
        data_start_row = max(header_rows) + 1
        data_df = df_raw.iloc[data_start_row:].copy()
        
        if len(final_headers) > len(data_df.columns):
            final_headers = final_headers[:len(data_df.columns)]
        elif len(final_headers) < len(data_df.columns):
            for i in range(len(final_headers), len(data_df.columns)):
                final_headers.append(f'Column_{i}')
        
        data_df.columns = final_headers
        data_df = data_df.dropna(how='all').reset_index(drop=True)
        
        return data_df
    
    def _clean_excel_formula(self, text: str) -> str:
        """清理Excel公式和特殊字符"""
        if not text:
            return text
        
        text_str = str(text)
        
        if text_str.startswith('='):
            import re
            quoted_texts = re.findall(r'["\']([^"\']+)["\']', text_str)
            if quoted_texts:
                text_str = ''.join(quoted_texts)
            else:
                cleaned = re.sub(r'[=&()]', '', text_str)
                cleaned = re.sub(r'CHAR\s*\(\s*\d+\s*\)', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'CONCATENATE\s*\([^)]*\)', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_]', '', cleaned)
                if not cleaned:
                    return ""
                text_str = cleaned
        
        if '@@' in text_str or text_str.startswith('@'):
            import re
            text_str = re.sub(r'@@[^\u4e00-\u9fff-]*', '', text_str)
            text_str = text_str.replace('@', '')
            if not text_str.strip():
                return ""
        
        text_str = text_str.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
        return text_str
    
    def _remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除完全为空的列"""
        df_cleaned = df.dropna(axis=1, how='all')
        
        empty_cols = [col for col in df_cleaned.columns 
                     if df_cleaned[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).all()]
        
        if empty_cols:
            df_cleaned = df_cleaned.drop(columns=empty_cols)
        
        return df_cleaned
    
    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除列名和数据值都完全重复的列"""
        columns_to_remove = []
        
        for i, col1 in enumerate(df.columns):
            if col1 in columns_to_remove:
                continue
            
            for col2 in df.columns[i+1:]:
                if col2 in columns_to_remove or col1 != col2:
                    continue
                
                try:
                    if df[col1].equals(df[col2]):
                        columns_to_remove.append(col2)
                except:
                    try:
                        if df[col1].fillna('__NULL__').equals(df[col2].fillna('__NULL__')):
                            columns_to_remove.append(col2)
                    except:
                        pass
        
        if columns_to_remove:
            df_cleaned = df.drop(columns=columns_to_remove)
        else:
            df_cleaned = df
        
        return df_cleaned
    
    def _format_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化日期列为YYYY-MM-DD格式"""
        df_formatted = df.copy()
        
        for col in df_formatted.columns:
            if pd.api.types.is_datetime64_any_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
                )
            elif df_formatted[col].dtype == 'object':
                non_null_values = df_formatted[col].dropna()
                if len(non_null_values) > 0:
                    sample_val = non_null_values.iloc[0]
                    if isinstance(sample_val, (pd.Timestamp, datetime)):
                        df_formatted[col] = df_formatted[col].apply(
                            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and isinstance(x, (pd.Timestamp, datetime)) else x
                        )
        
        return df_formatted
    
    def _detect_header_rows_rule_based(self, df_raw: pd.DataFrame) -> List[int]:
        """基于规则的表头行检测"""
        import numpy as np
        
        header_keywords = [
            'id', 'ID', '编号', '序号', '行', '订单',
            '日期', 'date', 'Date', '时间', 'time', 'Time',
            '名称', 'name', 'Name', '客户', '产品',
            '金额', '价格', 'price', 'Price', '销售额', '利润',
            '数量', 'quantity', 'Quantity',
            '类别', 'category', 'Category', '类型', 'type', 'Type',
            '区域', 'region', 'Region', '城市', 'city', 'City',
            '信息', 'info', 'Info', '数据', 'data', 'Data'
        ]
        
        max_check_rows = min(20, len(df_raw))
        candidate_rows = []
        
        for i in range(max_check_rows):
            row = df_raw.iloc[i]
            non_null_count = row.notna().sum()
            row_text = ' '.join([str(val) for val in row if pd.notna(val)])
            keyword_matches = sum(1 for keyword in header_keywords if keyword.lower() in row_text.lower())
            score = non_null_count + keyword_matches * 2
            
            if score > 0:
                candidate_rows.append((i, score, non_null_count, keyword_matches))
        
        if not candidate_rows:
            return [0]
        
        candidate_rows.sort(key=lambda x: x[1], reverse=True)
        main_header_row = candidate_rows[0][0]
        header_rows = [main_header_row]
        
        for offset in range(1, min(4, main_header_row + 1)):
            check_row = main_header_row - offset
            if check_row >= 0:
                row = df_raw.iloc[check_row]
                if row.notna().sum() >= 2:
                    header_rows.insert(0, check_row)
                else:
                    break
        
        return sorted(header_rows)
    
    def process_excel(
        self,
        excel_file_path: str,
        table_name: str = None,
        force_reimport: bool = False,
        original_filename: str = None,
        conv_uid: str = None
    ) -> Dict:
        """处理Excel文件，自动注册到数据源"""
        df_raw = pd.read_excel(excel_file_path, header=None)
        
        df = self._process_multi_level_header(df_raw, excel_file_path)
        
        if original_filename is None:
            original_filename = Path(excel_file_path).name
        
        content_hash = self.cache_manager.calculate_excel_hash(df, original_filename)
        
        if not force_reimport:
            cached_info = self.cache_manager.get_cached_info(content_hash)
            if cached_info and os.path.exists(cached_info["db_path"]):
                cached_schema_json = cached_info.get("data_schema_json")
                
                top_10_rows_raw = df.head(10).values.tolist()
                top_10_rows = self._convert_to_json_serializable(top_10_rows_raw)
                    
                return {
                    "status": "cached",
                    "message": "使用缓存数据",
                    "content_hash": content_hash,
                    "db_name": cached_info["db_name"],
                    "db_path": cached_info["db_path"],
                    "table_name": cached_info["table_name"],
                    "row_count": cached_info["row_count"],
                    "column_count": cached_info["column_count"],
                    "columns_info": cached_info["columns_info"],
                    "summary_prompt": cached_info["summary_prompt"],
                    "data_schema_json": cached_schema_json,
                    "top_10_rows": top_10_rows,
                    "access_count": cached_info["access_count"],
                    "last_accessed": cached_info["last_accessed"],
                    "conv_uid": conv_uid
                }
        
        if table_name is None:
            base_name = Path(original_filename).stem
            base_name = "".join(c if c.isalnum() or c == '_' else '_' for c in base_name)
            if base_name and base_name[0].isdigit():
                base_name = f"tbl_{base_name}"
            if not base_name or len(base_name) < 2:
                base_name = f"excel_table_{content_hash[:8]}"
            table_name = base_name
        
        db_name = f"excel_{content_hash[:8]}"
        db_filename = f"{db_name}.db"
        db_path = str(self.db_storage_dir / db_filename)
        
        df = self._remove_empty_columns(df)
        df = self._remove_duplicate_columns(df)
        df = self._format_date_columns(df)
        df.columns = [str(col).replace(' ', '').replace('\u00A0', '').replace('\n', '').replace('\r', '').replace('\t', '') for col in df.columns]
        
        # 清理后再次去重列名,防止SQLite报duplicate column name错误
        final_columns = []
        seen_columns = {}
        for col in df.columns:
            if col in seen_columns:
                seen_columns[col] += 1
                final_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                final_columns.append(col)
        df.columns = final_columns
        
        conn = sqlite3.connect(db_path)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.commit()
        except Exception as e:
            conn.close()
            logger.error(f"数据写入SQLite失败: {e}")
            raise Exception(f"Excel数据转换为数据库失败: {e}")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        conn.close()
        
        columns_info = [{"name": col[1], "type": col[2], "dtype": str(df[col[1]].dtype)} for col in columns]
        
        schema_understanding_json = self._generate_schema_understanding_with_llm(df, table_name)
        summary_prompt = self._format_schema_as_prompt(schema_understanding_json, df, table_name)
        
        self.cache_manager.save_cache_info(
            content_hash=content_hash,
            original_filename=original_filename,
            table_name=table_name,
            db_name=db_name,
            db_path=db_path,
            df=df,
            summary_prompt=summary_prompt,
            data_schema_json=schema_understanding_json
        )
        
        self._register_to_dbgpt(db_name, db_path, table_name)
        
        top_10_rows_raw = df.head(10).values.tolist()
        top_10_rows = self._convert_to_json_serializable(top_10_rows_raw)
        
        return {
            "status": "imported",
            "message": "成功导入新数据",
            "content_hash": content_hash,
            "db_name": db_name,
            "db_path": db_path,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_info": columns_info,
            "summary_prompt": summary_prompt,
            "data_schema_json": schema_understanding_json,
            "top_10_rows": top_10_rows,
            "conv_uid": conv_uid
        }
    
    
    def _generate_schema_understanding_with_llm(self, df: pd.DataFrame, table_name: str) -> str:
        """使用LLM生成Schema理解JSON"""
        er_info = self._prepare_er_info(df, table_name)
        numeric_stats = self._prepare_numeric_stats(df)
        categorical_distribution = self._prepare_categorical_distribution(df)
        
        prompt = self._build_schema_understanding_prompt(
            table_name=table_name,
            er_info=er_info,
            numeric_stats=numeric_stats,
            categorical_distribution=categorical_distribution,
            sample_data=df.head(3).to_dict('records')
        )
        
        # 调用LLM生成简化的Schema JSON（只包含业务理解字段）
        simplified_json = self._call_llm_for_schema(prompt)
        
        # 通过代码补充技术性字段，生成完整的Schema JSON
        enriched_json = self._enrich_schema_json(simplified_json, df, table_name)
        
        return enriched_json
       
    
    def _prepare_er_info(self, df: pd.DataFrame, table_name: str) -> str:
        """准备ER信息（表结构）"""
        er_lines = [f"表名: {table_name}"]
        er_lines.append(f"行数: {len(df)}")
        er_lines.append(f"列数: {len(df.columns)}")
        er_lines.append("\n字段列表:")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            er_lines.append(f"  - {col} ({dtype}, 缺失率: {null_pct:.1f}%)")
        
        return "\n".join(er_lines)
    
    def _prepare_numeric_stats(self, df: pd.DataFrame) -> str:
        """准备数值列的描述统计"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return "无数值列"
        
        stats_lines = ["数值列描述统计:"]
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_lines.append(f"\n  {col}:")
                stats_lines.append(f"    最小值: {col_data.min():.2f}")
                stats_lines.append(f"    最大值: {col_data.max():.2f}")
                stats_lines.append(f"    平均值: {col_data.mean():.2f}")
                stats_lines.append(f"    中位数: {col_data.median():.2f}")
                stats_lines.append(f"    标准差: {col_data.std():.2f}")
        
        return "\n".join(stats_lines)
    
    def _prepare_categorical_distribution(self, df: pd.DataFrame) -> str:
        """准备分类列的唯一值分布"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return "无分类列"
        
        dist_lines = ["分类列唯一值分布:"]
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            unique_count = len(unique_vals)
            
            dist_lines.append(f"\n  {col} (唯一值数量: {unique_count}):")
            
            if unique_count <= 20:
                # 显示所有唯一值
                value_counts = df[col].value_counts()
                for val, count in value_counts.head(20).items():
                    dist_lines.append(f"    - '{val}': {count}条 ({count/len(df)*100:.1f}%)")
            else:
                # 只显示前10个最常见的值
                value_counts = df[col].value_counts()
                dist_lines.append(f"    前10个最常见值:")
                for val, count in value_counts.head(10).items():
                    dist_lines.append(f"    - '{val}': {count}条 ({count/len(df)*100:.1f}%)")
        
        return "\n".join(dist_lines)
    
    def _build_schema_understanding_prompt(
        self, 
        table_name: str,
        er_info: str,
        numeric_stats: str,
        categorical_distribution: str,
        sample_data: list
    ) -> str:
        """构建Schema理解Prompt（简化版，只生成必要的业务理解字段）"""
        
        # 转换sample_data中的特殊类型（如Timestamp）为可JSON序列化的格式
        def convert_to_serializable(obj):
            """递归转换对象为可JSON序列化的格式"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            elif hasattr(obj, 'isoformat'):  # datetime, date, time, Timestamp
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_sample_data = convert_to_serializable(sample_data)
        sample_data_str = json.dumps(serializable_sample_data, ensure_ascii=False, indent=2)
        
        prompt = f"""你是一个数据分析专家，请分析以下数据表的结构和语义，生成Schema理解的JSON。

=== 数据表ER信息 ===
{er_info}

=== 数值列描述统计 ===
{numeric_stats}

=== 分类列唯一值分布 ===
{categorical_distribution}

=== 数据示例（前3行） ===
{sample_data_str}

请生成一个简化的JSON格式，只包含需要业务理解的核心信息：

1. **table_description**: 表的整体描述，说明这是什么数据，适合做什么分析
2. **columns**: 每个字段的业务理解信息（只需要以下字段）：
   - column_name: 字段名（必须使用完整的字段名，不能删减）
   - semantic_type: 语义类型（如：时间维度、地域维度、数值指标、分类维度、标识字段等）
   - description: 字段的业务含义和用途描述

请严格按照以下JSON格式输出（只包含上述字段）：

```json
{{
  "table_description": "表的整体描述...",
  "columns": [
    {{
      "column_name": "完整的字段名",
      "semantic_type": "时间维度/地域维度/数值指标/分类维度/标识字段",
      "description": "详细的业务含义描述"
    }}
  ]
}}
```

注意：
1. 深入理解字段的业务含义，不要只是简单重复字段名
2. semantic_type要准确，这对后续分析非常重要
3. column_name必须与数据表中的字段名完全一致

请直接输出JSON，不要有其他文字：
"""
        return prompt
    
    def _extract_chunk_text(self, chunk) -> str:
        """统一的chunk文本提取方法"""
        try:
            if hasattr(chunk, 'text'):
                return chunk.text
            elif hasattr(chunk, 'content'):
                if hasattr(chunk.content, 'get_text'):
                    try:
                        return chunk.content.get_text()
                    except:
                        pass
                elif isinstance(chunk.content, str):
                    return chunk.content
        except Exception as e:
            logger.debug(f"提取chunk文本失败: {e}")
        return ""
    
    def _enrich_schema_json(self, simplified_json: str, df: pd.DataFrame, table_name: str) -> str:
        """通过代码补充技术性字段，生成完整的Schema JSON"""
        try:
            schema = json.loads(simplified_json)
        except json.JSONDecodeError as e:
            logger.error(f"解析简化JSON失败: {e}")
            raise
        
        # 构建字段名映射
        llm_map = {col.get('column_name'): col for col in schema.get('columns', []) if col.get('column_name')}
        
        # 构建完整的columns列表
        enriched_columns = []
        for col_name in df.columns:
            col_data = df[col_name]
            dtype = str(col_data.dtype)
            llm_info = llm_map.get(col_name, {})
            
            col_info = {
                "column_name": col_name,
                "data_type": dtype,
                "semantic_type": llm_info.get('semantic_type', '未知'),
                "description": llm_info.get('description', f'{col_name}字段'),
                "is_key_field": self._is_potential_key_field(col_name, col_data)
            }
            
            # 分类字段：唯一值和分布
            if dtype in ['object', 'category'] and '时间' not in col_info['semantic_type'] and '日期' not in col_info['semantic_type']:
                unique_vals = col_data.dropna().unique().tolist()
                col_info["unique_values"] = [str(v) for v in unique_vals[:50]]
                value_counts = col_data.value_counts()
                col_info["value_distribution"] = (
                    f"共{len(unique_vals)}个唯一值" if len(value_counts) <= 10
                    else f"共{len(unique_vals)}个唯一值，前5个: {', '.join([str(v) for v in value_counts.head(5).index])}"
                )
            
            # 数值字段：统计信息
            elif dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    col_info["statistics_summary"] = (
                        f"范围: [{numeric_data.min():.2f}, {numeric_data.max():.2f}], "
                        f"均值: {numeric_data.mean():.2f}, 中位数: {numeric_data.median():.2f}"
                    )
            
            enriched_columns.append(col_info)
        
        return json.dumps({
            "table_name": table_name,
            "table_description": schema.get('table_description', ''),
            "columns": enriched_columns
        }, ensure_ascii=False, indent=2)
    
    def _call_llm_for_schema(self, prompt: str) -> str:
        """调用LLM生成Schema JSON"""
        try:
            from dbgpt.core import ModelRequest, ModelMessage, ModelMessageRoleType
            import asyncio
            import logging
            
            worker_logger = logging.getLogger('dbgpt.model.cluster.worker.default_worker')
            original_level = worker_logger.level
            
            try:
                worker_logger.setLevel(logging.ERROR)
                
                request_params = {
                    "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
                    "temperature": 0.1,
                    "max_new_tokens": 20480,
                }
                
                if self.model_name:
                    request_params["model"] = self.model_name
                
                request = ModelRequest(**request_params)
                
                if hasattr(self.llm_client, 'generate_stream'):
                    import inspect
                    stream_response = self.llm_client.generate_stream(request)
                    
                    full_text = ""
                    if inspect.isasyncgen(stream_response):
                        async def collect_async():
                            text = ""
                            async for chunk in stream_response:
                                chunk_text = self._extract_chunk_text(chunk)
                                if chunk_text:
                                    text = chunk_text
                            return text
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            full_text = loop.run_until_complete(collect_async())
                        finally:
                            loop.close()
                    elif inspect.isgenerator(stream_response):
                        for chunk in stream_response:
                            chunk_text = self._extract_chunk_text(chunk)
                            if chunk_text:
                                full_text = chunk_text
                    else:
                        raise Exception(f"Unexpected response type: {type(stream_response)}")
                    
                    class FakeResponse:
                        def __init__(self, text):
                            self.text = text
                    response = FakeResponse(full_text)
                    
                else:
                    raise Exception("LLM客户端没有generate_stream方法")
                
            finally:
                worker_logger.setLevel(original_level)
            
            if response and hasattr(response, 'text') and response.text:
                text = response.text.strip()
                json_str = None
                
                if "```json" in text.lower():
                    start_idx = text.lower().find("```json")
                    if start_idx >= 0:
                        content_start = text.find('\n', start_idx) + 1
                        if content_start > 0:
                            end_idx = text.find("```", content_start)
                            if end_idx > content_start:
                                json_str = text[content_start:end_idx].strip()
                
                if not json_str and "```" in text:
                    start_idx = text.find("```")
                    content_start = text.find('\n', start_idx) + 1
                    if content_start > 0:
                        end_idx = text.find("```", content_start)
                        if end_idx > content_start:
                            json_str = text[content_start:end_idx].strip()
                
                if not json_str:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = text[start:end].strip()
                
                if not json_str:
                    raise Exception("无法提取JSON内容")
                
                try:
                    parsed = json.loads(json_str)
                    return json_str
                except json.JSONDecodeError as e:
                    logger.error(f"JSON格式错误: {e}")
                    raise
            else:
                raise Exception("LLM返回空结果")
                
        except Exception as e:
            logger.error(f"调用LLM失败: {e}")
            raise
    
    
    
    def _format_schema_as_prompt(self, schema_json: str, df: pd.DataFrame, table_name: str) -> str:
        """
        将Schema JSON格式化为文本prompt
        用于后续的query改写和SQL生成
        """
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回简单格式
            return f"数据表: {table_name}\n数据规模: {len(df)}行 × {len(df.columns)}列"
        
        # 构建易读的文本格式
        lines = []
        lines.append(f"=== 数据表Schema理解 ===")
        lines.append(f"表名: {schema.get('table_name', table_name)}")
        lines.append(f"表描述: {schema.get('table_description', '')}")
        lines.append(f"")
        
        lines.append(f"=== 字段详细信息 ===")
        for col in schema.get('columns', []):
            lines.append(f"\n字段: {col.get('column_name')}")
            lines.append(f"  类型: {col.get('data_type')}")
            lines.append(f"  语义: {col.get('semantic_type')}")
            lines.append(f"  描述: {col.get('description')}")
            
            if 'unique_values' in col:
                unique_vals = col['unique_values']
                if len(unique_vals) <= 10:
                    lines.append(f"  可选值: {', '.join([str(v) for v in unique_vals])}")
                else:
                    lines.append(f"  可选值: {', '.join([str(v) for v in unique_vals[:10]])}... (共{len(unique_vals)}个)")
            
            if 'statistics_summary' in col:
                lines.append(f"  统计: {col['statistics_summary']}")
        
        return "\n".join(lines)
    
    
    def _get_sample_values(self, col_data: pd.Series, n: int = 3) -> list:
        """
        获取列的示例值
        """
        non_null_values = col_data.dropna().unique()
        if len(non_null_values) > 0:
            sample = non_null_values[:n].tolist()
            return [str(v) for v in sample]
        return []
    
    def _is_potential_key_field(self, col_name: str, col_data: pd.Series) -> bool:
        """
        判断是否是潜在的关键字段
        """
        col_name_lower = col_name.lower()
        
        # 基于列名判断
        if any(keyword in col_name_lower for keyword in ['id', '编号', 'key', 'code']):
            return True
        
        # 基于唯一性判断
        if len(col_data.dropna().unique()) == len(col_data.dropna()):
            return True
        
        return False
    
    def _convert_to_json_serializable(self, obj):
        """
        递归转换对象为可JSON序列化的格式
        
        Args:
            obj: 要转换的对象
        
        Returns:
            可JSON序列化的对象
        """
        from datetime import datetime, date
        
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            # 如果是日期类型，检查时间部分是否为00:00:00，如果是则只返回日期部分
            if isinstance(obj, pd.Timestamp):
                if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                    return obj.strftime('%Y-%m-%d')
                else:
                    return obj.isoformat()
            elif isinstance(obj, datetime):
                if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                    return obj.strftime('%Y-%m-%d')
                else:
                    return obj.isoformat()
            elif isinstance(obj, date):
                return obj.strftime('%Y-%m-%d')
            else:
                return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    
    
    
    def _register_to_dbgpt(self, db_name: str, db_path: str, table_name: str):
        """注册到 DB-GPT 数据源管理器"""
        try:
            logger.info(f"SQLite数据库已创建: {db_name}, 路径: {db_path}, 表名: {table_name}")
            return
        except Exception as e:
            logger.warning(f"注册到 DB-GPT 失败: {e}")
    
    def update_summary_prompt(self, content_hash: str, summary_prompt: str):
        """
        更新数据理解提示词
        
        Args:
            content_hash: 内容哈希值
            summary_prompt: 新的数据理解提示词
        """
        self.cache_manager.update_summary_prompt(content_hash, summary_prompt)
    
    def get_excel_info(self, content_hash: str) -> Optional[Dict]:
        """
        获取 Excel 信息
        
        Args:
            content_hash: 内容哈希值
        
        Returns:
            Excel 信息字典
        """
        return self.cache_manager.get_cached_info(content_hash)


def main():
    """测试主函数"""
    llm_client = None
    model_name = None
    
    try:
        import sys
        from pathlib import Path
        
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            packages_path = project_root / "packages"
            if str(packages_path) not in sys.path:
                sys.path.insert(0, str(packages_path))
        
        from dbgpt.model.cluster.client import DefaultLLMClient
        from dbgpt.model.cluster.manager import get_worker_manager
        
        worker_manager = get_worker_manager()
        if worker_manager:
            llm_client = DefaultLLMClient(worker_manager)
            from dbgpt.configs.config import CFG
            model_name = getattr(CFG, 'LLM_MODEL', None)
            print(f"LLM客户端已初始化, 模型: {model_name or '默认'}")
        else:
            print("LLM不可用, 使用规则方法")
    except Exception as e:
        print(f"LLM初始化失败: {e}, 使用规则方法")
    
    service = ExcelAutoRegisterService(llm_client=llm_client, model_name=model_name)
    excel_path = "/Users/luchun/Desktop/work/DB-GPT/示例-超市_多级表头2.xlsx"
    
    print("\n第一次导入...")
    result1 = service.process_excel(excel_path)
    print(f"状态: {result1['status']}, 数据库: {result1['db_name']}, "
          f"行数: {result1['row_count']}, 列数: {result1['column_count']}")
    
    
    print("\n第二次导入(缓存)...")
    result2 = service.process_excel(excel_path)
    print(f"状态: {result2['status']}, 访问次数: {result2.get('access_count', 0)}")
    print("\n测试完成")


if __name__ == "__main__":
    main()

