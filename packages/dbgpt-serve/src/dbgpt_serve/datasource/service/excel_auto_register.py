#!/usr/bin/env python3
"""
Excel 自动注册到数据源服务
支持自动缓存和增量导入
"""
# ruff: noqa: E501

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl
import pandas as pd

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
            cache_dir = (
                current_dir.parent.parent.parent.parent.parent
                / "pilot"
                / "data"
                / "excel_cache"
            )

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
            cursor.execute(
                "ALTER TABLE excel_metadata ADD COLUMN data_schema_json TEXT"
            )

        conn.commit()
        conn.close()

    @staticmethod
    def calculate_excel_hash(df: pd.DataFrame, filename: str) -> str:
        """
        计算 Excel 的内容哈希值（基于DataFrame，用于向后兼容）

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
            df.to_csv(index=False),
        ]
        content = "\n".join(content_parts)

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_file_hash(file_path: str, sheet_names: List[str] = None) -> str:
        """
        计算文件级别的哈希值（基于文件内容和sheet列表）

        Args:
            file_path: Excel文件路径
            sheet_names: 要处理的sheet名称列表（如果为None则不考虑sheet信息）

        Returns:
            SHA256 哈希值
        """
        sha256_hash = hashlib.sha256()

        # 读取文件内容并计算哈希
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件占用过多内存
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        # 如果指定了sheet_names，将其也纳入哈希计算（确保不同sheet组合产生不同哈希）
        if sheet_names:
            sheet_info = ",".join(sorted(sheet_names))
            sha256_hash.update(sheet_info.encode("utf-8"))

        return sha256_hash.hexdigest()

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

        cursor.execute(
            """
            SELECT 
                content_hash, original_filename, table_name, db_name, db_path,
                row_count, column_count, columns_info, summary_prompt, data_schema_json,
                created_at, last_accessed, access_count
            FROM excel_metadata
            WHERE content_hash = ?
        """,
            (content_hash,),
        )

        row = cursor.fetchone()

        if row:
            # 更新访问统计
            cursor.execute(
                """
                UPDATE excel_metadata
                SET last_accessed = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE content_hash = ?
            """,
                (content_hash,),
            )
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
                "access_count": row[12],
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
        data_schema_json: str = None,
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
            {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
        ]

        cursor.execute(
            """
            INSERT OR REPLACE INTO excel_metadata
            (content_hash, original_filename, table_name, db_name, db_path,
             row_count, column_count, columns_info, summary_prompt, data_schema_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                content_hash,
                original_filename,
                table_name,
                db_name,
                db_path,
                len(df),
                len(df.columns),
                json.dumps(columns_info, ensure_ascii=False),
                summary_prompt,
                data_schema_json,
            ),
        )

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

        cursor.execute(
            """
            UPDATE excel_metadata
            SET summary_prompt = ?
            WHERE content_hash = ?
        """,
            (summary_prompt, content_hash),
        )

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

        cursor.execute(
            "DELETE FROM excel_metadata WHERE content_hash = ?", (content_hash,)
        )
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

        cursor.execute(
            "DELETE FROM excel_metadata WHERE original_filename = ?", (filename,)
        )
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
            result.append(
                {
                    "content_hash": row[0],
                    "original_filename": row[1],
                    "table_name": row[2],
                    "db_name": row[3],
                    "row_count": row[4],
                    "column_count": row[5],
                    "created_at": row[6],
                    "last_accessed": row[7],
                    "access_count": row[8],
                }
            )

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
        if not hasattr(self, "_initialized"):
            self.cache_manager = ExcelCacheManager()
            self.llm_client = llm_client
            self.model_name = model_name

            current_dir = Path(__file__).parent
            base_dir = (
                current_dir.parent.parent.parent.parent.parent / "pilot" / "meta_data"
            )
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

        if cell.data_type == "f":
            try:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cleaned = self._clean_excel_formula(cell.value)
                    return cleaned if cleaned else None
            except Exception as e:
                logger.warning(f"获取公式计算结果失败: {e}")

        value_str = str(cell.value)
        return (
            value_str.replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
            .replace(" ", "")
        )

    def _get_cell_bg_color(self, cell) -> Optional[str]:
        """获取单元格背景色"""
        fill = cell.fill
        if fill.fill_type == "solid" or fill.fill_type == "patternFill":
            fg_color = fill.fgColor
            if fg_color.type == "rgb":
                rgb = fg_color.rgb
                if rgb and len(rgb) == 8:
                    return rgb[2:]
                return rgb
            elif fg_color.type == "indexed":
                return f"indexed_{fg_color.indexed}"
            elif fg_color.type == "theme":
                tint = fg_color.tint if fg_color.tint else 0
                return f"theme_{fg_color.theme}_{tint:.2f}"
        return None

    def _detect_header_rows_with_color(
        self, excel_file_path: str, sheet_name: str = None
    ) -> Tuple[List[int], Dict]:
        """使用颜色信息和LLM检测表头行

        Args:
            excel_file_path: Excel文件路径
            sheet_name: sheet名称，如果为None则使用active sheet
        """
        wb = openpyxl.load_workbook(excel_file_path)
        ws = wb[sheet_name] if sheet_name else wb.active

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
                header_rows = self._detect_header_rows_with_llm_and_color(
                    rows_data, rows_colors
                )
                if header_rows:
                    color_info = {
                        "rows_data": rows_data,
                        "rows_colors": rows_colors,
                        "header_rows": header_rows,
                        "max_cols": max_cols,
                    }
                    return header_rows, color_info
            except Exception as e:
                logger.warning(f"LLM检测失败: {e}")

        df_raw = pd.DataFrame(rows_data)
        header_rows = self._detect_header_rows_rule_based(df_raw)
        color_info = {
            "rows_data": rows_data,
            "rows_colors": rows_colors,
            "header_rows": header_rows,
            "max_cols": max_cols,
        }
        return header_rows, color_info

    def _detect_header_rows_with_llm_and_color(
        self, rows_data: List[List], rows_colors: List[List]
    ) -> List[int]:
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

        from dbgpt.core import (
            ModelMessage,
            ModelMessageRoleType,
            ModelRequest,
            ModelRequestContext,
        )

        # 构建表格文本表示（包含颜色信息）
        table_text = "行号\t列1\t列2\t列3\t...\t颜色信息\n"
        for idx, (row_data, row_colors) in enumerate(
            zip(rows_data[:20], rows_colors[:20])
        ):
            # 只显示前10列数据
            row_values = [str(val) if val else "" for val in row_data[:10]]
            # 统计颜色分布
            color_counts = {}
            for color in row_colors:
                if color:
                    color_counts[color] = color_counts.get(color, 0) + 1
            color_info = (
                ", ".join(
                    [f"{color[:8]}({count}列)" for color, count in color_counts.items()]
                )
                if color_counts
                else "无背景色"
            )
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
            "messages": [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            "temperature": 0.1,
            "max_new_tokens": 1000,
            "context": ModelRequestContext(stream=False),
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
            start_idx = full_text.find("{")
            end_idx = full_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = full_text[start_idx:end_idx]
                result = json.loads(json_str)

                header_rows = result.get("header_rows", [])
                reason = result.get("reason", "")

                logger.info(f"LLM判断理由: {reason}")

                # 验证结果
                if isinstance(header_rows, list) and all(
                    isinstance(x, int) for x in header_rows
                ):
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
        rows_data = color_info["rows_data"]
        rows_colors = color_info["rows_colors"]
        header_rows = color_info["header_rows"]
        max_cols = color_info["max_cols"]

        # 提取表头行的数据和颜色
        header_data = [rows_data[i] for i in header_rows]
        header_colors = [rows_colors[i] for i in header_rows]

        # 对每一行表头，按颜色和位置连续性分组
        filled_headers = []
        for row_idx, (row_data, row_colors) in enumerate(
            zip(header_data, header_colors)
        ):
            color_position_groups = []
            current_group = None
            current_color = None

            for col_idx, (value, color) in enumerate(zip(row_data, row_colors)):
                if color:
                    if (
                        color == current_color
                        and current_group
                        and col_idx == current_group[-1][0] + 1
                    ):
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
                non_empty_values = [
                    (idx, val) for idx, val in cells if val and str(val).strip()
                ]

                if len(non_empty_values) == 1:
                    fill_value = non_empty_values[0][1]
                    for col_idx, _ in cells:
                        filled_row[col_idx] = fill_value
                elif len(non_empty_values) > 1:
                    non_empty_indices = sorted([idx for idx, _ in non_empty_values])

                    for i, (val_idx, val) in enumerate(
                        sorted(non_empty_values, key=lambda x: x[0])
                    ):
                        start_idx = val_idx
                        if i < len(non_empty_indices) - 1:
                            end_idx = non_empty_indices[i + 1]
                        else:
                            end_idx = max([idx for idx, _ in cells]) + 1

                        cells_to_fill = [
                            (idx, v) for idx, v in cells if start_idx <= idx < end_idx
                        ]
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
                    val_str = (
                        val_str.replace("\n", "")
                        .replace("\r", "")
                        .replace("\t", "")
                        .replace(" ", "")
                        .replace("\u00a0", "")
                    )
                    # 避免重复值
                    if not col_values or val_str != col_values[-1]:
                        col_values.append(val_str)

            # 用"-"连接多级表头
            if col_values:
                combined = "-".join(col_values)
            else:
                combined = f"Column_{col_idx}"

            combined_headers.append(combined)

        return combined_headers

    def _process_multi_level_header(
        self, df_raw: pd.DataFrame, excel_file_path: str, sheet_name: str = None
    ) -> pd.DataFrame:
        """处理多级表头

        Args:
            df_raw: 原始DataFrame
            excel_file_path: Excel文件路径
            sheet_name: sheet名称
        """

        header_rows, color_info = self._detect_header_rows_with_color(
            excel_file_path, sheet_name
        )

        if not header_rows:
            header_rows = [0]

        combined_headers = self._merge_headers_by_color(color_info)

        cleaned_headers = []
        for header in combined_headers:
            cleaned = str(header)
            parts = cleaned.split("-")
            valid_parts = [
                p
                for p in parts
                if "=" not in p
                and "@@" not in p
                and "@" not in p
                and p.strip()
                and p.strip() not in ["-", "_", ""]
            ]

            if valid_parts:
                cleaned = "-".join(valid_parts)
            else:
                cleaned = f"Column_{len(cleaned_headers)}"

            cleaned = (
                cleaned.replace("\n", "")
                .replace("\r", "")
                .replace("\t", "")
                .replace(" ", "")
                .replace("\u00a0", "")
            )
            cleaned = cleaned.replace("--", "-").replace("__", "_").strip("-_")

            if not cleaned or cleaned in ["-", "_"]:
                cleaned = f"Column_{len(cleaned_headers)}"

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
            final_headers = final_headers[: len(data_df.columns)]
        elif len(final_headers) < len(data_df.columns):
            for i in range(len(final_headers), len(data_df.columns)):
                final_headers.append(f"Column_{i}")

        data_df.columns = final_headers
        data_df = data_df.dropna(how="all").reset_index(drop=True)

        return data_df

    def _clean_excel_formula(self, text: str) -> str:
        """清理Excel公式和特殊字符"""
        if not text:
            return text

        text_str = str(text)

        if text_str.startswith("="):
            import re

            quoted_texts = re.findall(r'["\']([^"\']+)["\']', text_str)
            if quoted_texts:
                text_str = "".join(quoted_texts)
            else:
                cleaned = re.sub(r"[=&()]", "", text_str)
                cleaned = re.sub(
                    r"CHAR\s*\(\s*\d+\s*\)", "", cleaned, flags=re.IGNORECASE
                )
                cleaned = re.sub(
                    r"CONCATENATE\s*\([^)]*\)", "", cleaned, flags=re.IGNORECASE
                )
                cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_]", "", cleaned)
                if not cleaned:
                    return ""
                text_str = cleaned

        if "@@" in text_str or text_str.startswith("@"):
            import re

            text_str = re.sub(r"@@[^\u4e00-\u9fff-]*", "", text_str)
            text_str = text_str.replace("@", "")
            if not text_str.strip():
                return ""

        text_str = (
            text_str.replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
            .replace(" ", "")
        )
        return text_str

    def _remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除完全为空的列"""
        df_cleaned = df.dropna(axis=1, how="all")

        empty_cols = [
            col
            for col in df_cleaned.columns
            if df_cleaned[col]
            .apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == ""))
            .all()
        ]

        if empty_cols:
            df_cleaned = df_cleaned.drop(columns=empty_cols)

        return df_cleaned

    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除列名和数据值都完全重复的列"""
        columns_to_remove = []

        for i, col1 in enumerate(df.columns):
            if col1 in columns_to_remove:
                continue

            for col2 in df.columns[i + 1 :]:
                if col2 in columns_to_remove or col1 != col2:
                    continue

                try:
                    if df[col1].equals(df[col2]):
                        columns_to_remove.append(col2)
                except Exception:
                    try:
                        if (
                            df[col1]
                            .fillna("__NULL__")
                            .equals(df[col2].fillna("__NULL__"))
                        ):
                            columns_to_remove.append(col2)
                    except Exception:
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
                    lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
                )
            elif df_formatted[col].dtype == "object":
                non_null_values = df_formatted[col].dropna()
                if len(non_null_values) > 0:
                    sample_val = non_null_values.iloc[0]
                    if isinstance(sample_val, (pd.Timestamp, datetime)):
                        df_formatted[col] = df_formatted[col].apply(
                            lambda x: x.strftime("%Y-%m-%d")
                            if pd.notna(x) and isinstance(x, (pd.Timestamp, datetime))
                            else x
                        )

        return df_formatted

    def _convert_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据字段的实际值进行智能类型转换

        转换策略：
        1. 尝试转换为日期类型
        2. 尝试转换为数值类型（整数或浮点数）
        3. 如果都失败，保持字符串类型

        Args:
            df: 原始DataFrame

        Returns:
            转换后的DataFrame
        """
        df_converted = df.copy()

        for col in df_converted.columns:
            # 跳过已经是数值类型的列
            if df_converted[col].dtype in ["int64", "float64", "int32", "float32"]:
                continue

            # 跳过已经是日期类型的列
            if pd.api.types.is_datetime64_any_dtype(df_converted[col]):
                continue

            # 只处理object类型的列
            if df_converted[col].dtype == "object":
                non_null_data = df_converted[col].dropna()

                if len(non_null_data) == 0:
                    continue

                # 策略1: 尝试转换为日期类型
                # 检查是否有足够的数据看起来像日期（至少30%的非空值能解析为日期）
                date_success_count = 0
                try:
                    # 尝试解析前100个非空值
                    sample_size = min(100, len(non_null_data))
                    sample_data = non_null_data.head(sample_size)

                    for val in sample_data:
                        if pd.notna(val):
                            try:
                                pd.to_datetime(str(val), errors="raise")
                                date_success_count += 1
                            except Exception:
                                pass

                    # 如果超过30%的值能解析为日期，则转换整个列为日期
                    if date_success_count > sample_size * 0.3:
                        try:
                            df_converted[col] = pd.to_datetime(
                                df_converted[col], errors="coerce"
                            )
                            # 转换为字符串格式 YYYY-MM-DD
                            df_converted[col] = df_converted[col].apply(
                                lambda x: x.strftime("%Y-%m-%d")
                                if pd.notna(x)
                                else None
                            )
                            logger.debug(f"列 '{col}' 转换为日期类型")
                            continue
                        except Exception:
                            pass
                except Exception:
                    pass

                # 策略2: 尝试转换为数值类型
                # 检查是否有足够的数据看起来像数字（至少50%的非空值能解析为数字）
                numeric_success_count = 0
                has_decimal = False

                try:
                    sample_size = min(100, len(non_null_data))
                    sample_data = non_null_data.head(sample_size)

                    for val in sample_data:
                        if pd.notna(val):
                            val_str = str(val).strip()
                            # 移除常见的数字格式字符（千位分隔符、货币符号等）
                            val_str = (
                                val_str.replace(",", "")
                                .replace("￥", "")
                                .replace("$", "")
                                .replace("€", "")
                                .replace(" ", "")
                            )

                            # 检查是否为数字
                            try:
                                float_val = float(val_str)
                                numeric_success_count += 1
                                # 检查是否有小数部分
                                if "." in str(val) and float_val != int(float_val):
                                    has_decimal = True
                            except Exception:
                                pass

                    # 如果超过50%的值能解析为数字，则转换整个列为数值类型
                    if numeric_success_count > sample_size * 0.5:
                        try:
                            # 先转换为字符串，移除格式字符，再转换为数值
                            df_converted[col] = df_converted[col].astype(str)
                            df_converted[col] = (
                                df_converted[col]
                                .str.replace(",", "")
                                .str.replace("￥", "")
                                .str.replace("$", "")
                                .str.replace("€", "")
                                .str.strip()
                            )
                            df_converted[col] = pd.to_numeric(
                                df_converted[col], errors="coerce"
                            )

                            # 根据是否有小数决定使用整数还是浮点数
                            if not has_decimal and df_converted[col].notna().any():
                                # 检查所有非空值是否都是整数
                                all_integers = True
                                for val in df_converted[col].dropna():
                                    if pd.notna(val) and val != int(val):
                                        all_integers = False
                                        break

                                if all_integers:
                                    df_converted[col] = df_converted[col].astype(
                                        "Int64"
                                    )  # 可空整数类型
                                    logger.debug(f"列 '{col}' 转换为整数类型")
                                else:
                                    df_converted[col] = df_converted[col].astype(
                                        "float64"
                                    )
                                    logger.debug(f"列 '{col}' 转换为浮点数类型")
                            else:
                                df_converted[col] = df_converted[col].astype("float64")
                                logger.debug(f"列 '{col}' 转换为浮点数类型")

                            continue
                        except Exception as e:
                            logger.debug(f"列 '{col}' 转换为数值类型失败: {e}")
                            pass
                except Exception as e:
                    logger.debug(f"列 '{col}' 数值类型检测失败: {e}")
                    pass

                # 策略3: 保持为字符串类型（object）
                # 不做任何转换

        return df_converted

    def _format_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        格式化数值列为两位小数

        Args:
            df: 原始DataFrame

        Returns:
            格式化后的DataFrame（数值列保留两位小数）
        """
        df_formatted = df.copy()

        for col in df_formatted.columns:
            # 只处理数值类型的列
            if df_formatted[col].dtype in [
                "int64",
                "float64",
                "int32",
                "float32",
                "Int64",
            ]:
                try:
                    # 将数值列转换为浮点数，并保留两位小数
                    # 使用 apply 确保所有值都格式化为两位小数（包括整数）
                    df_formatted[col] = pd.to_numeric(
                        df_formatted[col], errors="coerce"
                    ).apply(lambda x: round(float(x), 2) if pd.notna(x) else x)
                    # 确保数据类型为 float64，这样 DuckDB 会正确存储为浮点数
                    df_formatted[col] = df_formatted[col].astype("float64")
                    logger.debug(f"列 '{col}' 格式化为两位小数 (float64)")
                except Exception as e:
                    logger.warning(f"格式化列 '{col}' 失败: {e}")
                    # 如果格式化失败，保持原样
                    pass

        return df_formatted

    def _detect_header_rows_rule_based(self, df_raw: pd.DataFrame) -> List[int]:
        """基于规则的表头行检测"""

        header_keywords = [
            "id",
            "ID",
            "编号",
            "序号",
            "行",
            "订单",
            "日期",
            "date",
            "Date",
            "时间",
            "time",
            "Time",
            "名称",
            "name",
            "Name",
            "客户",
            "产品",
            "金额",
            "价格",
            "price",
            "Price",
            "销售额",
            "利润",
            "数量",
            "quantity",
            "Quantity",
            "类别",
            "category",
            "Category",
            "类型",
            "type",
            "Type",
            "区域",
            "region",
            "Region",
            "城市",
            "city",
            "City",
            "信息",
            "info",
            "Info",
            "数据",
            "data",
            "Data",
        ]

        max_check_rows = min(20, len(df_raw))
        candidate_rows = []

        for i in range(max_check_rows):
            row = df_raw.iloc[i]
            non_null_count = row.notna().sum()
            row_text = " ".join([str(val) for val in row if pd.notna(val)])
            keyword_matches = sum(
                1 for keyword in header_keywords if keyword.lower() in row_text.lower()
            )
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

    def _merge_multiple_sheets(
        self,
        sheets_data: List[Tuple[str, pd.DataFrame]],
        source_column_name: str = "数据类型",
    ) -> pd.DataFrame:
        """
        合并多个sheet的数据，添加来源标识列

        Args:
            sheets_data: [(sheet_name, dataframe), ...] 列表
            source_column_name: 来源列的列名，默认为"数据类型"

        Returns:
            合并后的DataFrame
        """
        if not sheets_data:
            raise ValueError("sheets_data不能为空")

        if len(sheets_data) == 1:
            # 只有一个sheet，直接添加来源列
            sheet_name, df = sheets_data[0]
            df_copy = df.copy()
            df_copy[source_column_name] = sheet_name
            return df_copy

        # 多个sheet的情况
        merged_dfs = []

        # 收集所有列名（按出现顺序）
        all_columns = []
        seen_columns = set()
        for sheet_name, df in sheets_data:
            for col in df.columns:
                if col not in seen_columns:
                    all_columns.append(col)
                    seen_columns.add(col)

        logger.info(f"合并{len(sheets_data)}个sheet，共{len(all_columns)}个唯一列")

        # 对每个sheet进行列对齐
        for sheet_name, df in sheets_data:
            df_copy = df.copy()

            # 添加缺失的列（填充为None）
            for col in all_columns:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # 按统一的列顺序重新排列
            df_copy = df_copy[all_columns]

            # 添加来源列
            df_copy[source_column_name] = sheet_name

            merged_dfs.append(df_copy)
            logger.debug(f"Sheet '{sheet_name}': {len(df)}行 -> 对齐后{len(df_copy)}行")

        # 合并所有DataFrame
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        logger.info(f"合并完成：总行数 {len(merged_df)}")

        return merged_df

    def process_excel(
        self,
        excel_file_path: str,
        table_name: str = None,
        force_reimport: bool = False,
        original_filename: str = None,
        conv_uid: str = None,
        sheet_names: List[str] = None,
        merge_sheets: bool = False,
        source_column_name: str = "数据类型",
    ) -> Dict:
        """处理Excel文件，自动注册到数据源

        Args:
            excel_file_path: Excel文件路径
            table_name: 表名（可选）
            force_reimport: 是否强制重新导入
            original_filename: 原始文件名（可选）
            conv_uid: 会话ID（可选）
            sheet_names: 要处理的sheet名称列表，如果为None则处理所有sheet
            merge_sheets: 是否合并多个sheet（如果为True，将多个sheet合并为一张表）
            source_column_name: 合并时添加的来源列名，默认为"数据类型"
        """
        if original_filename is None:
            original_filename = Path(excel_file_path).name

        # 读取Excel获取sheet信息
        excel_file = pd.ExcelFile(excel_file_path)
        all_sheet_names = excel_file.sheet_names

        # 确定要处理的sheet
        if sheet_names is None:
            target_sheets = all_sheet_names
        else:
            # 验证指定的sheet是否存在
            target_sheets = []
            for name in sheet_names:
                if name in all_sheet_names:
                    target_sheets.append(name)
                else:
                    logger.warning(f"Sheet '{name}' 不存在，跳过")

            if not target_sheets:
                raise ValueError(f"指定的sheet都不存在。可用的sheet: {all_sheet_names}")

        # 使用文件级别的哈希（包含sheet信息）
        content_hash = self.cache_manager.calculate_file_hash(
            excel_file_path, target_sheets if merge_sheets else None
        )
        logger.debug(f"文件哈希: {content_hash[:16]}... (文件: {original_filename})")

        # 检查缓存（在读取Excel和处理表头之前）
        if not force_reimport:
            cached_info = self.cache_manager.get_cached_info(content_hash)
            if cached_info and os.path.exists(cached_info["db_path"]):
                # 缓存命中，直接返回（无需读取Excel和处理表头）
                cached_schema_json = cached_info.get("data_schema_json")

                # 为了返回top_10_rows，需要从数据库读取
                try:
                    import duckdb

                    conn = duckdb.connect(cached_info["db_path"])
                    # 获取列名
                    columns_result = conn.execute(
                        f"DESCRIBE {cached_info['table_name']}"
                    ).fetchall()
                    columns = [col[0] for col in columns_result]

                    # 获取前10行数据
                    rows = conn.execute(
                        f"SELECT * FROM {cached_info['table_name']} LIMIT 10"
                    ).fetchall()
                    conn.close()

                    # 转换为字典列表
                    top_10_rows = [dict(zip(columns, row)) for row in rows]
                    top_10_rows = self._convert_to_json_serializable(top_10_rows)
                except Exception as e:
                    logger.warning(f"从数据库读取top_10_rows失败: {e}，将返回空列表")
                    top_10_rows = []

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
                    "conv_uid": conv_uid,
                }

        # 没有缓存或强制重新导入，需要完整处理
        logger.info(f"📝 开始处理Excel文件（需要识别表头）: {original_filename}")

        # 处理多个sheet
        if merge_sheets and len(target_sheets) > 1:
            # 合并多个sheet的场景
            logger.info(f"🔄 合并 {len(target_sheets)} 个sheet...")
            sheets_data = []

            for sheet_name in target_sheets:
                logger.info(f"  处理 sheet: {sheet_name}")
                df_raw = pd.read_excel(
                    excel_file_path, sheet_name=sheet_name, header=None
                )
                df_processed = self._process_multi_level_header(
                    df_raw, excel_file_path, sheet_name
                )
                sheets_data.append((sheet_name, df_processed))

            # 合并所有sheet
            df = self._merge_multiple_sheets(sheets_data, source_column_name)
        else:
            # 只处理第一个sheet（原有逻辑）
            target_sheet = target_sheets[0]
            logger.info(f"📄 处理单个sheet: {target_sheet}")
            df_raw = pd.read_excel(
                excel_file_path, sheet_name=target_sheet, header=None
            )
            df = self._process_multi_level_header(df_raw, excel_file_path, target_sheet)

        if table_name is None:
            base_name = Path(original_filename).stem
            base_name = "".join(
                c if c.isalnum() or c == "_" else "_" for c in base_name
            )
            if base_name and base_name[0].isdigit():
                base_name = f"tbl_{base_name}"
            if not base_name or len(base_name) < 2:
                base_name = f"excel_table_{content_hash[:8]}"
            table_name = base_name

        db_name = f"excel_{content_hash[:8]}"
        db_filename = f"{db_name}.duckdb"
        db_path = str(self.db_storage_dir / db_filename)

        df = self._remove_empty_columns(df)
        df = self._remove_duplicate_columns(df)
        df = self._format_date_columns(df)
        df = self._convert_column_types(df)  # 智能类型转换
        df = self._format_numeric_columns(df)  # 格式化数值列为两位小数

        df.columns = [
            str(col)
            .replace(" ", "")
            .replace("\u00a0", "")
            .replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
            for col in df.columns
        ]

        # 清理后再次去重列名,防止DuckDB报duplicate column name错误
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

        # 直接使用DuckDB保存数据（跳过SQLite）
        import duckdb

        conn = None
        try:
            conn = duckdb.connect(db_path)
            # 将DataFrame注册为临时视图
            conn.register("temp_df", df)
            
            # 识别数值列，并在创建表时使用 ROUND 函数确保保留两位小数
            numeric_columns = []
            for col in df.columns:
                if df[col].dtype in ["int64", "float64", "int32", "float32", "Int64"]:
                    numeric_columns.append(col)
            
            # 构建 SELECT 语句，对数值列应用 ROUND 函数
            if numeric_columns:
                select_parts = []
                for col in df.columns:
                    # 使用双引号转义列名，防止特殊字符问题
                    col_quoted = f'"{col}"'
                    if col in numeric_columns:
                        # 对数值列使用 ROUND 函数保留两位小数
                        select_parts.append(f"ROUND(CAST({col_quoted} AS DOUBLE), 2) AS {col_quoted}")
                    else:
                        select_parts.append(col_quoted)
                select_sql = ", ".join(select_parts)
                table_name_quoted = f'"{table_name}"'
                conn.execute(f"CREATE TABLE {table_name_quoted} AS SELECT {select_sql} FROM temp_df")
            else:
                # 如果没有数值列，直接创建表
                table_name_quoted = f'"{table_name}"'
                conn.execute(f"CREATE TABLE {table_name_quoted} AS SELECT * FROM temp_df")
            
            # DuckDB 会自动提交，但显式关闭连接确保数据写入磁盘
            conn.close()
            conn = None
            logger.info(
                f"✅ 数据已保存到DuckDB: {db_path} (表: {table_name}, 行数: {len(df)})"
            )
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            logger.error(f"数据写入DuckDB失败: {e}")
            print(f"❌ DEBUG: 保存失败: {e}")
            raise Exception(f"Excel数据转换为数据库失败: {e}")

        # 获取列信息
        conn = duckdb.connect(db_path)
        try:
            columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns_info = [
                {"name": col[0], "type": col[1], "dtype": str(df[col[0]].dtype)}
                for col in columns_result
            ]
        finally:
            conn.close()

        schema_understanding_json = self._generate_schema_understanding_with_llm(
            df, table_name
        )
        summary_prompt = self._format_schema_as_prompt(
            schema_understanding_json, df, table_name
        )

        self.cache_manager.save_cache_info(
            content_hash=content_hash,
            original_filename=original_filename,
            table_name=table_name,
            db_name=db_name,
            db_path=db_path,
            df=df,
            summary_prompt=summary_prompt,
            data_schema_json=schema_understanding_json,
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
            "conv_uid": conv_uid,
        }

    def _generate_schema_understanding_with_llm(
        self, df: pd.DataFrame, table_name: str
    ) -> str:
        """使用LLM生成Schema理解JSON"""
        er_info = self._prepare_er_info(df, table_name)
        numeric_stats = self._prepare_numeric_stats(df)
        categorical_distribution = self._prepare_categorical_distribution(df)

        prompt = self._build_schema_understanding_prompt(
            table_name=table_name,
            er_info=er_info,
            numeric_stats=numeric_stats,
            categorical_distribution=categorical_distribution,
            sample_data=df.head(3).to_dict("records"),
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
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

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
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

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
                    dist_lines.append(
                        f"    - '{val}': {count}条 ({count / len(df) * 100:.1f}%)"
                    )
            else:
                # 只显示前10个最常见的值
                value_counts = df[col].value_counts()
                dist_lines.append("    前10个最常见值:")
                for val, count in value_counts.head(10).items():
                    dist_lines.append(
                        f"    - '{val}': {count}条 ({count / len(df) * 100:.1f}%)"
                    )

        return "\n".join(dist_lines)

    def _build_schema_understanding_prompt(
        self,
        table_name: str,
        er_info: str,
        numeric_stats: str,
        categorical_distribution: str,
        sample_data: list,
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
            elif hasattr(obj, "isoformat"):  # datetime, date, time, Timestamp
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_sample_data = convert_to_serializable(sample_data)
        sample_data_str = json.dumps(
            serializable_sample_data, ensure_ascii=False, indent=2
        )

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
            if hasattr(chunk, "text"):
                return chunk.text
            elif hasattr(chunk, "content"):
                if hasattr(chunk.content, "get_text"):
                    try:
                        return chunk.content.get_text()
                    except Exception:
                        pass
                elif isinstance(chunk.content, str):
                    return chunk.content
        except Exception as e:
            logger.debug(f"提取chunk文本失败: {e}")
        return ""

    def _enrich_schema_json(
        self, simplified_json: str, df: pd.DataFrame, table_name: str
    ) -> str:
        """通过代码补充技术性字段，生成完整的Schema JSON"""
        try:
            schema = json.loads(simplified_json)
        except json.JSONDecodeError as e:
            logger.error(f"解析简化JSON失败: {e}")
            raise

        # 构建字段名映射
        llm_map = {
            col.get("column_name"): col
            for col in schema.get("columns", [])
            if col.get("column_name")
        }

        # 构建完整的columns列表
        enriched_columns = []
        for col_name in df.columns:
            col_data = df[col_name]
            dtype = str(col_data.dtype)
            llm_info = llm_map.get(col_name, {})

            col_info = {
                "column_name": col_name,
                "data_type": dtype,
                "semantic_type": llm_info.get("semantic_type", "未知"),
                "description": llm_info.get("description", f"{col_name}字段"),
                "is_key_field": self._is_potential_key_field(col_name, col_data),
            }

            # 判断字段类型（优先根据语义类型判断，避免数据类型误判）
            semantic_type = col_info["semantic_type"]
            is_numeric_by_dtype = dtype in ["int64", "float64", "int32", "float32"]
            is_numeric_by_semantic = "指标" in semantic_type or "数值" in semantic_type

            # 数值字段：优先根据语义类型判断，如果语义类型是数值指标，即使data_type是object也按数值处理
            if is_numeric_by_dtype or is_numeric_by_semantic:
                # 尝试转换为数值类型
                if dtype in ["object", "category"]:
                    try:
                        numeric_data = pd.to_numeric(col_data, errors="coerce").dropna()
                    except Exception:
                        numeric_data = col_data.dropna()
                else:
                    numeric_data = col_data.dropna()

                if len(numeric_data) > 0:
                    try:
                        min_val = numeric_data.min()
                        max_val = numeric_data.max()
                        mean_val = numeric_data.mean()
                        median_val = numeric_data.median()
                        col_info["statistics_summary"] = (
                            f"范围: [{min_val:.2f}, {max_val:.2f}], "
                            f"均值: {mean_val:.2f}, 中位数: {median_val:.2f}"
                        )
                    except Exception:
                        # 如果转换失败，不添加统计信息
                        pass

            # 分类字段：只列出出现次数最高的5个值，并合并统计信息
            # 排除数值指标字段（即使data_type是object）
            elif (
                dtype in ["object", "category"]
                and "时间" not in semantic_type
                and "日期" not in semantic_type
                and "指标" not in semantic_type
                and "数值" not in semantic_type
            ):
                value_counts = col_data.value_counts()
                total_unique = len(col_data.dropna().unique())

                # 只取出现次数最高的5个值
                top_5_values = value_counts.head(5)
                col_info["unique_values_top5"] = [
                    str(v) for v in top_5_values.index.tolist()
                ]

                # 合并统计信息到 value_distribution（包含出现次数）
                top_5_list = [
                    f"{str(v)}({count}次)" for v, count in top_5_values.items()
                ]
                col_info["value_distribution"] = (
                    f"共{total_unique}个唯一值，出现次数前5: {', '.join(top_5_list)}"
                )

            enriched_columns.append(col_info)

        return json.dumps(
            {
                "table_name": table_name,
                "table_description": schema.get("table_description", ""),
                "columns": enriched_columns,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _call_llm_for_schema(self, prompt: str) -> str:
        """调用LLM生成Schema JSON"""
        try:
            import asyncio
            import logging

            from dbgpt.core import ModelMessage, ModelMessageRoleType, ModelRequest

            worker_logger = logging.getLogger(
                "dbgpt.model.cluster.worker.default_worker"
            )
            original_level = worker_logger.level

            try:
                worker_logger.setLevel(logging.ERROR)

                request_params = {
                    "messages": [
                        ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)
                    ],
                    "temperature": 0.1,
                    "max_new_tokens": 20480,
                }

                if self.model_name:
                    request_params["model"] = self.model_name

                request = ModelRequest(**request_params)

                if hasattr(self.llm_client, "generate_stream"):
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
                        raise Exception(
                            f"Unexpected response type: {type(stream_response)}"
                        )

                    class FakeResponse:
                        def __init__(self, text):
                            self.text = text

                    response = FakeResponse(full_text)

                else:
                    raise Exception("LLM客户端没有generate_stream方法")

            finally:
                worker_logger.setLevel(original_level)

            if response and hasattr(response, "text") and response.text:
                text = response.text.strip()
                json_str = None

                if "```json" in text.lower():
                    start_idx = text.lower().find("```json")
                    if start_idx >= 0:
                        content_start = text.find("\n", start_idx) + 1
                        if content_start > 0:
                            end_idx = text.find("```", content_start)
                            if end_idx > content_start:
                                json_str = text[content_start:end_idx].strip()

                if not json_str and "```" in text:
                    start_idx = text.find("```")
                    content_start = text.find("\n", start_idx) + 1
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

    def _format_schema_as_prompt(
        self, schema_json: str, df: pd.DataFrame, table_name: str
    ) -> str:
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
        lines.append("=== 数据表Schema理解 ===")
        lines.append(f"表名: {schema.get('table_name', table_name)}")
        lines.append(f"表描述: {schema.get('table_description', '')}")
        lines.append("")

        lines.append("=== 字段详细信息 ===")
        for col in schema.get("columns", []):
            lines.append(f"\n字段: {col.get('column_name')}")
            lines.append(f"  类型: {col.get('data_type')}")
            lines.append(f"  语义: {col.get('semantic_type')}")
            lines.append(f"  描述: {col.get('description')}")

            if "unique_values_top5" in col:
                unique_vals = col["unique_values_top5"]
                lines.append(
                    f"  出现次数前5的值: {', '.join([str(v) for v in unique_vals])}"
                )

            if "statistics_summary" in col:
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
        if any(keyword in col_name_lower for keyword in ["id", "编号", "key", "code"]):
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
        from datetime import date, datetime

        if isinstance(obj, (datetime, date, pd.Timestamp)):
            # 如果是日期类型，检查时间部分是否为00:00:00，如果是则只返回日期部分
            if isinstance(obj, pd.Timestamp):
                if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                    return obj.strftime("%Y-%m-%d")
                else:
                    return obj.isoformat()
            elif isinstance(obj, datetime):
                if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                    return obj.strftime("%Y-%m-%d")
                else:
                    return obj.isoformat()
            elif isinstance(obj, date):
                return obj.strftime("%Y-%m-%d")
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
            logger.info(
                f"SQLite数据库已创建: {db_name}, 路径: {db_path}, 表名: {table_name}"
            )
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
