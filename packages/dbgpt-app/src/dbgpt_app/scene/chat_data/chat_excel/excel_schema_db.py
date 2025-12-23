"""Excel Schema DB Model for storing Excel metadata information."""

import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from dbgpt.storage.metadata import BaseDao, Model

logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj):
    """
    递归转换对象为可JSON序列化的格式

    Args:
        obj: 要转换的对象

    Returns:
        可JSON序列化的对象
    """
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


class ExcelSchemaEntity(Model):
    """Excel schema entity."""

    __tablename__ = "excel_schema"
    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="autoincrement id"
    )

    conv_uid = Column(String(255), nullable=False, comment="conversation uid")
    file_name = Column(String(255), nullable=False, comment="excel file name")
    file_path = Column(String(512), nullable=True, comment="excel file path")
    table_name = Column(String(255), nullable=False, comment="table name in database")
    db_path = Column(String(512), nullable=True, comment="database path")

    # 基本信息
    row_count = Column(Integer, nullable=True, comment="total row count")
    column_count = Column(Integer, nullable=True, comment="total column count")
    top_10_rows = Column(Text, nullable=True, comment="top 10 rows data, JSON format")

    # 数据描述
    data_description = Column(Text, nullable=True, comment="data description from LLM")
    data_schema_json = Column(Text, nullable=True, comment="data schema JSON")

    # 推荐问题
    suggested_questions = Column(
        Text, nullable=True, comment="suggested questions, JSON format"
    )

    # 元数据
    user_id = Column(String(128), index=True, nullable=True, comment="User id")
    user_name = Column(String(128), index=True, nullable=True, comment="User name")
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")

    __table_args__ = (
        UniqueConstraint("conv_uid", name="uk_conv_uid"),
        Index("idx_file_name", "file_name"),
        Index("idx_user_id", "user_id"),
    )


class ExcelSchemaDao(BaseDao):
    """Excel schema dao."""

    def get_by_conv_uid(self, conv_uid: str) -> Optional[ExcelSchemaEntity]:
        """Get excel schema by conversation uid."""
        session = self.get_raw_session()
        try:
            result = (
                session.query(ExcelSchemaEntity)
                .filter(ExcelSchemaEntity.conv_uid == conv_uid)
                .first()
            )
            return result
        finally:
            session.close()

    def get_by_file_name(
        self, file_name: str, user_id: Optional[str] = None
    ) -> List[ExcelSchemaEntity]:
        """Get excel schema by file name."""
        session = self.get_raw_session()
        try:
            query = session.query(ExcelSchemaEntity).filter(
                ExcelSchemaEntity.file_name == file_name
            )
            if user_id:
                query = query.filter(ExcelSchemaEntity.user_id == user_id)
            result = query.all()
            return result
        finally:
            session.close()

    def save_or_update(
        self,
        conv_uid: str,
        file_name: str,
        table_name: str,
        row_count: int,
        column_count: int,
        top_10_rows: List[List[Any]],
        data_description: Optional[str] = None,
        data_schema_json: Optional[str] = None,
        suggested_questions: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        db_path: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        sys_code: Optional[str] = None,
    ) -> ExcelSchemaEntity:
        """Save or update excel schema."""
        session = self.get_raw_session()
        try:
            # 检查是否已存在
            existing = (
                session.query(ExcelSchemaEntity)
                .filter(ExcelSchemaEntity.conv_uid == conv_uid)
                .first()
            )

            # 转换 top_10_rows 为可JSON序列化的格式
            serializable_top_10_rows = convert_to_json_serializable(top_10_rows)
            top_10_rows_json = json.dumps(serializable_top_10_rows, ensure_ascii=False)

            suggested_questions_json = (
                json.dumps(suggested_questions, ensure_ascii=False)
                if suggested_questions
                else None
            )

            if existing:
                # 更新
                existing.file_name = file_name
                existing.table_name = table_name
                existing.row_count = row_count
                existing.column_count = column_count
                existing.top_10_rows = top_10_rows_json
                existing.data_description = data_description
                existing.data_schema_json = data_schema_json
                existing.suggested_questions = suggested_questions_json
                existing.file_path = file_path
                existing.db_path = db_path
                existing.gmt_modified = datetime.now()
                if user_id:
                    existing.user_id = user_id
                if user_name:
                    existing.user_name = user_name
                if sys_code:
                    existing.sys_code = sys_code
                session.commit()
                return existing
            else:
                # 新建
                entity = ExcelSchemaEntity(
                    conv_uid=conv_uid,
                    file_name=file_name,
                    table_name=table_name,
                    row_count=row_count,
                    column_count=column_count,
                    top_10_rows=top_10_rows_json,
                    data_description=data_description,
                    data_schema_json=data_schema_json,
                    suggested_questions=suggested_questions_json,
                    file_path=file_path,
                    db_path=db_path,
                    user_id=user_id,
                    user_name=user_name,
                    sys_code=sys_code,
                )
                session.add(entity)
                session.commit()
                return entity
        except Exception as e:
            session.rollback()
            logger.error(f"保存 Excel schema 失败: {str(e)}")
            raise
        finally:
            session.close()

    def to_dict(self, entity: ExcelSchemaEntity) -> Dict[str, Any]:
        """Convert entity to dict."""
        result = {
            "id": entity.id,
            "conv_uid": entity.conv_uid,
            "file_name": entity.file_name,
            "file_path": entity.file_path,
            "table_name": entity.table_name,
            "db_path": entity.db_path,
            "row_count": entity.row_count,
            "column_count": entity.column_count,
            "data_description": entity.data_description,
            "data_schema_json": entity.data_schema_json,
            "user_id": entity.user_id,
            "user_name": entity.user_name,
            "sys_code": entity.sys_code,
            "gmt_created": (
                entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S")
                if entity.gmt_created
                else None
            ),
            "gmt_modified": (
                entity.gmt_modified.strftime("%Y-%m-%d %H:%M:%S")
                if entity.gmt_modified
                else None
            ),
        }

        # 解析 JSON 字段
        if entity.top_10_rows:
            try:
                result["top_10_rows"] = json.loads(entity.top_10_rows)
            except:
                result["top_10_rows"] = []
        else:
            result["top_10_rows"] = []

        if entity.suggested_questions:
            try:
                result["suggested_questions"] = json.loads(entity.suggested_questions)
            except:
                result["suggested_questions"] = []
        else:
            result["suggested_questions"] = []

        return result
