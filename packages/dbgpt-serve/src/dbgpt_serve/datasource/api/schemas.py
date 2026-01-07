from typing import Any, Dict, Optional

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

from ..config import SERVE_APP_NAME_HUMP


class DatasourceServeRequest(BaseModel):
    """name: knowledge space name"""

    """vector_type: vector type"""
    id: Optional[int] = Field(None, description="The datasource id")
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")
    db_name: str = Field(..., description="Database name.")
    db_path: Optional[str] = Field("", description="File path for file-based database.")
    db_host: Optional[str] = Field("", description="Database host.")
    db_port: Optional[int] = Field(0, description="Database port.")
    db_user: Optional[str] = Field("", description="Database user.")
    db_pwd: Optional[str] = Field("", description="Database password.")
    comment: Optional[str] = Field("", description="Comment for the database.")
    ext_config: Optional[Dict[str, Any]] = Field(
        None, description="Extra configuration for the datasource."
    )


class DatasourceServeResponse(BaseModel):
    """Flow response model"""

    model_config = ConfigDict(title=f"ServeResponse for {SERVE_APP_NAME_HUMP}")

    """name: knowledge space name"""

    """vector_type: vector type"""
    id: int = Field(None, description="The datasource id")
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")
    db_name: str = Field(..., description="Database name.")
    db_path: Optional[str] = Field("", description="File path for file-based database.")
    db_host: Optional[str] = Field("", description="Database host.")
    db_port: Optional[int] = Field(0, description="Database port.")
    db_user: Optional[str] = Field("", description="Database user.")
    db_pwd: Optional[str] = Field("", description="Database password.")
    comment: Optional[str] = Field("", description="Comment for the database.")
    ext_config: Optional[Dict[str, Any]] = Field(
        None, description="Extra configuration for the datasource."
    )

    gmt_created: Optional[str] = Field(
        None,
        description="The datasource created time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )
    gmt_modified: Optional[str] = Field(
        None,
        description="The datasource modified time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )


class DatasourceCreateRequest(BaseModel):
    """Request model for datasource connection

    Attributes:
        type (str): The type of datasource (e.g., "mysql", "tugraph")
        params (Dict[str, Any]): Dynamic parameters for the datasource connection.
            The keys should match the param_name defined in the datasource type
            configuration.
    """

    type: str = Field(
        ..., description="The type of datasource (e.g., 'mysql', 'tugraph')"
    )
    params: Dict[str, Any] = Field(
        ..., description="Dynamic parameters for the datasource connection."
    )
    description: Optional[str] = Field(
        None, description="Optional description of the datasource."
    )
    id: Optional[int] = Field(None, description="The datasource id")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "tugraph",
                "params": {
                    "host": "localhost",
                    "user": "test_user",
                    "password": "test_password",
                    "port": 7687,
                    "database": "default",
                },
            }
        }


class DatasourceQueryResponse(DatasourceCreateRequest):
    """Response model for datasource query"""

    gmt_created: Optional[str] = Field(
        None,
        description="The datasource created time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )
    gmt_modified: Optional[str] = Field(
        None,
        description="The datasource modified time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )


class ExcelUploadRequest(BaseModel):
    """Request model for Excel upload"""

    table_name: Optional[str] = Field(
        None, description="Target table name, auto-generated if not provided"
    )
    force_reimport: bool = Field(False, description="Force reimport even if cached")


class ExcelUploadResponse(BaseModel):
    """Response model for Excel upload"""

    status: str = Field(..., description="Status: 'cached' or 'imported'")
    message: str = Field(..., description="Status message")
    content_hash: str = Field(..., description="Content hash of the Excel file")
    db_name: str = Field(..., description="Database name")
    db_path: str = Field(..., description="Database file path")
    table_name: str = Field(..., description="Table name")
    row_count: int = Field(..., description="Number of rows")
    column_count: int = Field(..., description="Number of columns")
    columns_info: list = Field(..., description="Column information")
    summary_prompt: Optional[str] = Field(None, description="Data understanding prompt")
    preview_data: Optional[Dict[str, Any]] = Field(
        None, description="Preview data with columns and rows for table display"
    )
    access_count: Optional[int] = Field(None, description="Access count (for cached)")
    last_accessed: Optional[str] = Field(
        None, description="Last access time (for cached)"
    )


class ExcelInfoResponse(BaseModel):
    """Response model for Excel info query"""

    content_hash: str = Field(..., description="Content hash")
    original_filename: str = Field(..., description="Original filename")
    table_name: str = Field(..., description="Table name")
    db_name: str = Field(..., description="Database name")
    db_path: str = Field(..., description="Database path")
    row_count: int = Field(..., description="Row count")
    column_count: int = Field(..., description="Column count")
    columns_info: list = Field(..., description="Columns information")
    summary_prompt: Optional[str] = Field(None, description="Summary prompt")
    preview_data: Optional[Dict[str, Any]] = Field(
        None, description="Preview data with columns and rows for table display"
    )
    created_at: str = Field(..., description="Created time")
    last_accessed: str = Field(..., description="Last accessed time")
    access_count: int = Field(..., description="Access count")
