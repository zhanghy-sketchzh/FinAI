"""
Chat Excel API 路由
将 Excel 上传功能集成到 Chat Excel 模式中
"""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from dbgpt_serve.core import Result, blocking_func_to_async
from dbgpt_serve.datasource.api.endpoints import (
    check_api_key,
    global_system_app,
)
from dbgpt_serve.datasource.api.schemas import ExcelUploadResponse
from dbgpt_serve.datasource.service.excel_auto_register import ExcelAutoRegisterService

logger = logging.getLogger(__name__)


def _get_llm_client_and_model():
    """获取 LLM 客户端和模型名称"""
    try:
        from dbgpt.model.cluster import WorkerManagerFactory
        from dbgpt.model.cluster.client import DefaultLLMClient
        from dbgpt.component import ComponentType
        
        worker_manager = global_system_app.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        llm_client = DefaultLLMClient(worker_manager, auto_convert_message=True)
        
        # 获取可用模型
        models = worker_manager.sync_supported_models()
        if models and len(models) > 0:
            model_name = models[0].model
            logger.info(f"获取到可用模型: {model_name}")
            return llm_client, model_name
    except Exception as e:
        logger.warning(f"获取 LLM 客户端失败: {e}")
    
    return None, None

chat_excel_router = APIRouter()


@chat_excel_router.post(
    "/chat-excel/upload",
    dependencies=[Depends(check_api_key)],
    response_model=Result[ExcelUploadResponse],
)
async def upload_excel_for_chat(
    file: UploadFile = File(..., description="Excel file to upload"),
    table_name: Optional[str] = Query(None, description="Target table name"),
    auto_start_chat: bool = Query(True, description="Auto redirect to chat"),
) -> Result[ExcelUploadResponse]:
    """
    上传 Excel 文件并自动注册到数据源，用于 Chat Excel

    Args:
        file: Excel 文件
        table_name: 目标表名（可选）
        auto_start_chat: 是否自动开始对话

    Returns:
        包含数据库信息的上传结果，前端可以直接用于 Chat

    Raises:
        HTTPException: 上传或处理失败时抛出
    """
    # 检查文件类型
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持 Excel 文件 (.xlsx, .xls)")

    # 保存临时文件
    temp_file_path = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 获取 LLM 客户端和模型
        llm_client, model_name = _get_llm_client_and_model()
        
        # 处理 Excel
        excel_service = ExcelAutoRegisterService(
            llm_client=llm_client, model_name=model_name
        )
        result = await blocking_func_to_async(
            global_system_app,
            excel_service.process_excel,
            temp_file_path,
            table_name,
            False,  # 默认使用缓存
        )

        # 如果是缓存数据，添加提示
        if result["status"] == "cached":
            access_count = result.get("access_count", 0)
            result["message"] = (
                f"检测到相同文件，使用缓存数据（已访问 {access_count} 次）"
            )

        return Result.succ(ExcelUploadResponse(**result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理 Excel 文件失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@chat_excel_router.get(
    "/chat-excel/info/{content_hash}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[dict],
)
async def get_chat_excel_info(content_hash: str) -> Result[dict]:
    """
    获取已上传 Excel 的信息

    Args:
        content_hash: Excel 内容哈希

    Returns:
        Excel 信息和数据库连接信息
    """
    try:
        # get_excel_info 不需要 LLM，可以直接创建
        excel_service = ExcelAutoRegisterService()
        info = await blocking_func_to_async(
            global_system_app, excel_service.get_excel_info, content_hash
        )

        if info is None:
            raise HTTPException(status_code=404, detail="未找到对应的 Excel 信息")

        return Result.succ(info)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Excel 信息失败: {str(e)}")


@chat_excel_router.post(
    "/chat-excel/query",
    dependencies=[Depends(check_api_key)],
    response_model=Result[dict],
)
async def query_excel_data(
    content_hash: str = Query(..., description="Excel content hash"),
    sql: str = Query(..., description="SQL query to execute"),
    limit: int = Query(100, description="Result limit"),
) -> Result[dict]:
    """
    对上传的 Excel 数据执行 SQL 查询

    Args:
        content_hash: Excel 内容哈希
        sql: SQL 查询语句
        limit: 结果限制

    Returns:
        查询结果
    """
    import sqlite3

    import pandas as pd

    try:
        # 获取 Excel 信息
        excel_service = ExcelAutoRegisterService()
        info = await blocking_func_to_async(
            global_system_app, excel_service.get_excel_info, content_hash
        )

        if info is None:
            raise HTTPException(status_code=404, detail="未找到对应的 Excel 信息")

        # 执行查询
        db_path = info["db_path"]
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="数据库文件不存在")

        # 添加 LIMIT 子句
        if "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        # 执行查询
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()

        # 转换为 JSON
        result = {
            "columns": df.columns.tolist(),
            "data": df.to_dict("records"),
            "row_count": len(df),
            "sql": sql,
        }

        return Result.succ(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
