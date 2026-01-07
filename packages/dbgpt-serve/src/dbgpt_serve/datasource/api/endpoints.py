import logging
import os
import tempfile
from functools import cache
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from dbgpt.component import SystemApp
from dbgpt_serve.core import ResourceTypes, Result, blocking_func_to_async
from dbgpt_serve.datasource.api.schemas import (
    DatasourceCreateRequest,
    DatasourceQueryResponse,
    DatasourceServeRequest,
    ExcelInfoResponse,
    ExcelUploadResponse,
)
from dbgpt_serve.datasource.config import SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from dbgpt_serve.datasource.service.excel_auto_register import ExcelAutoRegisterService
from dbgpt_serve.datasource.service.service import Service

logger = logging.getLogger(__name__)

router = APIRouter()

# Add your API endpoints here

global_system_app: Optional[SystemApp] = None


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


def get_service() -> Service:
    """Get the service instance"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


get_bearer_token = HTTPBearer(auto_error=False)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """Parse the string api keys to a list

    Args:
        api_keys (str): The string api keys

    Returns:
        List[str]: The list of api keys
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    service: Service = Depends(get_service),
) -> Optional[str]:
    """Check the api key

    If the api key is not set, allow all.

    Your can pass the token in you request header like this:

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

    """
    if service.config.api_keys:
        api_keys = _parse_api_keys(service.config.api_keys)
        if auth is None or (token := auth.credentials) not in api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


@router.get("/health", dependencies=[Depends(check_api_key)])
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """Test auth endpoint"""
    return {"status": "ok"}


@router.post(
    "/datasources",
    response_model=Result[DatasourceQueryResponse],
    dependencies=[Depends(check_api_key)],
)
async def create(
    request: Union[DatasourceCreateRequest, DatasourceServeRequest],
    service: Service = Depends(get_service),
) -> Result[DatasourceQueryResponse]:
    """Create a new Space entity

    Args:
        request (Union[DatasourceCreateRequest, DatasourceServeRequest]): The request
            to create a datasource. DatasourceServeRequest is deprecated.
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    res = await blocking_func_to_async(global_system_app, service.create, request)
    return Result.succ(res)


@router.put(
    "/datasources",
    response_model=Result[DatasourceQueryResponse],
    dependencies=[Depends(check_api_key)],
)
async def update(
    request: Union[DatasourceCreateRequest, DatasourceServeRequest],
    service: Service = Depends(get_service),
) -> Result[DatasourceQueryResponse]:
    """Update a Space entity

    Args:
        request (DatasourceServeRequest): The request
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    res = await blocking_func_to_async(global_system_app, service.update, request)
    return Result.succ(res)


@router.delete(
    "/datasources/{datasource_id}",
    response_model=Result[None],
    dependencies=[Depends(check_api_key)],
)
async def delete(
    datasource_id: str, service: Service = Depends(get_service)
) -> Result[None]:
    """Delete a Space entity

    Args:
        request (DatasourceServeRequest): The request
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    await blocking_func_to_async(global_system_app, service.delete, datasource_id)
    return Result.succ(None)


@router.delete(
    "/datasources/by-name/{db_name}",
    response_model=Result[bool],
    dependencies=[Depends(check_api_key)],
)
async def delete_by_name(
    db_name: str, service: Service = Depends(get_service)
) -> Result[bool]:
    """Delete a datasource by database name

    Args:
        db_name (str): The database name
        service (Service): The service
    Returns:
        Result[bool]: True if deletion was successful, False if datasource not found
    """
    result = await blocking_func_to_async(
        global_system_app, service.delete_by_db_name, db_name
    )
    return Result.succ(result)


@router.get(
    "/datasources/{datasource_id}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[DatasourceQueryResponse],
)
async def query(
    datasource_id: str, service: Service = Depends(get_service)
) -> Result[DatasourceQueryResponse]:
    """Query Space entities

    Args:
        request (DatasourceServeRequest): The request
        service (Service): The service
    Returns:
        List[ServeResponse]: The response
    """
    res = await blocking_func_to_async(global_system_app, service.get, datasource_id)
    return Result.succ(res)


@router.get(
    "/datasources",
    dependencies=[Depends(check_api_key)],
    response_model=Result[List[DatasourceQueryResponse]],
)
async def query_page(
    db_type: Optional[str] = Query(
        None, description="Database type, e.g. sqlite, mysql, etc."
    ),
    service: Service = Depends(get_service),
) -> Result[List[DatasourceQueryResponse]]:
    """Query Space entities

    Args:
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    res = await blocking_func_to_async(
        global_system_app, service.get_list, db_type=db_type
    )
    return Result.succ(res)


@router.get(
    "/datasource-types",
    dependencies=[Depends(check_api_key)],
    response_model=Result[ResourceTypes],
)
async def get_datasource_types(
    service: Service = Depends(get_service),
) -> Result[ResourceTypes]:
    """Get the datasource types."""
    res = await blocking_func_to_async(global_system_app, service.datasource_types)
    return Result.succ(res)


@router.post(
    "/datasources/test-connection",
    dependencies=[Depends(check_api_key)],
    response_model=Result[bool],
)
async def test_connection(
    request: DatasourceCreateRequest, service: Service = Depends(get_service)
) -> Result[bool]:
    """Test the connection using datasource configuration before creating it

    Args:
        request (DatasourceServeRequest): The datasource configuration to test
        service (Service): The service instance

    Returns:
        Result[bool]: The test result, True if connection is successful

    Raises:
        HTTPException: When the connection test fails
    """
    res = await blocking_func_to_async(
        global_system_app, service.test_connection, request
    )
    return Result.succ(res)


@router.post(
    "/datasources/{datasource_id}/refresh",
    dependencies=[Depends(check_api_key)],
    response_model=Result[bool],
)
async def refresh_datasource(
    datasource_id: str, service: Service = Depends(get_service)
) -> Result[bool]:
    """Refresh a datasource by its ID

    Args:
        datasource_id (str): The ID of the datasource to refresh
        service (Service): The service instance

    Returns:
        Result[bool]: The refresh result, True if the refresh was successful

    Raises:
        HTTPException: When the refresh operation fails
    """
    res = await blocking_func_to_async(
        global_system_app, service.refresh, datasource_id
    )
    return Result.succ(res)


@router.post(
    "/datasources/upload-excel",
    dependencies=[Depends(check_api_key)],
    response_model=Result[ExcelUploadResponse],
)
async def upload_excel(
    file: UploadFile = File(..., description="Excel file to upload"),
    table_name: Optional[str] = Query(None, description="Target table name"),
    force_reimport: bool = Query(False, description="Force reimport"),
    service: Service = Depends(get_service),
) -> Result[ExcelUploadResponse]:
    """Upload an Excel file and auto-register to datasource

    Args:
        file: Excel file to upload
        table_name: Target table name (optional, auto-generated if not provided)
        force_reimport: Force reimport even if cached
        service: Service instance

    Returns:
        Result[ExcelUploadResponse]: Upload result with database info

    Raises:
        HTTPException: When upload or processing fails
    """
    # 检查文件类型
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="Only Excel files (.xlsx, .xls) are supported"
        )

    # 保存临时文件
    temp_file = None
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
            force_reimport,
        )

        return Result.succ(ExcelUploadResponse(**result))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process Excel file: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@router.get(
    "/datasources/excel-info/{content_hash}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[ExcelInfoResponse],
)
async def get_excel_info(
    content_hash: str,
    service: Service = Depends(get_service),
) -> Result[ExcelInfoResponse]:
    """Get Excel info by content hash

    Args:
        content_hash: Content hash of the Excel file
        service: Service instance

    Returns:
        Result[ExcelInfoResponse]: Excel information

    Raises:
        HTTPException: When Excel info not found
    """
    try:
        excel_service = ExcelAutoRegisterService()
        info = await blocking_func_to_async(
            global_system_app, excel_service.get_excel_info, content_hash
        )

        if info is None:
            raise HTTPException(
                status_code=404, detail=f"Excel info not found for hash: {content_hash}"
            )

        return Result.succ(ExcelInfoResponse(**info))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get Excel info: {str(e)}"
        )


@router.put(
    "/datasources/excel-summary/{content_hash}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[bool],
)
async def update_excel_summary(
    content_hash: str,
    summary_prompt: str = Query(..., description="New summary prompt"),
    service: Service = Depends(get_service),
) -> Result[bool]:
    """Update Excel summary prompt

    Args:
        content_hash: Content hash of the Excel file
        summary_prompt: New summary prompt
        service: Service instance

    Returns:
        Result[bool]: Update result

    Raises:
        HTTPException: When update fails
    """
    try:
        excel_service = ExcelAutoRegisterService()
        await blocking_func_to_async(
            global_system_app,
            excel_service.update_summary_prompt,
            content_hash,
            summary_prompt,
        )

        return Result.succ(True)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update summary prompt: {str(e)}"
        )


def init_endpoints(system_app: SystemApp, config: ServeConfig) -> None:
    """Initialize the endpoints"""
    global global_system_app
    system_app.register(Service, config=config)
    global_system_app = system_app
