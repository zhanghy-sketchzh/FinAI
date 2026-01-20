import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import Executor
from typing import List, Optional, cast

import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from dbgpt._private.config import Config
from dbgpt.component import ComponentType
from dbgpt.configs import TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE
from dbgpt.core import ModelOutput
from dbgpt.core.awel import BaseOperator, CommonLLMHttpRequestBody
from dbgpt.core.awel.dag.dag_manager import DAGManager
from dbgpt.core.awel.util.chat_util import (
    _v1_create_completion_response,
    safe_chat_stream_with_dag_task,
)
from dbgpt.core.interface.file import FileStorageClient
from dbgpt.core.schema.api import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
from dbgpt.model.base import FlatSupportedModel
from dbgpt.model.cluster import BaseModelController, WorkerManager, WorkerManagerFactory
from dbgpt.util.executor_utils import (
    DefaultExecutorFactory,
    ExecutorFactory,
)
from dbgpt.util.file_client import FileClient
from dbgpt.util.tracer import SpanType, root_tracer
from dbgpt_app.knowledge.request.request import KnowledgeSpaceRequest
from dbgpt_app.knowledge.service import KnowledgeService
from dbgpt_app.openapi.api_view_model import (
    ChatSceneVo,
    ConversationVo,
    MessageVo,
    Result,
)
from dbgpt_app.scene import BaseChat, ChatFactory, ChatParam, ChatScene
from dbgpt_serve.agent.db.gpts_app import UserRecentAppsDao, adapt_native_app_model
from dbgpt_serve.core import blocking_func_to_async
from dbgpt_serve.datasource.manages.db_conn_info import DBConfig, DbTypeInfo
from dbgpt_serve.datasource.service.db_summary_client import DBSummaryClient
from dbgpt_serve.datasource.service.excel_auto_register import ExcelAutoRegisterService
from dbgpt_serve.flow.service.service import Service as FlowService
from dbgpt_serve.utils.auth import UserRequest, get_user_from_headers

router = APIRouter()
CFG = Config()
CHAT_FACTORY = ChatFactory()
logger = logging.getLogger(__name__)
knowledge_service = KnowledgeService()

model_semaphore = None
global_counter = 0


user_recent_app_dao = UserRecentAppsDao()


def __get_conv_user_message(conversations: dict):
    messages = conversations["messages"]
    for item in messages:
        if item["type"] == "human":
            return item["data"]["content"]
    return ""


def __new_conversation(chat_mode, user_name: str, sys_code: str) -> ConversationVo:
    unique_id = uuid.uuid1()
    return ConversationVo(
        conv_uid=str(unique_id),
        chat_mode=chat_mode,
        user_name=user_name,
        sys_code=sys_code,
    )


def get_db_list(user_id: str = None):
    dbs = CFG.local_db_manager.get_db_list(user_id=user_id)
    db_params = []
    for item in dbs:
        params: dict = {}
        params.update({"param": item["db_name"]})
        params.update({"type": item["db_type"]})
        db_params.append(params)
    return db_params


def plugins_select_info():
    plugins_infos: dict = {}
    for plugin in CFG.plugins:
        plugins_infos.update(
            {f"【{plugin._name}】=>{plugin._description}": plugin._name}
        )
    return plugins_infos


def get_db_list_info(user_id: str = None):
    dbs = CFG.local_db_manager.get_db_list(user_id=user_id)
    params: dict = {}
    for item in dbs:
        comment = item["comment"]
        if comment is not None and len(comment) > 0:
            params.update({item["db_name"]: comment})
    return params


def knowledge_list_info():
    """return knowledge space list"""
    params: dict = {}
    request = KnowledgeSpaceRequest()
    spaces = knowledge_service.get_knowledge_space(request)
    for space in spaces:
        params.update({space.name: space.desc})
    return params


def knowledge_list(user_id: str = None):
    """return knowledge space list"""
    request = KnowledgeSpaceRequest(user_id=user_id)
    spaces = knowledge_service.get_knowledge_space(request)
    space_list = []
    for space in spaces:
        params: dict = {}
        params.update({"param": space.name})
        params.update({"type": "space"})
        params.update({"space_id": space.id})
        space_list.append(params)
    return space_list


def get_model_controller() -> BaseModelController:
    controller = CFG.SYSTEM_APP.get_component(
        ComponentType.MODEL_CONTROLLER, BaseModelController
    )
    return controller


def get_worker_manager() -> WorkerManager:
    worker_manager = CFG.SYSTEM_APP.get_component(
        ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
    ).create()
    return worker_manager


def get_fs() -> FileStorageClient:
    return FileStorageClient.get_instance(CFG.SYSTEM_APP)


def get_dag_manager() -> DAGManager:
    """Get the global default DAGManager"""
    return DAGManager.get_instance(CFG.SYSTEM_APP)


def get_chat_flow() -> FlowService:
    """Get Chat Flow Service."""
    return FlowService.get_instance(CFG.SYSTEM_APP)


def get_executor() -> Executor:
    """Get the global default executor"""
    return CFG.SYSTEM_APP.get_component(
        ComponentType.EXECUTOR_DEFAULT,
        ExecutorFactory,
        or_register_component=DefaultExecutorFactory,
    ).create()


@router.get("/v1/chat/db/list", response_model=Result)
async def db_connect_list(
    db_name: Optional[str] = Query(default=None, description="database name"),
    user_info: UserRequest = Depends(get_user_from_headers),
):
    results = CFG.local_db_manager.get_db_list(
        db_name=db_name, user_id=user_info.user_id
    )
    # 排除部分数据库不允许用户访问
    if results and len(results):
        results = [
            d
            for d in results
            if d.get("db_name") not in ["auth", "dbgpt", "test", "public"]
        ]
    return Result.succ(results)


@router.post("/v1/chat/db/add", response_model=Result)
async def db_connect_add(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ(CFG.local_db_manager.add_db(db_config, user_token.user_id))


@router.get("/v1/permission/db/list", response_model=Result[List])
async def permission_db_list(
    db_name: str = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ()


@router.post("/v1/chat/db/edit", response_model=Result)
async def db_connect_edit(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ(CFG.local_db_manager.edit_db(db_config))


@router.post("/v1/chat/db/delete", response_model=Result[bool])
async def db_connect_delete(db_name: str = None):
    CFG.local_db_manager.db_summary_client.delete_db_profile(db_name)
    return Result.succ(CFG.local_db_manager.delete_db(db_name))


@router.post("/v1/chat/db/refresh", response_model=Result[bool])
async def db_connect_refresh(db_config: DBConfig = Body()):
    CFG.local_db_manager.db_summary_client.delete_db_profile(db_config.db_name)
    success = await CFG.local_db_manager.async_db_summary_embedding(
        db_config.db_name, db_config.db_type
    )
    return Result.succ(success)


async def async_db_summary_embedding(db_name, db_type):
    db_summary_client = DBSummaryClient(system_app=CFG.SYSTEM_APP)
    db_summary_client.db_summary_embedding(db_name, db_type)


@router.post("/v1/chat/db/test/connect", response_model=Result[bool])
async def test_connect(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    try:
        # TODO Change the synchronous call to the asynchronous call
        CFG.local_db_manager.test_connect(db_config)
        return Result.succ(True)
    except Exception as e:
        return Result.failed(code="E1001", msg=str(e))


@router.post("/v1/chat/db/summary", response_model=Result[bool])
async def db_summary(db_name: str, db_type: str):
    # TODO Change the synchronous call to the asynchronous call
    async_db_summary_embedding(db_name, db_type)
    return Result.succ(True)


@router.get("/v1/chat/db/support/type", response_model=Result[List[DbTypeInfo]])
async def db_support_types():
    support_types = CFG.local_db_manager.get_all_completed_types()
    db_type_infos = []
    for type in support_types:
        db_type_infos.append(
            DbTypeInfo(db_type=type.value(), is_file_db=type.is_file_db())
        )
    return Result[DbTypeInfo].succ(db_type_infos)


@router.post("/v1/chat/dialogue/scenes", response_model=Result[List[ChatSceneVo]])
async def dialogue_scenes(user_info: UserRequest = Depends(get_user_from_headers)):
    scene_vos: List[ChatSceneVo] = []
    new_modes: List[ChatScene] = [
        ChatScene.ChatWithDbExecute,
        ChatScene.ChatWithDbQA,
        ChatScene.ChatExcel,
        ChatScene.ChatKnowledge,
        ChatScene.ChatDashboard,
        ChatScene.ChatAgent,
    ]
    for scene in new_modes:
        scene_vo = ChatSceneVo(
            chat_scene=scene.value(),
            scene_name=scene.scene_name(),
            scene_describe=scene.describe(),
            param_title=",".join(scene.param_types()),
            show_disable=scene.show_disable(),
        )
        scene_vos.append(scene_vo)
    return Result.succ(scene_vos)


@router.post("/v1/resource/params/list", response_model=Result[List[dict]])
async def resource_params_list(
    resource_type: str,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    if resource_type == "database":
        result = get_db_list()
    elif resource_type == "knowledge":
        result = knowledge_list()
    elif resource_type == "tool":
        result = plugins_select_info()
    else:
        return Result.succ()
    return Result.succ(result)


@router.post("/v1/chat/mode/params/list", response_model=Result[List[dict]])
async def params_list(
    chat_mode: str = ChatScene.ChatNormal.value(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    if ChatScene.ChatWithDbQA.value() == chat_mode:
        result = get_db_list()
    elif ChatScene.ChatWithDbExecute.value() == chat_mode:
        result = get_db_list()
    elif ChatScene.ChatDashboard.value() == chat_mode:
        result = get_db_list()
        # 在 chat dashboard 模式下过滤掉 ad_data 数据库
        if result:
            result = [db for db in result if db.get("param") != "ad_data"]
    elif ChatScene.ChatExecution.value() == chat_mode:
        result = plugins_select_info()
    elif ChatScene.ChatKnowledge.value() == chat_mode:
        result = knowledge_list()
    elif ChatScene.ChatKnowledge.ExtractRefineSummary.value() == chat_mode:
        result = knowledge_list()
    else:
        return Result.succ()
    return Result.succ(result)


@router.post("/v1/resource/file/upload")
async def file_upload(
    chat_mode: str,
    conv_uid: str,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    sys_code: Optional[str] = None,
    model_name: Optional[str] = None,
    multi_table_mode: Optional[bool] = None,  # 多表模式：每个sheet存为独立的表
    doc_files: List[UploadFile] = File(...),
    user_token: UserRequest = Depends(get_user_from_headers),
    fs: FileStorageClient = Depends(get_fs),
):
    logger.info(
        f"file_upload:{conv_uid}, files:{[file.filename for file in doc_files]}"
    )

    bucket = "dbgpt_app_file"
    file_params = []

    for doc_file in doc_files:
        file_name = doc_file.filename
        custom_metadata = {
            "user_name": user_token.user_id,
            "sys_code": sys_code,
            "conv_uid": conv_uid,
        }

        file_uri = await blocking_func_to_async(
            CFG.SYSTEM_APP,
            fs.save_file,
            bucket,
            file_name,
            doc_file.file,
            custom_metadata=custom_metadata,
        )

        _, file_extension = os.path.splitext(file_name)
        file_param = {
            "is_oss": True,
            "file_path": file_uri,
            "file_name": file_name,
            "file_learning": False,
            "bucket": bucket,
        }
        file_params.append(file_param)
    if chat_mode == ChatScene.ChatExcel.value():
        if len(file_params) != 1:
            return Result.failed(msg="Only one file is supported for Excel chat.")
        file_param = file_params[0]
        _, file_extension = os.path.splitext(file_param["file_name"])
        if file_extension.lower() in [".xls", ".xlsx", ".csv", ".json", ".parquet"]:
            # 自动注册到数据源（仅支持 Excel）
            if file_extension.lower() in [".xls", ".xlsx"]:
                try:
                    # 获取本地文件路径 - download_file 返回 (path, metadata)
                    local_file_path, _ = await blocking_func_to_async(
                        CFG.SYSTEM_APP,
                        fs.download_file,
                        file_param["file_path"],  # URI
                        cache=True,
                    )

                    logger.info(f"Downloaded Excel to: {local_file_path}")

                    # 获取原始文件名（不是UUID）
                    original_filename = file_param.get("file_name", "unknown.xlsx")

                    # 获取LLM客户端（用于生成Schema理解）
                    try:
                        from dbgpt.model.cluster.client import DefaultLLMClient

                        worker_manager = get_worker_manager()
                        llm_client = DefaultLLMClient(worker_manager)

                        # 获取模型名称
                        if model_name:
                            default_model = model_name
                        else:
                            # 从 worker_manager 获取当前运行的 LLM 模型实例
                            from dbgpt.model.parameter import WorkerType
                            llm_instances = await worker_manager.get_all_model_instances(
                                WorkerType.LLM.value, healthy_only=True
                            )
                            if llm_instances and len(llm_instances) > 0:
                                default_model = llm_instances[0].worker_key.split("@")[0]
                            else:
                                # 从配置文件获取模型名称
                                from dbgpt.configs import GLOBAL_SYSTEM_CONFIG
                                if GLOBAL_SYSTEM_CONFIG and GLOBAL_SYSTEM_CONFIG.exists("models.llms"):
                                    from dbgpt.core.interface.parameter import LLMDeployModelParameters
                                    llm_configs = GLOBAL_SYSTEM_CONFIG.parse_config(
                                        LLMDeployModelParameters,
                                        prefix="models.llms",
                                    )
                                    if llm_configs:
                                        default_model = llm_configs[0].name
                                    else:
                                        default_model = CFG.LLM_MODEL
                                else:
                                    default_model = CFG.LLM_MODEL

                        logger.info(f"✅ 成功获取LLM客户端，使用模型: {default_model}")
                    except Exception as e:
                        logger.warning(f"⚠️ 获取LLM客户端失败: {e}，将使用fallback方法")
                        llm_client = None
                        default_model = None
                    # 自动注册到数据源 - 传递LLM客户端、模型名称、原始文件名和会话ID
                    # 同时传入 system_app，以便在需要时能够重新获取 LLM 客户端
                    excel_service = ExcelAutoRegisterService(
                        llm_client=llm_client,
                        model_name=default_model,
                        system_app=CFG.SYSTEM_APP,
                    )

                    # 检测Excel文件中的sheet数量
                    import pandas as pd

                    try:
                        excel_file = pd.ExcelFile(local_file_path)
                        sheet_count = len(excel_file.sheet_names)
                        sheet_names = excel_file.sheet_names
                        logger.info(
                            f"Excel文件包含 {sheet_count} 个sheet: {sheet_names}"
                        )
                    except Exception as e:
                        logger.warning(f"无法读取Excel sheet信息: {e}，将使用默认设置")
                        sheet_count = 1
                        sheet_names = None

                    # 判断是否使用多表模式
                    # 如果明确指定了 multi_table_mode，则使用指定的值
                    # 否则，如果有多个sheet，默认使用多表模式
                    use_multi_table_mode = multi_table_mode if multi_table_mode is not None else (sheet_count > 1)
                    
                    if use_multi_table_mode and sheet_count > 1:
                        # 多表模式：每个sheet存为独立的表
                        logger.info("使用多表模式，每个sheet将存为独立的表")
                        register_result = await blocking_func_to_async(
                            CFG.SYSTEM_APP,
                            excel_service.process_excel_multi_tables,
                            local_file_path,
                            False,  # 使用缓存
                            original_filename,  # 传递原始文件名
                            conv_uid,  # 传递会话ID
                        )
                        
                        # 多表模式的结果处理
                        file_param["multi_table_mode"] = True
                        file_param["excel_registered"] = True
                        file_param["db_name"] = register_result["db_name"]
                        file_param["db_path"] = register_result.get("db_path")
                        file_param["file_hash"] = register_result.get("file_hash")
                        file_param["tables"] = register_result.get("tables", [])
                        file_param["content_hash"] = register_result.get("file_hash")
                        file_param["register_status"] = register_result["status"]
                        
                        # 构建多表预览数据
                        tables = register_result.get("tables", [])
                        file_param["preview_data"] = {
                            "file_name": original_filename,
                            "tables": [
                                {
                                    "sheet_name": t.get("sheet_name", ""),
                                    "table_name": t.get("table_name", ""),
                                    "columns": t.get("preview_data", {}).get("columns", []),
                                    "rows": t.get("preview_data", {}).get("rows", []),
                                    "total": t.get("row_count", 0),
                                    "file_name": original_filename,
                                }
                                for t in tables
                            ]
                        }
                        
                        # 汇总信息
                        total_rows = sum(t.get("row_count", 0) for t in tables)
                        total_cols = max((t.get("column_count", 0) for t in tables), default=0)
                        file_param["row_count"] = total_rows
                        file_param["column_count"] = total_cols
                        file_param["table_name"] = ", ".join(t.get("table_name", "") for t in tables)
                        
                        # 合并所有表的 summary_prompt
                        summary_prompts = [t.get("summary_prompt", "") for t in tables if t.get("summary_prompt")]
                        file_param["summary_prompt"] = "\n\n".join(summary_prompts) if summary_prompts else None
                        
                        logger.info(
                            f"✅ Excel multi-table registered: {original_filename} -> "
                            f"db={register_result['db_name']}, "
                            f"tables={len(tables)}, "
                            f"status={register_result['status']}"
                        )
                    else:
                        # 单表模式或合并模式
                        merge_sheets = sheet_count > 1 and not use_multi_table_mode
                        if merge_sheets:
                            logger.info("检测到多个sheet，使用合并模式")
                        
                        register_result = await blocking_func_to_async(
                            CFG.SYSTEM_APP,
                            excel_service.process_excel,
                            local_file_path,
                            None,  # 自动生成表名
                            False,  # 使用缓存
                            original_filename,  # 传递原始文件名
                            conv_uid,  # 传递会话ID
                            sheet_names,  # 传递sheet名称列表（None表示所有sheet）
                            merge_sheets,  # 是否合并多个sheet
                            "数据类型",  # 来源列名
                        )

                        # 将注册结果添加到 file_param 中（单表模式）
                        file_param["multi_table_mode"] = False
                        file_param["excel_registered"] = True
                        file_param["db_name"] = register_result["db_name"]
                        file_param["db_path"] = register_result.get(
                            "db_path"
                        )  # 添加SQLite数据库路径
                        file_param["table_name"] = register_result["table_name"]
                        file_param["content_hash"] = register_result["content_hash"]
                        file_param["register_status"] = register_result["status"]
                        file_param["summary_prompt"] = register_result.get("summary_prompt")
                        file_param["data_schema_json"] = register_result.get(
                            "data_schema_json"
                        )
                        file_param["row_count"] = register_result.get("row_count")
                        file_param["column_count"] = register_result.get("column_count")
                        file_param["top_10_rows"] = register_result.get("top_10_rows")
                        file_param["preview_data"] = register_result.get("preview_data")
                        file_param["suggested_questions"] = register_result.get(
                            "suggested_questions"
                        )

                        # 存储到 excel_schema 表
                        try:
                            from dbgpt_app.scene.chat_data.chat_excel.excel_schema_db import (  # noqa: E501
                                ExcelSchemaDao,
                            )

                            excel_schema_dao = ExcelSchemaDao()
                            excel_schema_dao.save_or_update(
                                conv_uid=conv_uid,
                                file_name=original_filename,
                                table_name=register_result["table_name"],
                                row_count=register_result["row_count"],
                                column_count=register_result["column_count"],
                                top_10_rows=register_result.get("top_10_rows", []),
                                data_description=register_result.get("summary_prompt"),
                                data_schema_json=register_result.get("data_schema_json"),
                                suggested_questions=register_result.get(
                                    "suggested_questions"
                                ),
                                file_path=file_param["file_path"],
                                db_path=register_result.get("db_path"),
                                user_id=user_token.user_id,
                                sys_code=sys_code,
                            )
                            logger.info(
                                f"✅ Excel schema saved to database for conv_uid={conv_uid}"
                            )
                        except Exception as e:
                            logger.error(f"❌ Failed to save Excel schema to database: {e}")
                            import traceback

                            logger.error(traceback.format_exc())

                        logger.info(
                            f"✅ Excel auto-registered: {file_param['file_name']} -> "
                            f"db={register_result['db_name']}, "
                            f"table={register_result['table_name']}, "
                            f"status={register_result['status']}, "
                            f"rows={register_result['row_count']}"
                        )
                except Exception as e:
                    logger.error(f"❌ Failed to auto-register Excel to datasource: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    file_param["excel_registered"] = False
                    file_param["register_error"] = str(e)

            # Prepare the chat
            file_param["file_learning"] = True
            dialogue = ConversationVo(
                user_input="Learn from the file",
                conv_uid=conv_uid,
                chat_mode=chat_mode,
                select_param=file_param,
                model_name=model_name,
                user_name=user_token.user_id,
                sys_code=sys_code,
            )

            if temperature is not None:
                dialogue.temperature = temperature
            if max_new_tokens is not None:
                dialogue.max_new_tokens = max_new_tokens

            chat: BaseChat = await get_chat_instance(dialogue)
            await chat.prepare()
            # Refresh messages

    # If only one file was uploaded, return the single file_param directly
    # Otherwise return the array of file_params
    result = file_params[0] if len(file_params) == 1 else file_params
    return Result.succ(result)


@router.post("/v1/resource/file/delete")
async def file_delete(
    conv_uid: str,
    file_key: str,
    user_name: Optional[str] = None,
    sys_code: Optional[str] = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"file_delete:{conv_uid},{file_key}")
    oss_file_client = FileClient()

    return Result.succ(
        await oss_file_client.delete_file(conv_uid=conv_uid, file_key=file_key)
    )


@router.get("/v1/resource/excel/info")
async def get_excel_info(
    conv_uid: str,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    """获取Excel文件的基本信息

    Args:
        conv_uid: 会话ID
        user_token: 用户令牌

    Returns:
        Excel基本信息，包括：
        - 前10行数据
        - 行列数
        - 数据描述
        - 推荐问题列表
    """
    try:
        from dbgpt_app.scene.chat_data.chat_excel.excel_schema_db import ExcelSchemaDao

        excel_schema_dao = ExcelSchemaDao()
        schema_entity = excel_schema_dao.get_by_conv_uid(conv_uid)

        if not schema_entity:
            return Result.failed(msg=f"未找到会话 {conv_uid} 的Excel信息")

        # 转换为字典
        excel_info = excel_schema_dao.to_dict(schema_entity)

        return Result.succ(excel_info)
    except Exception as e:
        logger.error(f"获取Excel信息失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return Result.failed(msg=f"获取Excel信息失败: {str(e)}")


@router.post("/v1/resource/file/read")
async def file_read(
    conv_uid: str,
    file_key: str,
    user_name: Optional[str] = None,
    sys_code: Optional[str] = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"file_read:{conv_uid},{file_key}")
    file_client = FileClient()
    res = await file_client.read_file(conv_uid=conv_uid, file_key=file_key)
    df = pd.read_excel(res, index_col=False)
    return Result.succ(df.to_json(orient="records", date_format="iso", date_unit="s"))


def get_hist_messages(conv_uid: str, user_name: str = None):
    from dbgpt_serve.conversation.service.service import Service as ConversationService

    instance: ConversationService = ConversationService.get_instance(CFG.SYSTEM_APP)
    return instance.get_history_messages({"conv_uid": conv_uid, "user_name": user_name})


async def get_chat_instance(dialogue: ConversationVo = Body()) -> BaseChat:
    logger.info(f"get_chat_instance:{dialogue}")
    if not dialogue.chat_mode:
        dialogue.chat_mode = ChatScene.ChatNormal.value()
    if not dialogue.conv_uid:
        conv_vo = __new_conversation(
            dialogue.chat_mode, dialogue.user_name, dialogue.sys_code
        )
        dialogue.conv_uid = conv_vo.conv_uid

    if not ChatScene.is_valid_mode(dialogue.chat_mode):
        raise StopAsyncIteration(
            Result.failed("Unsupported Chat Mode," + dialogue.chat_mode + "!")
        )

    chat_param = ChatParam(
        chat_session_id=dialogue.conv_uid,
        user_name=dialogue.user_name,
        sys_code=dialogue.sys_code,
        current_user_input=dialogue.user_input,
        select_param=dialogue.select_param,
        model_name=dialogue.model_name,
        app_code=dialogue.app_code,
        ext_info=dialogue.ext_info,
        temperature=dialogue.temperature,
        max_new_tokens=dialogue.max_new_tokens,
        prompt_code=dialogue.prompt_code,
        chat_mode=ChatScene.of_mode(dialogue.chat_mode),
    )
    chat: BaseChat = await blocking_func_to_async(
        CFG.SYSTEM_APP,
        CHAT_FACTORY.get_implementation,
        dialogue.chat_mode,
        CFG.SYSTEM_APP,
        **{"chat_param": chat_param},
    )
    return chat


@router.post("/v1/chat/prepare")
async def chat_prepare(
    dialogue: ConversationVo = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(json.dumps(dialogue.__dict__))
    # dialogue.model_name = CFG.LLM_MODEL
    dialogue.user_name = user_token.user_id if user_token else dialogue.user_name
    logger.info(f"chat_prepare:{dialogue}")
    ## check conv_uid
    chat: BaseChat = await get_chat_instance(dialogue)

    await chat.prepare()

    # Refresh messages
    return Result.succ(get_hist_messages(dialogue.conv_uid, user_token.user_id))


@router.post("/v1/chat/completions")
async def chat_completions(
    dialogue: ConversationVo = Body(),
    flow_service: FlowService = Depends(get_chat_flow),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(
        f"chat_completions:{dialogue.chat_mode},{dialogue.select_param},"
        f"{dialogue.model_name}, timestamp={int(time.time() * 1000)}"
    )
    dialogue.user_name = user_token.user_id if user_token else dialogue.user_name
    dialogue = adapt_native_app_model(dialogue)
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
    }
    try:
        domain_type = _parse_domain_type(dialogue)
        if dialogue.chat_mode == ChatScene.ChatAgent.value():
            from dbgpt_serve.agent.agents.controller import multi_agents

            dialogue.ext_info.update({"model_name": dialogue.model_name})
            dialogue.ext_info.update({"incremental": dialogue.incremental})
            dialogue.ext_info.update({"temperature": dialogue.temperature})
            return StreamingResponse(
                multi_agents.app_agent_chat(
                    conv_uid=dialogue.conv_uid,
                    chat_mode=dialogue.chat_mode,
                    gpts_name=dialogue.app_code,
                    user_query=dialogue.user_input,
                    user_code=dialogue.user_name,
                    sys_code=dialogue.sys_code,
                    **dialogue.ext_info,
                ),
                headers=headers,
                media_type="text/event-stream",
            )
        elif dialogue.chat_mode == ChatScene.ChatFlow.value():
            flow_req = CommonLLMHttpRequestBody(
                model=dialogue.model_name,
                messages=dialogue.user_input,
                stream=True,
                conv_uid=dialogue.conv_uid,
                span_id=root_tracer.get_current_span_id(),
                chat_mode=dialogue.chat_mode,
                chat_param=dialogue.select_param,
                user_name=dialogue.user_name,
                sys_code=dialogue.sys_code,
                incremental=dialogue.incremental,
            )
            return StreamingResponse(
                flow_service.chat_stream_flow_str(dialogue.select_param, flow_req),
                headers=headers,
                media_type="text/event-stream",
            )
        elif domain_type is not None and domain_type != "Normal":
            return StreamingResponse(
                chat_with_domain_flow(dialogue, domain_type),
                headers=headers,
                media_type="text/event-stream",
            )

        else:
            with root_tracer.start_span(
                "get_chat_instance", span_type=SpanType.CHAT, metadata=dialogue.dict()
            ):
                chat: BaseChat = await get_chat_instance(dialogue)

            if not chat.prompt_template.stream_out:
                # 确保model_name不为None
                effective_model_name = dialogue.model_name or "default"
                return StreamingResponse(
                    no_stream_generator(chat, effective_model_name, dialogue.conv_uid),
                    headers=headers,
                    media_type="text/event-stream",
                )
            else:
                # 确保model_name不为None
                effective_model_name = dialogue.model_name or "default"
                return StreamingResponse(
                    stream_generator(
                        chat,
                        dialogue.incremental,
                        effective_model_name,
                        openai_format=dialogue.incremental,
                    ),
                    headers=headers,
                    media_type="text/plain",
                )
    except Exception as e:
        logger.exception(f"Chat Exception!{dialogue}", e)

        async def error_text(err_msg):
            yield f"data:{err_msg}\n\n"

        return StreamingResponse(
            error_text(str(e)),
            headers=headers,
            media_type="text/plain",
        )
    finally:
        # write to recent usage app.
        if dialogue.user_name is not None and dialogue.app_code is not None:
            user_recent_app_dao.upsert(
                user_code=dialogue.user_name,
                sys_code=dialogue.sys_code,
                app_code=dialogue.app_code,
            )


@router.post("/v1/chat/topic/terminate")
async def terminate_topic(
    conv_id: str,
    round_index: int,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"terminate_topic:{conv_id},{round_index}")
    try:
        from dbgpt_serve.agent.agents.controller import multi_agents

        return Result.succ(await multi_agents.topic_terminate(conv_id))
    except Exception as e:
        logger.exception("Topic terminate error!")
        return Result.failed(code="E0102", msg=str(e))


@router.get("/v1/model/types")
async def model_types(controller: BaseModelController = Depends(get_model_controller)):
    logger.info("/controller/model/types")
    try:
        types = set()
        models = await controller.get_all_instances(healthy_only=True)
        for model in models:
            worker_name, worker_type = model.model_name.split("@")
            if worker_type == "llm" and worker_name not in [
                "codegpt_proxyllm",
                "text2sql_proxyllm",
            ]:
                types.add(worker_name)
        return Result.succ(list(types))

    except Exception as e:
        return Result.failed(code="E000X", msg=f"controller model types error {e}")


@router.get("/v1/test")
async def test():
    return "service status is UP"


@router.get(
    "/v1/model/supports",
    deprecated=True,
    description="This endpoint is deprecated. Please use "
    "`/api/v2/serve/model/model-types` instead. It will be removed in v0.8.0.",
)
async def model_supports(worker_manager: WorkerManager = Depends(get_worker_manager)):
    logger.warning(
        "The endpoint `/api/v1/model/supports` is deprecated. Please use "
        "`/api/v2/serve/model/model-types` instead. It will be removed in v0.8.0."
    )
    try:
        models = await worker_manager.supported_models()
        return Result.succ(FlatSupportedModel.from_supports(models))
    except Exception as e:
        return Result.failed(code="E000X", msg=f"Fetch supportd models error {e}")


async def flow_stream_generator(func, incremental: bool, model_name: str):
    stream_id = f"chatcmpl-{str(uuid.uuid1())}"
    previous_response = ""
    async for chunk in func:
        if chunk:
            msg = chunk.replace("\ufffd", "")
            if incremental:
                incremental_output = msg[len(previous_response) :]
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=incremental_output),
                )
                chunk = ChatCompletionStreamResponse(
                    id=stream_id, choices=[choice_data], model=model_name
                )
                _content = json.dumps(
                    chunk.dict(exclude_unset=True), ensure_ascii=False
                )
                yield f"data: {_content}\n\n"
            else:
                # TODO generate an openai-compatible streaming responses
                msg = msg.replace("\n", "\\n")
                yield f"data:{msg}\n\n"
            previous_response = msg
    if incremental:
        yield "data: [DONE]\n\n"


async def no_stream_generator(chat, model_name: str, conv_uid: Optional[str] = None):
    with root_tracer.start_span("no_stream_generator"):
        msg = await chat.nostream_call()
        stream_id = conv_uid or f"chatcmpl-{str(uuid.uuid1())}"
        yield _v1_create_completion_response(msg, None, model_name, stream_id)


async def stream_generator(
    chat,
    incremental: bool,
    model_name: str,
    text_output: bool = True,
    openai_format: bool = False,
    conv_uid: Optional[str] = None,
):
    """Generate streaming responses

    Our goal is to generate an openai-compatible streaming responses.
    Currently, the incremental response is compatible, and the full response will be
    transformed in the future.

    Args:
        chat (BaseChat): Chat instance.
        incremental (bool): Used to control whether the content is returned
            incrementally or in full each time.
        model_name (str): The model name

    Yields:
        _type_: streaming responses
    """
    span = root_tracer.start_span("stream_generator")
    msg = "[LLM_ERROR]: llm server has no output, maybe your prompt template is wrong."

    stream_id = conv_uid or f"chatcmpl-{str(uuid.uuid1())}"
    try:
        if incremental and not openai_format:
            raise ValueError("Incremental response must be openai-compatible format.")
        async for chunk in chat.stream_call(
            text_output=text_output, incremental=incremental
        ):
            if not chunk:
                await asyncio.sleep(0.02)
                continue

            if openai_format:
                # Must be ModelOutput
                output: ModelOutput = cast(ModelOutput, chunk)
                text = None
                think_text = None
                if output.has_text:
                    text = output.text
                if output.has_thinking:
                    think_text = output.thinking_text
                if incremental:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            role="assistant", content=text, reasoning_content=think_text
                        ),
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=stream_id, choices=[choice_data], model=model_name
                    )
                    _content = json.dumps(
                        chunk.dict(exclude_unset=True), ensure_ascii=False
                    )
                    yield f"data: {_content}\n\n"
                else:
                    if output.usage:
                        usage = UsageInfo(**output.usage)
                    else:
                        usage = UsageInfo()
                    _content = _v1_create_completion_response(
                        text, think_text, model_name, stream_id, usage
                    )
                    yield _content
            else:
                msg = chunk.replace("\ufffd", "")
                _content = _v1_create_completion_response(
                    msg, None, model_name, stream_id
                )
                yield _content
            await asyncio.sleep(0.02)
        if incremental:
            yield "data: [DONE]\n\n"
        span.end()
    except Exception as e:
        logger.exception("stream_generator error")
        yield f"data: [SERVER_ERROR]{str(e)}\n\n"
        if incremental:
            yield "data: [DONE]\n\n"


def message2Vo(message: dict, order, model_name) -> MessageVo:
    return MessageVo(
        role=message["type"],
        context=message["data"]["content"],
        order=order,
        model_name=model_name,
    )


def _parse_domain_type(dialogue: ConversationVo) -> Optional[str]:
    if dialogue.chat_mode == ChatScene.ChatKnowledge.value():
        # Supported in the knowledge chat
        if dialogue.app_code == "" or dialogue.app_code == "chat_knowledge":
            spaces = knowledge_service.get_knowledge_space(
                KnowledgeSpaceRequest(name=dialogue.select_param)
            )
        else:
            spaces = knowledge_service.get_knowledge_space(
                KnowledgeSpaceRequest(name=dialogue.select_param)
            )
        if len(spaces) == 0:
            raise ValueError(f"Knowledge space {dialogue.select_param} not found")
        dialogue.select_param = spaces[0].name
        if spaces[0].domain_type:
            return spaces[0].domain_type
    else:
        return None


async def chat_with_domain_flow(dialogue: ConversationVo, domain_type: str):
    """Chat with domain flow"""
    dag_manager = get_dag_manager()
    dags = dag_manager.get_dags_by_tag(TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE, domain_type)
    if not dags or not dags[0].leaf_nodes:
        raise ValueError(f"Cant find the DAG for domain type {domain_type}")

    end_task = cast(BaseOperator, dags[0].leaf_nodes[0])
    space = dialogue.select_param
    connector_manager = CFG.local_db_manager
    # TODO: Some flow maybe not connector
    db_list = [item["db_name"] for item in connector_manager.get_db_list()]
    db_names = [item for item in db_list if space in item]
    if len(db_names) == 0:
        raise ValueError(f"fin repost dbname {space}_fin_report not found.")
    flow_ctx = {"space": space, "db_name": db_names[0]}
    request = CommonLLMHttpRequestBody(
        model=dialogue.model_name,
        messages=dialogue.user_input,
        stream=True,
        extra=flow_ctx,
        conv_uid=dialogue.conv_uid,
        span_id=root_tracer.get_current_span_id(),
        chat_mode=dialogue.chat_mode,
        chat_param=dialogue.select_param,
        user_name=dialogue.user_name,
        sys_code=dialogue.sys_code,
        incremental=dialogue.incremental,
    )
    async for output in safe_chat_stream_with_dag_task(end_task, request, False):
        text = output.gen_text_with_thinking()
        if text:
            text = text.replace("\n", "\\n")
        if output.error_code != 0:
            yield _v1_create_completion_response(
                f"[SERVER_ERROR]{text}", None, dialogue.model_name, dialogue.conv_uid
            )
            break
        else:
            yield _v1_create_completion_response(
                text, None, dialogue.model_name, dialogue.conv_uid
            )
