import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Type, Union, Optional

from dbgpt import SystemApp
from dbgpt.agent.util.api_call import ApiCall
from dbgpt.configs.model_config import DATA_DIR
from dbgpt.core import (
    ModelOutput,
    ModelRequest,
    ModelMessage,
    ModelMessageRoleType,
    ChatPromptTemplate,
    SystemPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
)
from dbgpt.core.interface.file import _SCHEMA, FileStorageClient
from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.util.json_utils import EnhancedJSONEncoder
from dbgpt.util.tracer import root_tracer, trace
from dbgpt_app.scene import BaseChat, ChatScene
from dbgpt_app.scene.base_chat import ChatParam
from dbgpt_app.scene.chat_data.chat_excel.config import ChatExcelConfig
from dbgpt_app.scene.chat_data.chat_excel.excel_learning.chat import ExcelLearning
from dbgpt_app.scene.chat_data.chat_excel.excel_reader import ExcelReader
from dbgpt_app.scene.chat_data.chat_excel.excel_schema_db import ExcelSchemaDao
from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.language_detector import (
    detect_language,
)

logger = logging.getLogger(__name__)


class ChatExcel(BaseChat):
    """a Excel analyzer to analyze Excel Data"""

    chat_scene: str = ChatScene.ChatExcel.value()

    @classmethod
    def param_class(cls) -> Type[ChatExcelConfig]:
        return ChatExcelConfig

    def __init__(self, chat_param: ChatParam, system_app: SystemApp):
        """Chat Excel Module Initialization
        Args:
           - chat_param: Dict
            - chat_session_id: (str) chat session_id
            - current_user_input: (str) current user input
            - model_name:(str) llm model name
            - select_param:(str) file path
        """
        self.fs_client = FileStorageClient.get_instance(system_app)
        self.select_param = chat_param.select_param
        if not self.select_param:
            raise ValueError("Please upload the Excel document you want to talk to！")
        self.model_name = chat_param.model_name
        self.curr_config = chat_param.real_app_config(ChatExcelConfig)
        self.chat_param = chat_param
        self._bucket = "dbgpt_app_file"

        # 检查是否有缓存的DuckDB数据库路径
        use_existing_db = False
        duckdb_path = None
        duckdb_table_name = None  # 保存DuckDB中的实际表名

        # ✅ 调试：打印select_param的类型和内容
        logger.info(f"🔍 select_param类型: {type(self.select_param)}")
        logger.info(
            f"🔍 select_param内容: {self.select_param if isinstance(self.select_param, dict) else str(self.select_param)[:200]}"
        )

        # ✅ 修复：如果select_param是字符串，先解析为字典
        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                import json

                select_param_dict = json.loads(self.select_param)
                logger.info(f"✅ 成功解析select_param字符串为字典")
            except json.JSONDecodeError as e:
                logger.error(f"❌ 解析select_param失败: {e}")
                select_param_dict = {}

        if isinstance(select_param_dict, dict):
            # 如果有db_path，说明excel_auto_register已经处理过了
            duckdb_path = select_param_dict.get("db_path")
            duckdb_table_name = select_param_dict.get("table_name")  # 获取实际表名
            self._content_hash = select_param_dict.get(
                "content_hash"
            )  # 保存 content_hash 用于更新领域知识
            logger.info(f"🔍 db_path: {duckdb_path}")
            logger.info(f"🔍 table_name: {duckdb_table_name}")
            logger.info(
                f"🔍 content_hash: {self._content_hash[:16] if self._content_hash else 'None'}..."
            )

            # 如果有 content_hash，从数据库重新加载最新的 data_schema_json
            if self._content_hash:
                try:
                    import sqlite3
                    from pathlib import Path

                    current_file = Path(__file__)
                    project_root = (
                        current_file.parent.parent.parent.parent.parent.parent.parent.parent.parent
                    )
                    cache_dir = (
                        project_root / "packages" / "pilot" / "data" / "excel_cache"
                    )
                    meta_db_path = cache_dir / "excel_metadata.db"

                    if meta_db_path.exists():
                        conn = sqlite3.connect(str(meta_db_path))
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            SELECT data_schema_json 
                            FROM excel_metadata 
                            WHERE content_hash = ?
                        """,
                            (self._content_hash,),
                        )
                        result = cursor.fetchone()
                        conn.close()

                        if result and result[0]:
                            select_param_dict["data_schema_json"] = result[0]
                            if isinstance(self.select_param, str):
                                self.select_param = json.dumps(
                                    select_param_dict, ensure_ascii=False
                                )
                            else:
                                self.select_param = select_param_dict
                except Exception as e:
                    logger.warning(f"从数据库重新加载 data_schema_json 失败: {e}")

            if duckdb_path and os.path.exists(duckdb_path):
                use_existing_db = True
                logger.info(f"✅ 检测到已存在的DuckDB数据库: {duckdb_path}")
                logger.info(f"   DuckDB表名: {duckdb_table_name}")
            else:
                if duckdb_path:
                    logger.warning(f"⚠️ db_path存在但文件不存在: {duckdb_path}")
                else:
                    logger.warning(f"⚠️ select_param中没有db_path字段")
        else:
            logger.warning(f"⚠️ select_param不是字典类型，使用传统Excel导入")

        file_path, file_name, database_file_path, database_file_id = self._resolve_path(
            select_param_dict,  # ✅ 使用解析后的字典
            chat_param.chat_session_id,
            self.fs_client,
            self._bucket,
            duckdb_path=duckdb_path,  # 传递DuckDB路径
        )

        # 如果有DuckDB数据库，直接使用DuckDB连接
        if use_existing_db and duckdb_path:
            # 使用DuckDB缓存时，直接使用实际表名，无需创建新表
            self._curr_table = duckdb_table_name if duckdb_table_name else "data_analysis_table"
            self.excel_reader = self._create_reader_from_duckdb(
                chat_param.chat_session_id,
                duckdb_path,
                file_name,
                duckdb_table_name,  # 传递DuckDB中的实际表名
            )
            logger.info(f"✅ 使用DuckDB缓存，直接使用表名: {self._curr_table}")
        else:
            # 传统方式：从Excel文件导入到DuckDB
            self._curr_table = "data_analysis_table"
            self.excel_reader = ExcelReader(
                chat_param.chat_session_id,
                file_path,
                file_name,
                read_type="direct",
                database_name=database_file_path,
                table_name=self._curr_table,
                duckdb_extensions_dir=self.curr_config.duckdb_extensions_dir,
                force_install=self.curr_config.force_install,
            )

        self._file_name = file_name
        self._database_file_path = database_file_path
        self._database_file_id = database_file_id
        self._query_rewrite_result = None  # 保存Query改写结果
        self._last_sql_error = None  # 保存最后一次SQL执行错误

        self.api_call = ApiCall()
        super().__init__(chat_param=chat_param, system_app=system_app)

    def _create_reader_from_duckdb(
        self,
        conv_uid: str,
        duckdb_path: str,
        file_name: str,
        duckdb_table_name: str = None,
    ):
        """
        从DuckDB数据库创建ExcelReader（直接使用DuckDB连接）

        Args:
            conv_uid: 会话ID
            duckdb_path: DuckDB数据库文件路径
            file_name: 文件名
            duckdb_table_name: DuckDB中的实际表名（如果为None，会尝试自动检测）
        """
        import duckdb

        # 直接连接DuckDB数据库文件（只读模式）
        db_conn = duckdb.connect(database=duckdb_path, read_only=True)

        try:
            # 如果没有提供表名，尝试自动检测
            if not duckdb_table_name:
                logger.info("未提供DuckDB表名，尝试自动检测...")
                tables_result = db_conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
                if tables_result:
                    duckdb_table_name = tables_result[0][0]
                    logger.info(f"自动检测到表名: {duckdb_table_name}")
                else:
                    raise ValueError(f"在DuckDB数据库中未找到任何表: {duckdb_path}")

            # 直接使用DuckDB中的表（无需导入）
            logger.info(
                f"✅ 直接使用DuckDB表 '{duckdb_table_name}'（无需导入）"
            )

            # ✅ 验证表结构
            try:
                # 获取列名
                columns_info = db_conn.execute(
                    f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{duckdb_table_name}' AND table_schema = 'main'
                    ORDER BY ordinal_position
                """
                ).fetchall()

                column_names = [col[0] for col in columns_info]
                logger.info(f"DuckDB表 '{duckdb_table_name}' 的列名: {column_names}")

                # 获取行数
                row_count = db_conn.execute(
                    f"SELECT COUNT(*) FROM {duckdb_table_name}"
                ).fetchone()[0]
                logger.info(f"DuckDB表 '{duckdb_table_name}' 的行数: {row_count}")

                # 获取前3行数据用于验证
                sample_data = db_conn.execute(
                    f"SELECT * FROM {duckdb_table_name} LIMIT 3"
                ).fetchall()
                logger.info(
                    f"DuckDB表 '{duckdb_table_name}' 的前3行: {sample_data[:2]}"
                )  # 只打印前2行避免日志过长

            except Exception as e:
                logger.error(f"验证表结构时出错: {e}")

            # 创建一个虚拟的ExcelReader对象，直接使用DuckDB连接
            reader = object.__new__(ExcelReader)
            reader.conv_uid = conv_uid
            reader.db = db_conn
            # 使用DuckDB缓存时，temp_table_name和table_name都设置为实际表名
            reader.temp_table_name = duckdb_table_name  # 设置为实际表名，供ExcelLearning使用
            reader.table_name = duckdb_table_name  # 直接使用DuckDB中的表名
            reader.excel_file_name = file_name

            return reader

        except Exception as e:
            logger.error(f"从DuckDB读取数据失败: {e}")
            db_conn.close()
            raise

    def _resolve_path(
        self,
        file_param: Any,
        conv_uid: str,
        fs_client: FileStorageClient,
        bucket: str,
        duckdb_path: str = None,
    ) -> Union[str, str, str]:
        if isinstance(file_param, str) and os.path.isabs(file_param):
            file_path = file_param
            file_name = os.path.basename(file_param)
        else:
            if isinstance(file_param, dict):
                file_path = file_param.get("file_path", None)
                if not file_path:
                    raise ValueError("Not find file path!")
                else:
                    file_name = os.path.basename(file_path.replace(f"{conv_uid}_", ""))

            else:
                temp_obj = json.loads(file_param)
                file_path = temp_obj.get("file_path")
                if not file_path:
                    raise ValueError("Not find file path!")
                file_name = os.path.basename(file_path.replace(f"{conv_uid}_", ""))

        # 如果有DuckDB路径，直接使用它作为database_file_path
        if duckdb_path and os.path.exists(duckdb_path):
            database_file_path = duckdb_path
            database_file_id = None
            logger.info(f"✅ 使用缓存的DuckDB数据库: {duckdb_path}")
        else:
            # 传统方式：使用DuckDB
            database_root_path = os.path.join(DATA_DIR, "_chat_excel_tmp")
            os.makedirs(database_root_path, exist_ok=True)
            database_file_path = os.path.join(
                database_root_path, f"_chat_excel_{file_name}.duckdb"
            )
            database_file_id = None

        if file_path.startswith(_SCHEMA):
            file_path, file_meta = fs_client.download_file(file_path, dest_dir=DATA_DIR)
            file_name = os.path.basename(file_path)

            if not duckdb_path:  # 只在没有DuckDB路径时才创建新的DuckDB
                database_file_path = os.path.join(
                    database_root_path, f"_chat_excel_{file_name}.duckdb"
                )
                database_file_id = f"{file_meta.file_id}_{conv_uid}"
                db_files = fs_client.list_files(
                    bucket,
                    filters={
                        "file_id": database_file_id,
                    },
                )
                if db_files:
                    logger.info("Database file exists in file storage. Downloading...")
                    fs_client.download_file(db_files[0].uri, database_file_path)
                    logger.info(f"Database file downloaded to {database_file_path}")

        return file_path, file_name, database_file_path, database_file_id

    @trace()
    async def generate_input_values(self) -> Dict:
        # 防止重复执行：如果已经生成过 input_values，直接返回缓存
        if (
            hasattr(self, "_cached_input_values")
            and self._cached_input_values is not None
        ):
            return self._cached_input_values

        # 确保 data_analysis_table 存在（特别是在有历史消息时，prepare()会被跳过）
        await self._ensure_data_analysis_table_exists()

        # ===== 新增：检测用户输入语言并动态选择 prompt =====
        user_input = self.current_user_input.last_text
        detected_language = detect_language(user_input)

        # 保存检测到的语言，供 _build_model_request 使用
        self._detected_language = detected_language

        # 动态导入 prompt 模板
        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        prompt_templates = get_prompt_templates_by_language(detected_language)

        table_schema = await blocking_func_to_async(
            self._executor, self.excel_reader.get_create_table_sql, self._curr_table
        )
        # table_summary = await blocking_func_to_async(
        #     self._executor, self.excel_reader.get_summary, self._curr_table
        # )
        colunms, datas = await blocking_func_to_async(
            self._executor, self.excel_reader.get_sample_data, self._curr_table
        )

        # 获取数据的时间范围（如果有日期列）
        data_time_range = await blocking_func_to_async(
            self._executor, self._get_data_time_range, self._curr_table
        )

        # === 新增：Query改写流程 ===
        query_rewrite_info = ""
        analysis_context = ""
        relevant_columns_info = ""  # 新增：相关列信息

        # 🔧 修复：如果select_param是JSON字符串，先解析为字典
        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                import json

                select_param_dict = json.loads(self.select_param)
                logger.info(f"成功将select_param从JSON字符串解析为字典")
            except Exception as e:
                logger.warning(f"⚠️ 解析select_param JSON失败: {e}")
                select_param_dict = None

        # 检查是否有缓存的data_schema_json
        if select_param_dict and isinstance(select_param_dict, dict):
            data_schema_json = select_param_dict.get("data_schema_json")

            if data_schema_json:
                logger.info(f"✅ 检测到data_schema_json，开始Query改写和列检索")
                try:
                    # 调用Query改写Agent
                    from dbgpt_app.scene.chat_data.chat_excel.query_rewrite import (
                        QueryRewriteAgent,
                    )

                    # 获取模型名称
                    model_name = self.llm_model

                    rewrite_agent = QueryRewriteAgent(self.llm_client, model_name)

                    # 获取历史对话（用于理解追问和指代）
                    chat_history = []
                    if hasattr(self, "history_messages") and self.history_messages:
                        # 按轮次组织历史消息，合并同一轮的多条助手消息
                        current_round_messages = []
                        last_role = None

                        for msg in self.history_messages:
                            if not hasattr(msg, "content"):
                                continue

                            role = getattr(msg, "role", "user")
                            content = msg.content

                            # 提取文本内容
                            if hasattr(content, "get_text"):
                                try:
                                    content = content.get_text()
                                except:
                                    content = str(content)
                            elif isinstance(content, list):
                                # 处理 MediaContent 列表
                                text_parts = []
                                for item in content:
                                    if hasattr(item, "object") and hasattr(
                                        item.object, "data"
                                    ):
                                        text_parts.append(str(item.object.data))
                                    else:
                                        text_parts.append(str(item))
                                content = " ".join(text_parts)
                            else:
                                content = str(content)

                            # 清理内容：移除数据结果，只保留 SQL 和引导文本
                            content = self._clean_history_content(content)

                            # 如果角色相同，合并内容；否则开始新的消息
                            if role == last_role and current_round_messages:
                                # 合并同角色的连续消息
                                current_round_messages[-1]["content"] += "\n" + content
                            else:
                                # 新角色，添加新消息
                                current_round_messages.append(
                                    {"role": role, "content": content}
                                )
                                last_role = role

                        chat_history = current_round_messages

                    logger.info(
                        f"📜 传递历史对话给Query改写Agent，共{len(chat_history)}条消息（已合并同轮消息）"
                    )

                    rewrite_result = await blocking_func_to_async(
                        self._executor,
                        rewrite_agent.rewrite_query,
                        self.current_user_input.last_text,
                        data_schema_json,
                        table_schema,
                        chat_history,  # 传入历史对话
                    )

                    # 构建分析上下文
                    if rewrite_result and rewrite_result.get("rewritten_query"):
                        query_rewrite_info = f"""


用户的问题：{rewrite_result['rewritten_query']}

相关字段：
{self._format_relevant_columns(rewrite_result.get('relevant_columns', []))}

分析建议：
{self._format_analysis_suggestions(rewrite_result.get('analysis_suggestions', []))}

分析逻辑：
{rewrite_result.get('analysis_logic', '')}

接下来请按照格式要求生成sql语句进行查询。
"""
                        # 保存改写结果供后续使用
                        self._query_rewrite_result = rewrite_result

                        logger.info(f"✅ Query改写成功")
                        logger.info(f"改写后问题: {rewrite_result['rewritten_query']}")

                        # 检查是否有提取到的领域知识
                        extracted_knowledge = rewrite_result.get("_extracted_knowledge")
                        if extracted_knowledge:
                            await self._save_domain_knowledge(
                                extracted_knowledge, data_schema_json
                            )

                        # === 新增：从改写结果中提取相关列的详细信息 ===
                        try:
                            # 解析data_schema_json
                            import json

                            schema_obj = (
                                json.loads(data_schema_json)
                                if isinstance(data_schema_json, str)
                                else data_schema_json
                            )
                            all_columns = schema_obj.get("columns", [])

                            # 从改写结果中获取相关列名
                            relevant_col_names = [
                                col.get("column_name", "")
                                for col in rewrite_result.get("relevant_columns", [])
                            ]

                            # 从schema中提取这些列的完整信息
                            relevant_columns_details = []
                            for col_name in relevant_col_names:
                                for col_info in all_columns:
                                    if col_info.get("column_name") == col_name:
                                        relevant_columns_details.append(col_info)
                                        break

                            # 格式化为prompt文本
                            if relevant_columns_details:
                                relevant_columns_info = (
                                    self._format_relevant_columns_for_prompt(
                                        relevant_columns_details
                                    )
                                )
                                logger.info(
                                    f"✅ 成功提取 {len(relevant_columns_details)} 个相关列的详细信息"
                                )
                                logger.info(
                                    f"相关列: {[col['column_name'] for col in relevant_columns_details]}"
                                )
                            else:
                                relevant_columns_info = "未找到相关列的详细信息。"
                                logger.warning(f"⚠️ 未能从schema中找到相关列的详细信息")

                        except Exception as col_err:
                            logger.warning(f"提取列详细信息失败: {col_err}")
                            relevant_columns_info = ""

                except Exception as e:
                    logger.warning(f"Query改写失败，使用原始问题: {e}")
                    self._query_rewrite_result = None
            else:
                logger.warning(f"⚠️ data_schema_json为空或不存在，跳过Query改写和列检索")
        else:
            logger.warning(f"⚠️ select_param不是字典或为空，跳过Query改写和列检索")

        # ===== 从 prompt.py 导入规则和示例块 =====
        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            _DUCKDB_RULES,
            _ANALYSIS_CONSTRAINTS_TEMPLATE,
            _EXAMPLES,
        )

        # 格式化约束条件（填入table_name）
        analysis_constraints = _ANALYSIS_CONSTRAINTS_TEMPLATE.format(
            table_name=self._curr_table, display_type=self._generate_numbered_list()
        )

        input_values = {
            "user_input": self.current_user_input.last_text,
            "table_name": self._curr_table,
            "display_type": self._generate_numbered_list(),
            # "table_summary": table_summary,
            "table_schema": table_schema,
            "data_example": json.dumps(
                datas, cls=EnhancedJSONEncoder, ensure_ascii=False
            ),
            "query_rewrite_info": query_rewrite_info,  # Query改写信息
            "relevant_columns_info": relevant_columns_info,  # 新增：相关列信息
            "data_time_range": data_time_range or "",  # 始终提供，避免KeyError
            # ===== 使用动态选择的规则、约束和示例块 =====
            "duckdb_syntax_rules": prompt_templates["duckdb_rules"],
            "analysis_constraints": analysis_constraints,
            "examples": prompt_templates["examples"],
        }

        # 🔧 缓存 input_values，避免重复执行 Query 改写
        self._cached_input_values = input_values

        return input_values

    async def _build_model_request(self) -> ModelRequest:
        """
        重写父类方法，动态替换 System Prompt 和 User Prompt Template 以支持多语言
        """
        # 获取检测到的语言（如果未检测则使用默认值 "zh"）
        detected_language = getattr(self, "_detected_language", "zh")

        # 动态获取对应语言的 prompt 模板
        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        prompt_templates = get_prompt_templates_by_language(detected_language)

        # 创建新的 ChatPromptTemplate，使用检测到的语言对应的模板
        dynamic_prompt = ChatPromptTemplate(
            messages=[
                SystemPromptTemplate.from_template(prompt_templates["system_prompt"]),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanPromptTemplate.from_template(
                    prompt_templates["user_prompt_template"]
                ),
            ]
        )

        # 临时替换 prompt_template 的 prompt
        original_prompt = self.prompt_template.prompt
        self.prompt_template.prompt = dynamic_prompt

        try:
            # 调用父类方法构建 ModelRequest
            model_request = await super()._build_model_request()
            return model_request
        finally:
            # 恢复原始 prompt（防止影响其他请求）
            self.prompt_template.prompt = original_prompt

    async def _save_domain_knowledge(self, knowledge: dict, current_schema_json: str):
        """
        保存领域知识到 excel_metadata.db 的 data_schema_json 中

        Args:
            knowledge: 提取到的知识，格式 {'column_name': '字段名', 'knowledge': '知识内容'}
            current_schema_json: 当前的 schema JSON 字符串
        """
        try:
            import json
            import sqlite3
            from datetime import datetime

            column_name = knowledge.get("column_name")
            knowledge_text = knowledge.get("knowledge")

            if not column_name or not knowledge_text:
                logger.warning("领域知识格式不完整，跳过保存")
                return

            # 解析当前 schema
            schema_obj = (
                json.loads(current_schema_json)
                if isinstance(current_schema_json, str)
                else current_schema_json
            )
            columns = schema_obj.get("columns", [])

            # 找到对应的字段并添加 domain_knowledge
            knowledge_saved = False
            for col in columns:
                if col.get("column_name") == column_name:
                    # 检查是否已有相同的知识
                    existing_knowledge = col.get("domain_knowledge", "")
                    if knowledge_text in existing_knowledge:
                        return

                    # 添加或追加知识
                    if existing_knowledge:
                        col["domain_knowledge"] = (
                            f"{existing_knowledge}\n• {knowledge_text}"
                        )
                    else:
                        col["domain_knowledge"] = knowledge_text

                    knowledge_saved = True
                    break

            if not knowledge_saved:
                logger.warning(f"未找到字段 {column_name}，无法保存知识")
                return

            # 更新到数据库
            # 获取数据库路径（与 ExcelAutoRegisterService 保持一致）
            # 从当前文件往上9层到达项目根目录，然后进入 packages/pilot/data/excel_cache
            current_file = Path(__file__)
            # chat.py -> excel_analyze -> chat_excel -> chat_data -> scene -> dbgpt_app -> src -> dbgpt-app -> packages -> 项目根目录
            project_root = (
                current_file.parent.parent.parent.parent.parent.parent.parent.parent.parent
            )
            cache_dir = project_root / "packages" / "pilot" / "data" / "excel_cache"
            meta_db_path = cache_dir / "excel_metadata.db"

            if not meta_db_path.exists():
                logger.warning(f"元数据数据库不存在: {meta_db_path}")
                return

            # 获取当前会话的 content_hash
            content_hash = getattr(self, "_content_hash", None)
            if not content_hash:
                logger.warning("无法获取 content_hash，无法更新数据库")
                return

            # 更新数据库
            conn = sqlite3.connect(str(meta_db_path))
            cursor = conn.cursor()

            updated_schema_json = json.dumps(schema_obj, ensure_ascii=False, indent=2)

            cursor.execute(
                """
                UPDATE excel_metadata
                SET data_schema_json = ?,
                    last_accessed = ?
                WHERE content_hash = ?
            """,
                (updated_schema_json, datetime.now().isoformat(), content_hash),
            )

            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()

            # 更新当前会话的 data_schema_json，使新知识立即生效
            if hasattr(self, "select_param"):
                if isinstance(self.select_param, dict):
                    self.select_param["data_schema_json"] = updated_schema_json
                elif isinstance(self.select_param, str):
                    try:
                        param_dict = json.loads(self.select_param)
                        param_dict["data_schema_json"] = updated_schema_json
                        self.select_param = json.dumps(param_dict, ensure_ascii=False)
                    except Exception as e:
                        logger.warning(f"更新 select_param 字符串失败: {e}")

        except Exception as e:
            logger.error(f"保存领域知识失败: {e}", exc_info=True)

    def _clean_history_content(self, content: str) -> str:
        """
        清理历史对话内容，移除数据结果，只保留 SQL 和引导文本

        Args:
            content: 原始内容

        Returns:
            清理后的内容
        """
        import re

        # 移除 <chart-view> 标签及其内容（包含大量数据）
        content = re.sub(
            r"<chart-view[^>]*>.*?</chart-view>", "", content, flags=re.DOTALL
        )

        # 移除过长的数据列表（通常是 JSON 数组）
        # 保留 <api-call> 中的 SQL，但移除执行结果
        lines = content.split("\n")
        cleaned_lines = []
        skip_data = False

        for line in lines:
            # 检测数据开始的标志
            if any(marker in line for marker in ["[{", '{"data":', '"rows":']):
                # 如果这行很长，可能是数据，跳过
                if len(line) > 200:
                    skip_data = True
                    continue

            # 检测数据结束
            if skip_data and ("}]" in line or line.strip() == "}"):
                skip_data = False
                continue

            if not skip_data:
                cleaned_lines.append(line)

        content = "\n".join(cleaned_lines)

        # 限制总长度（保留 SQL 和引导文本）
        if len(content) > 1000:
            # 尝试保留 <api-call> 部分
            api_call_match = re.search(r"<api-call>.*?</api-call>", content, re.DOTALL)
            if api_call_match:
                # 保留引导文本 + SQL
                before_api = content[: api_call_match.start()].strip()
                if len(before_api) > 200:
                    before_api = before_api[:200] + "..."
                content = before_api + "\n" + api_call_match.group(0)
            else:
                # 没有 api-call，直接截断
                content = content[:1000] + "..."

        return content.strip()

    def _format_relevant_columns(self, columns: List[Dict]) -> str:
        """格式化相关列信息"""
        if not columns:
            return "未指定"

        formatted = []
        for col in columns:
            col_name = col.get("column_name", "")
            usage = col.get("usage", "")
            formatted.append(f"  • {col_name}: {usage}")

        return "\n".join(formatted)

    def _format_analysis_suggestions(self, suggestions: List[str]) -> str:
        """格式化分析建议"""
        if not suggestions:
            return "无"

        formatted = []
        for i, suggestion in enumerate(suggestions, 1):
            formatted.append(f"  {i}. {suggestion}")

        return "\n".join(formatted)

    def _format_relevant_columns_for_prompt(self, columns: List[Dict]) -> str:
        """
        格式化相关列的详细信息，用于注入到prompt

        Args:
            columns: 列详细信息列表，每个元素是schema_json中的column对象

        Returns:
            格式化后的文本
        """
        if not columns:
            # 获取语言
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"
            return (
                "No relevant column information found."
                if is_english
                else "未找到相关列信息。"
            )

        # 获取语言并选择标题
        detected_language = getattr(self, "_detected_language", "zh")
        is_english = detected_language == "en"

        header = "Key fields to focus on:" if is_english else "你应该重点关注的字段为："
        formatted_parts = [header]

        for col in columns:
            col_name = col.get("column_name", "")
            data_type = col.get("data_type", "")
            description = col.get("description", "")
            semantic_type = col.get("semantic_type", "")
            analysis_usage = col.get("analysis_usage", [])
            domain_knowledge = col.get("domain_knowledge", "")

            col_text = f"  • {col_name}"
            if data_type:
                label = "Data type" if is_english else "数据类型"
                col_text += f"\n    {label}: {data_type}"
            if semantic_type:
                label = "Semantic type" if is_english else "语义类型"
                col_text += f"\n    {label}: {semantic_type}"
            if description:
                label = "Description" if is_english else "描述"
                col_text += f"\n    {label}: {description}"

            # 如果有领域知识，优先显示（放在描述后面）
            if domain_knowledge:
                label = "**Key Knowledge**" if is_english else "**关键知识**"
                col_text += f"\n    {label}: {domain_knowledge}"

            if analysis_usage:
                label = "Analysis usage" if is_english else "分析用途"
                col_text += f"\n    {label}: {', '.join(analysis_usage)}"

            # 如果有统计信息，也添加进来
            if "statistics_summary" in col:
                label = "Statistics" if is_english else "统计信息"
                col_text += f"\n    {label}: {col['statistics_summary']}"

            # 如果有唯一值，也添加进来（限制显示数量）
            if "unique_values_top5" in col:
                unique_vals = col["unique_values_top5"]
                label = "Possible values" if is_english else "可选值"
                label_partial = (
                    "Possible values (partial)" if is_english else "可选值(部分)"
                )
                if len(unique_vals) <= 10:
                    col_text += f"\n    {label}: {', '.join(map(str, unique_vals))}"
                else:
                    col_text += f"\n    {label_partial}: {', '.join(map(str, unique_vals[:10]))}..."

            formatted_parts.append(col_text)

        return "\n\n".join(formatted_parts)

    def _get_data_time_range(self, table_name: str) -> str:
        """
        获取数据的时间范围，帮助LLM理解可用的数据周期
        """
        try:
            # 查找可能的日期列
            columns_result = self.excel_reader.db.sql(
                f"DESCRIBE {table_name}"
            ).fetchall()
            date_columns = []

            for col_info in columns_result:
                col_name = col_info[0]
                col_type = col_info[1].upper()
                if "DATE" in col_type or "TIME" in col_type:
                    date_columns.append(col_name)

            if not date_columns:
                return ""

            # 对第一个日期列获取时间范围
            date_col = date_columns[0]
            query = f'SELECT MIN("{date_col}") as min_date, MAX("{date_col}") as max_date FROM "{table_name}"'
            result = self.excel_reader.db.sql(query).fetchone()

            if result and result[0] and result[1]:
                min_date = result[0]
                max_date = result[1]

                # 格式化日期
                if isinstance(min_date, str):
                    min_date = min_date[:10]  # 取前10个字符 YYYY-MM-DD
                else:
                    min_date = str(min_date)[:10]

                if isinstance(max_date, str):
                    max_date = max_date[:10]
                else:
                    max_date = str(max_date)[:10]

                time_range = f"\n\n数据时间范围：{min_date} 至 {max_date}"
                time_range += (
                    f"\n（注意：进行同比分析时，请确保SQL查询包含足够的历史数据）"
                )
                return time_range

        except Exception as e:
            logger.warning(f"获取数据时间范围失败: {e}")

        return ""

    async def prepare(self):
        logger.info(f"{self.chat_mode} prepare start!")
        if self.has_history_messages():
            return None

        # 检查是否有缓存的 summary_prompt（跳过 LLM 生成）
        # ✅ 修复：解析select_param字符串
        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                import json

                select_param_dict = json.loads(self.select_param)
            except:
                select_param_dict = {}

        if select_param_dict and isinstance(select_param_dict, dict):
            summary_prompt = select_param_dict.get("summary_prompt")

            if summary_prompt and isinstance(summary_prompt, str):
                logger.info(f"✅ 检测到缓存的 Data Summary，跳过 LLM 生成")
                # 检查是否使用DuckDB缓存（表名不是默认的 data_analysis_table）
                if self._curr_table != "data_analysis_table":
                    # 使用DuckDB缓存，表已存在，无需创建
                    logger.info(f"✅ 使用DuckDB缓存，表 {self._curr_table} 已存在，跳过创建")
                else:
                    # 传统方式，需要创建 data_analysis_table
                    try:
                        await blocking_func_to_async(
                            self._executor, self._create_simple_data_analysis_table
                        )
                        logger.info(f"✅ 使用简化方式创建了 data_analysis_table")
                    except Exception as e:
                        logger.warning(f"使用缓存创建表失败: {e}, 将重新生成")
                        # 继续执行后续的LLM生成流程
                        pass

                # 生成并保存 Excel 基本信息（即使使用缓存）
                await self._generate_and_save_excel_info(None)

                # 生成包含 Excel 基本信息的展示消息
                excel_info_message = await self._format_excel_info_message()

                # 如果有 Excel 基本信息，返回展示消息
                if excel_info_message:
                    return ModelOutput(
                        error_code=0, text=excel_info_message, finish_reason="stop"
                    )

                # 返回简化消息
                return ModelOutput(
                    error_code=0,
                    text="数据分析结构已加载（使用缓存）",
                    finish_reason="stop",
                )

        # 如果没有缓存，则调用 LLM 生成
        logger.info(f"⚠️ 未检测到缓存，将调用 LLM 生成 Data Summary")
        chat_param = ChatParam(
            chat_session_id=self.chat_session_id,
            current_user_input="["
            + self.excel_reader.excel_file_name
            + "]"
            + " Analyze！",
            select_param=self.select_param,
            chat_mode=ChatScene.ExcelLearning,
            model_name=self.model_name,
            user_name=self.chat_param.user_name,
            sys_code=self.chat_param.sys_code,
        )
        if self._chat_param.temperature is not None:
            chat_param.temperature = self._chat_param.temperature
        if self._chat_param.max_new_tokens is not None:
            chat_param.max_new_tokens = self._chat_param.max_new_tokens
        learn_chat = ExcelLearning(
            chat_param,
            system_app=self.system_app,
            parent_mode=self.chat_mode,
            excel_reader=self.excel_reader,
        )
        result = await learn_chat.nostream_call()

        if (
            os.path.exists(self._database_file_path)
            and self._database_file_id is not None
        ):
            await blocking_func_to_async(self._executor, self.excel_reader.close)
            await blocking_func_to_async(
                self._executor,
                self.fs_client.upload_file,
                self._bucket,
                self._database_file_path,
                file_id=self._database_file_id,
            )

        # 生成并保存 Excel 基本信息
        await self._generate_and_save_excel_info(result)

        # 生成包含 Excel 基本信息的展示消息
        excel_info_message = await self._format_excel_info_message()

        # 如果有 Excel 基本信息，修改返回消息
        if excel_info_message:
            return ModelOutput(
                error_code=0, text=excel_info_message, finish_reason="stop"
            )

        return result

    def _create_simple_data_analysis_table(self):
        """创建简化版的 data_analysis_table（从现有表复制）"""
        try:
            # 检查 data_analysis_table 是否已存在
            tables = self.excel_reader.db.sql("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            if self._curr_table in table_names:
                logger.info(f"✅ {self._curr_table} 已存在，跳过创建")
                return

            # 优先使用 excel_reader 的实际表名（DuckDB 缓存场景）
            source_table = None
            if hasattr(self.excel_reader, 'table_name') and self.excel_reader.table_name:
                source_table = self.excel_reader.table_name
                if source_table in table_names and source_table != self._curr_table:
                    logger.info(f"使用实际表名 {source_table} 创建 {self._curr_table}")
                    sql = f"CREATE TABLE {self._curr_table} AS SELECT * FROM {source_table};"
                    self.excel_reader.db.sql(sql)
                    logger.info(f"✅ Created {self._curr_table} from {source_table}")
                    return

            # 如果没有实际表名，尝试使用 temp_table（传统场景）
            if "temp_table" in table_names:
                logger.info(f"使用 temp_table 创建 {self._curr_table}")
                sql = f"CREATE TABLE {self._curr_table} AS SELECT * FROM temp_table;"
                self.excel_reader.db.sql(sql)
                logger.info(f"✅ Created {self._curr_table} from temp_table")
                return

            # 如果都没有，报错
            logger.error(f"⚠️ 找不到源表：temp_table 不存在，且 excel_reader.table_name ({getattr(self.excel_reader, 'table_name', None)}) 也不存在")
            raise ValueError(f"找不到可用的源表来创建 {self._curr_table}")
        except Exception as e:
            logger.error(f"Failed to create {self._curr_table}: {e}")
            raise

    async def _ensure_data_analysis_table_exists(self):
        """确保 data_analysis_table 存在（在 generate_input_values 之前调用）"""
        try:
            tables = await blocking_func_to_async(
                self._executor,
                lambda: self.excel_reader.db.sql("SHOW TABLES").fetchall(),
            )
            table_names = [t[0] for t in tables]

            if self._curr_table not in table_names:
                logger.info(f"⚠️ {self._curr_table} 不存在，尝试创建...")
                await blocking_func_to_async(
                    self._executor, self._create_simple_data_analysis_table
                )
            else:
                logger.info(f"✅ {self._curr_table} 已存在")
        except Exception as e:
            logger.error(f"确保表存在时失败: {e}")
            raise

    async def _generate_and_save_excel_info(self, learning_result: ModelOutput = None):
        """生成并保存 Excel 基本信息到数据库"""
        try:
            # 获取前十行数据
            columns, top_10_rows = await blocking_func_to_async(
                self._executor,
                self.excel_reader.get_sample_data,
                self._curr_table,
                10,  # 获取前10行
            )

            # 获取行列数
            row_count, column_count = await blocking_func_to_async(
                self._executor, self._get_table_stats, self._curr_table
            )

            # 获取数据描述（从 learning_result 或从缓存）
            data_description = None
            data_schema_json = None

            if learning_result and learning_result.has_text:
                # 从 learning_result 中提取数据描述
                data_description = learning_result.text
            else:
                # 尝试从 select_param 中获取缓存的描述
                select_param_dict = self.select_param
                if isinstance(self.select_param, str):
                    try:
                        select_param_dict = json.loads(self.select_param)
                    except:
                        select_param_dict = {}

                if isinstance(select_param_dict, dict):
                    data_description = select_param_dict.get("summary_prompt")
                    data_schema_json = select_param_dict.get("data_schema_json")

            # 生成推荐问题：优先从 data_schema_json 中提取
            suggested_questions = []
            if data_schema_json:
                try:
                    schema = json.loads(data_schema_json)
                    suggested_questions = schema.get("suggested_questions", [])
                    if suggested_questions:
                        logger.info(
                            f"✅ 从 data_schema_json 中提取了 {len(suggested_questions)} 个推荐问题"
                        )
                except Exception as e:
                    logger.warning(f"从 data_schema_json 提取推荐问题失败: {e}")

            # 如果没有提取到推荐问题，使用默认问题
            if not suggested_questions:
                logger.info("data_schema_json 中没有推荐问题，使用默认问题")
                suggested_questions = self._get_default_suggested_questions()

            # 保存到数据库
            schema_dao = ExcelSchemaDao()
            await blocking_func_to_async(
                self._executor,
                schema_dao.save_or_update,
                conv_uid=self.chat_session_id,
                file_name=self._file_name,
                table_name=self._curr_table,
                row_count=row_count,
                column_count=column_count,
                top_10_rows=top_10_rows,
                data_description=data_description,
                data_schema_json=data_schema_json,
                suggested_questions=suggested_questions,
                file_path=self._database_file_path,
                db_path=self._database_file_path,
                user_id=(
                    self.chat_param.user_id
                    if hasattr(self.chat_param, "user_id")
                    else None
                ),
                user_name=self.chat_param.user_name,
                sys_code=self.chat_param.sys_code,
            )

            logger.info(f"✅ Excel 基本信息已保存到数据库")
        except Exception as e:
            logger.error(f"生成并保存 Excel 基本信息失败: {e}")
            import traceback

            traceback.print_exc()

    def _get_table_stats(self, table_name: str) -> tuple:
        """获取表的行列数"""
        try:
            # 获取行数
            row_result = self.excel_reader.db.sql(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            row_count = row_result[0] if row_result else 0

            # 获取列数
            columns, _ = self.excel_reader.get_columns(table_name)
            column_count = len(columns) if columns else 0

            return row_count, column_count
        except Exception as e:
            logger.error(f"获取表统计信息失败: {e}")
            return 0, 0

    def _get_default_suggested_questions(self) -> List[str]:
        """生成默认推荐问题（当 data_schema_json 中没有推荐问题时使用）"""
        return [
            "查看数据的基本统计信息",
            "分析数据的分布情况",
            "找出数据中的异常值",
            "分析数据的趋势变化",
            "对比不同维度的数据",
        ]

    async def _format_excel_info_message(self) -> str:
        """格式化 Excel 基本信息展示消息"""
        try:
            schema_dao = ExcelSchemaDao()
            schema_entity = await blocking_func_to_async(
                self._executor, schema_dao.get_by_conv_uid, self.chat_session_id
            )

            if not schema_entity:
                return None

            schema_dict = schema_dao.to_dict(schema_entity)

            # 构建展示消息
            message_parts = []
            message_parts.append("## 📊 Excel 数据基本信息\n\n")

            # 基本信息
            message_parts.append(f"**文件名**: {schema_dict['file_name']}\n")
            message_parts.append(
                f"**数据规模**: {schema_dict['row_count']} 行 × {schema_dict['column_count']} 列\n\n"
            )

            # 前十行数据
            if schema_dict.get("top_10_rows"):
                message_parts.append("### 📋 数据预览（前10行）\n\n")
                message_parts.append("```\n")
                # 格式化表格显示
                top_rows = schema_dict["top_10_rows"]
                if top_rows and len(top_rows) > 0:
                    # 显示表头
                    if len(top_rows) > 0:
                        headers = [str(item) for item in top_rows[0]]
                        message_parts.append(" | ".join(headers[:10]))  # 最多显示10列
                        message_parts.append("\n")
                        message_parts.append(
                            " | ".join(["---"] * min(len(headers), 10))
                        )
                        message_parts.append("\n")
                        # 显示数据行
                        for row in top_rows[1:11]:  # 最多显示10行
                            row_data = [
                                str(item)[:30] for item in row[:10]
                            ]  # 每列最多30字符
                            message_parts.append(" | ".join(row_data))
                            message_parts.append("\n")
                message_parts.append("```\n\n")

            # 数据描述
            if schema_dict.get("data_description"):
                message_parts.append("### 📝 数据描述\n\n")
                message_parts.append(
                    f"{schema_dict['data_description'][:500]}...\n\n"
                )  # 最多显示500字符

            # 推荐问题
            if schema_dict.get("suggested_questions"):
                message_parts.append("### 💡 推荐问题\n\n")
                for i, question in enumerate(schema_dict["suggested_questions"][:8], 1):
                    message_parts.append(f"{i}. {question}\n")
                message_parts.append("\n")

            message_parts.append("---\n")
            message_parts.append(
                "💬 您可以基于以上信息开始数据分析，或直接提出您的问题。\n"
            )

            return "".join(message_parts)
        except Exception as e:
            logger.error(f"格式化 Excel 基本信息消息失败: {e}")
            return None

    def stream_plugin_call(self, text):
        with root_tracer.start_span(
            "ChatExcel.stream_plugin_call.run_display_sql", metadata={"text": text}
        ):
            result = self.api_call.display_sql_llmvis(
                text,
                self.excel_reader.get_df_by_sql_ex,
            )
            return result


    async def stream_call(self, text_output: bool = True, incremental: bool = False):
        """
        重写stream_call方法，分阶段流式输出中间结果，提升用户体验

        输出阶段：
        1. Query改写结果（如果有）
        2. SQL生成结果
        3. 最终总结和图表

        支持SQL错误自动修复：如果SQL执行失败，会自动重试一次
        """
        input_values = await self.generate_input_values()

        # ===== 阶段1：输出Query改写结果 =====
        if self._query_rewrite_result:
            thinking_stage1 = self._format_query_rewrite_thinking(
                self._query_rewrite_result
            )
            if thinking_stage1:
                # 包装成 vis-thinking 格式
                from dbgpt.vis.tags.vis_thinking import VisThinking

                vis_thinking_output = VisThinking().sync_display(
                    content=thinking_stage1
                )
                if text_output:
                    yield vis_thinking_output
                else:
                    # 直接作为 text 输出，前端会识别 vis-thinking 格式
                    stage1_output = ModelOutput.build(
                        text=vis_thinking_output, error_code=0, finish_reason="continue"
                    )
                    yield stage1_output

        # 调用父类的_build_model_request获取payload
        # 注意：这里会再次调用 generate_input_values，但由于已经执行过，会很快
        payload = await self._build_model_request()
        logger.info(f"payload request: \n{payload}")

        # 初始化错误跟踪
        self._last_sql_error = None

        # 使用非流式调用，直接获取完整结果（避免流式输出日志）
        full_output = await self.call_llm_operator(payload)

        # ===== 阶段2：执行SQL并生成最终总结 =====
        if full_output:
            try:
                ai_response_text, view_message = await self._handle_final_output(
                    full_output, incremental=incremental
                )

                # 阶段3：输出最终结果（追加在之前的思考过程后面）
                # 构建完整的输出：思考过程 + 最终结果
                final_output_parts = []

                # 只追加阶段1的思考过程（问题理解与分析）
                if self._query_rewrite_result:
                    thinking_stage1 = self._format_query_rewrite_thinking(
                        self._query_rewrite_result
                    )
                    if thinking_stage1:
                        from dbgpt.vis.tags.vis_thinking import VisThinking

                        vis_thinking_output = VisThinking().sync_display(
                            content=thinking_stage1
                        )
                        final_output_parts.append(vis_thinking_output)

                # 追加最终结果
                final_output_parts.append(view_message)

                # 合并所有部分
                complete_output = "\n\n".join(final_output_parts)

                if text_output:
                    yield complete_output
                else:
                    yield ModelOutput.build(
                        complete_output,
                        "",
                        error_code=full_output.error_code if full_output else 0,
                        finish_reason=(
                            full_output.finish_reason if full_output else "stop"
                        ),
                    )
            except Exception as e:
                logger.error(f"处理输出时出错: {e}")
                # SQL执行失败，返回错误信息
                error_msg = f"数据查询失败：{str(e)}"
                if text_output:
                    yield error_msg
                else:
                    yield ModelOutput.build(
                        error_msg,
                        "",
                        error_code=1,
                        finish_reason="error",
                    )
        else:
            # full_output为None，返回错误
            error_msg = "生成SQL失败，请重试"
            if text_output:
                yield error_msg
            else:
                yield ModelOutput.build(
                    error_msg,
                    "",
                    error_code=1,
                    finish_reason="error",
                )

    async def _handle_final_output(
        self,
        final_output: ModelOutput,
        incremental: bool = False,
        check_error: bool = True,
    ):
        text_msg = final_output.text if final_output.has_text else ""
        view_msg = self.stream_plugin_call(text_msg)

        # ⚠️ 关键修改：先检查SQL错误，再决定是否生成总结
        # 如果有SQL错误，_last_sql_error会在stream_plugin_call中被设置
        # check_error=False时跳过检查(用于重试后的执行)
        if check_error and self._last_sql_error:
            # 有错误，不生成总结，直接返回（让上层处理重试）
            logger.warning(f"SQL执行失败，跳过总结生成: {self._last_sql_error[:100]}")
            view_msg = final_output.gen_text_with_thinking(new_text=view_msg)
            # ⚠️ 合并 thinking 和 text 作为完整的 AI 回答
            ai_full_response = self._combine_thinking_and_text(final_output, view_msg)
            return ai_full_response, view_msg

        # 没有错误，尝试生成自然语言总结
        summary_text = await self._generate_result_summary(text_msg, view_msg)

        # 如果成功生成总结，替换掉原来的引导性文本
        if summary_text:
            # 从view_msg中提取所有的chart-view
            import re

            chart_pattern = r"(<chart-view.*?</chart-view>)"
            chart_matches = re.findall(chart_pattern, view_msg, re.DOTALL)

            if chart_matches:
                # 保留所有chart-view，用换行分隔
                all_chart_views = "\n\n".join(chart_matches)
                # 用总结替换掉原来的引导性文本
                view_msg = f"{summary_text}\n\n{all_chart_views}"
            else:
                # 如果没有找到chart-view，就在开头添加总结
                view_msg = f"{summary_text}\n\n{view_msg}"

        view_msg = final_output.gen_text_with_thinking(new_text=view_msg)
        # ⚠️ 合并 thinking 和 text 作为完整的 AI 回答
        ai_full_response = self._combine_thinking_and_text(final_output, view_msg)
        return ai_full_response, view_msg

    def _combine_thinking_and_text(
        self, final_output: ModelOutput, view_msg: str
    ) -> str:
        """
        合并 thinking 和 text 部分，作为完整的 AI 回答

        Args:
            final_output: LLM 输出
            view_msg: 处理后的视图消息（已包含 thinking + text）

        Returns:
            完整的 AI 回答（thinking + text）
        """
        # 安全地获取 thinking 属性（有些模型可能没有这个属性）
        thinking_text = getattr(final_output, "thinking", None)

        # 如果有 thinking，拼接 thinking 和 text
        if thinking_text:
            text_content = final_output.text if final_output.has_text else ""
            # 用换行分隔 thinking 和 text
            return f"{thinking_text}\n{text_content}".strip()
        else:
            # 没有 thinking，直接返回 text
            return final_output.text if final_output.has_text else ""

    def _format_query_rewrite_thinking(self, rewrite_result: dict) -> str:
        """
        格式化Query改写结果为thinking格式，用于流式输出

        Args:
            rewrite_result: Query改写结果

        Returns:
            格式化后的thinking文本
        """
        try:
            if not rewrite_result:
                return ""

            # 获取检测到的语言（默认为中文）
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"

            # 根据语言选择标题和标签
            if is_english:
                title = "Question Understanding & Analysis\n\n"
                label_question = "1. Understood Question: "
                label_columns = "\n2. Relevant Fields:\n"
                label_suggestions = "\n3. Analysis Approach:\n"
                separator = ": "
            else:
                title = "问题理解与分析\n\n"
                label_question = "1.理解的问题："
                label_columns = "\n2.需要关注的字段：\n"
                label_suggestions = "\n3.分析思路：\n"
                separator = "："

            thinking_parts = []
            thinking_parts.append(title)

            # 改写后的问题
            rewritten_query = rewrite_result.get("rewritten_query", "")
            if rewritten_query:
                thinking_parts.append(f"{label_question}{rewritten_query}\n")

            # 相关字段
            relevant_columns = rewrite_result.get("relevant_columns", [])
            if relevant_columns:
                thinking_parts.append(label_columns)
                for col in relevant_columns[:5]:  # 最多显示5个
                    col_name = col.get("column_name", "")
                    usage = col.get("usage", "")
                    if col_name:
                        thinking_parts.append(f"  • {col_name}")
                        if usage:
                            thinking_parts.append(f"{separator}{usage}")
                        thinking_parts.append("\n")

            # 分析建议
            analysis_suggestions = rewrite_result.get("analysis_suggestions", [])
            if analysis_suggestions:
                thinking_parts.append(label_suggestions)
                for i, suggestion in enumerate(
                    analysis_suggestions[:5], 1
                ):  # 最多显示5条
                    thinking_parts.append(f"  • {suggestion}\n")

            return "".join(thinking_parts)

        except Exception as e:
            logger.warning(f"格式化Query改写thinking失败: {e}")
            return ""

    async def _generate_result_summary(self, original_text: str, view_msg: str) -> str:
        """
        根据SQL执行结果生成自然语言总结

        Args:
            original_text: LLM生成的原始文本（包含引导性文本）
            view_msg: 包含查询结果的完整消息

        Returns:
            自然语言总结，如果生成失败则返回空字符串
        """
        try:
            import re
            import json

            # 从view_msg中提取所有的chart-view内容
            chart_pattern = r'<chart-view content="([^"]+)">'
            matches = re.findall(chart_pattern, view_msg)

            if not matches:
                logger.info("未找到chart-view内容，跳过总结生成")
                return ""

            # 收集所有SQL和查询结果
            all_sql_results = []
            import html

            for match_str in matches:
                # 解码HTML实体
                content_str = html.unescape(match_str)
                content_data = json.loads(content_str)

                # 获取SQL和查询结果
                sql = content_data.get("sql", "").strip()
                query_data = content_data.get("data", [])

                if query_data:
                    all_sql_results.append({"sql": sql, "result": query_data})

            if not all_sql_results:
                logger.info("所有查询结果为空，跳过总结生成")
                return ""

            # 构建总结提示词，包含历史对话、所有SQL和结果

            # 1. 构建历史对话上下文（清理后的版本）
            history_context = ""
            if self.history_messages and len(self.history_messages) > 0:
                history_context = "\n=== 历史对话 ===\n"
                for msg in self.history_messages[-6:]:  # 只取最近3轮（6条消息）
                    if not hasattr(msg, "content"):
                        continue

                    role = getattr(msg, "role", "user")
                    content = msg.content

                    # 提取文本内容
                    if hasattr(content, "get_text"):
                        try:
                            content = content.get_text()
                        except:
                            content = str(content)
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if hasattr(item, "object") and hasattr(item.object, "data"):
                                text_parts.append(str(item.object.data))
                            else:
                                text_parts.append(str(item))
                        content = " ".join(text_parts)
                    else:
                        content = str(content)

                    # 清理内容：移除数据结果
                    content = self._clean_history_content(content)

                    # 获取语言并格式化角色显示
                    detected_language = getattr(self, "_detected_language", "zh")
                    is_english = detected_language == "en"
                    role_display = (
                        "User"
                        if role == "human"
                        else (
                            "Assistant"
                            if is_english
                            else "用户" if role == "human" else "助手"
                        )
                    )
                    history_context += f"{role_display}: {content}\n\n"

            # 获取检测到的语言
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"

            # 2. 构建SQL执行结果（多语言）
            sql_results_text = ""
            for i, sql_result in enumerate(all_sql_results, 1):
                sql_label = f"Executed SQL {i}" if is_english else f"执行的SQL {i}"
                result_label = f"Query Result {i}" if is_english else f"查询结果 {i}"
                sql_results_text += f"\n{sql_label}：\n{sql_result['sql']}\n\n"
                sql_results_text += f"{result_label}：\n{json.dumps(sql_result['result'], ensure_ascii=False, indent=2)}\n"

            # 3. 构建完整的总结提示词（多语言）
            if is_english:
                summary_prompt = f"""{history_context}
=== User's Current Question ===
{self.current_user_input.last_text}
{sql_results_text}
**IMPORTANT - Language Requirement**:
- The user's question is in ENGLISH
- You MUST respond in ENGLISH
- Your answer MUST be in ENGLISH, not Chinese
- Based on the conversation history and all the SQL query results above, summarize and answer the user's current question in one sentence in ENGLISH.
- If the current question is a follow-up or continuation of previous topics, reflect continuity and contextual relationship in your summary.
- Use ENGLISH language style consistent with the user's question.

Answer:"""
            else:
                summary_prompt = f"""{history_context}
=== 用户当前问题 ===
{self.current_user_input.last_text}
{sql_results_text}
**重要 - 语言要求**：
- 用户的问题是**中文**
- 你必须用**中文**回答
- 请根据历史对话和上述所有SQL查询结果，用一句话总结并完整回答用户的当前问题。
- 如果当前问题是追问或延续之前的话题，请在总结中体现出连贯性和上下文关系。
- 语言风格和用户问题一致，使用**中文**。

回答："""

            # 构建ModelRequest - 简化版本，只包含必要参数
            summary_request = ModelRequest(
                model=self.llm_model,
                messages=[
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN, content=summary_prompt
                    )
                ],
                temperature=0.3,
                max_new_tokens=500,  # 增加token数量，因为可能需要总结多个结果
            )

            # 使用llm_client生成总结
            if self.llm_client:
                summary_output = await self.llm_client.generate(summary_request)
                if summary_output and summary_output.text:
                    summary_text = summary_output.text.strip()
                    logger.info(f"生成结果总结: {summary_text}")
                    return summary_text
            else:
                logger.warning("llm_client未初始化，无法生成总结")

            return ""

        except Exception as e:
            logger.warning(f"生成结果总结失败: {e}", exc_info=True)
            return ""
