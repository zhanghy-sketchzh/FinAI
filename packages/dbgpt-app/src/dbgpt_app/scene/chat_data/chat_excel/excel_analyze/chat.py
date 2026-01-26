# ruff: noqa: E501
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Type, Union

from dbgpt import SystemApp
from dbgpt.agent.util.api_call import ApiCall
from dbgpt.configs.model_config import DATA_DIR
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    ModelMessage,
    ModelMessageRoleType,
    ModelOutput,
    ModelRequest,
    ModelRequestContext,
    SystemPromptTemplate,
)
from dbgpt.core.interface.file import _SCHEMA, FileStorageClient
from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.util.json_utils import EnhancedJSONEncoder
from dbgpt.util.tracer import root_tracer, trace
from dbgpt_app.scene import BaseChat, ChatScene
from dbgpt_app.scene.base_chat import ChatParam
from dbgpt_app.scene.chat_data.chat_excel.config import ChatExcelConfig
from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.language_detector import (
    detect_language,
)
from dbgpt_app.scene.chat_data.chat_excel.excel_learning.chat import ExcelLearning
from dbgpt_app.scene.chat_data.chat_excel.excel_reader import ExcelReader
from dbgpt_app.scene.chat_data.chat_excel.excel_schema_db import ExcelSchemaDao

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

        use_existing_db = False
        duckdb_path = None
        duckdb_table_name = None

        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                select_param_dict = json.loads(self.select_param)
            except json.JSONDecodeError as e:
                logger.error(f"解析select_param失败: {e}")
                select_param_dict = {}

        if isinstance(select_param_dict, dict):
            duckdb_path = select_param_dict.get("db_path")
            duckdb_table_name = select_param_dict.get("table_name")
            self._content_hash = select_param_dict.get("content_hash")

            if self._content_hash:
                try:
                    import sqlite3

                    current_file = Path(__file__)
                    project_root = current_file
                    for _ in range(9):
                        project_root = project_root.parent
                    meta_db_path = (
                        project_root
                        / "packages"
                        / "pilot"
                        / "data"
                        / "excel_cache"
                        / "excel_metadata.db"
                    )

                    if meta_db_path.exists():
                        conn = sqlite3.connect(str(meta_db_path))
                        cursor = conn.cursor()
                        
                        # 先尝试从多表元数据表查询（excel_tables_metadata）
                        try:
                            cursor.execute(
                                """SELECT table_name, data_schema_json, create_table_sql, row_count, column_count, sheet_name
                                   FROM excel_tables_metadata WHERE file_hash = ?""",
                                (self._content_hash,),
                            )
                            multi_table_results = cursor.fetchall()
                        except Exception:
                            multi_table_results = []
                        
                        if multi_table_results and len(multi_table_results) > 1:
                            # 多表模式：从 excel_tables_metadata 加载
                            all_tables_info = []
                            for row in multi_table_results:
                                table_info = {
                                    "table_name": row[0],
                                    "data_schema_json": row[1],
                                    "create_table_sql": row[2],
                                    "row_count": row[3],
                                    "column_count": row[4],
                                    "sheet_name": row[5] if len(row) > 5 else row[0],
                                }
                                all_tables_info.append(table_info)
                            select_param_dict["all_tables_info"] = all_tables_info
                            # 使用第一个表的 schema 作为默认
                            if multi_table_results[0][1]:
                                select_param_dict["data_schema_json"] = multi_table_results[0][1]
                            logger.info(f"多表模式：从 excel_tables_metadata 加载了 {len(all_tables_info)} 个表的信息")
                        else:
                            # 单表模式：从 excel_metadata 查询
                            cursor.execute(
                                "SELECT data_schema_json FROM excel_metadata WHERE content_hash = ?",
                                (self._content_hash,),
                            )
                            result = cursor.fetchone()
                            if result and result[0]:
                                select_param_dict["data_schema_json"] = result[0]
                        
                        conn.close()
                        
                        self.select_param = (
                            json.dumps(select_param_dict, ensure_ascii=False)
                            if isinstance(self.select_param, str)
                            else select_param_dict
                        )
                except Exception as e:
                    logger.warning(f"从数据库重新加载 data_schema_json 失败: {e}")

            if duckdb_path and os.path.exists(duckdb_path):
                use_existing_db = True
                logger.info(f"使用DuckDB缓存: {duckdb_path}, 表名: {duckdb_table_name}")
            elif duckdb_path:
                logger.warning(f"db_path存在但文件不存在: {duckdb_path}")
        else:
            logger.warning("select_param不是字典类型，使用传统Excel导入")

        file_path, file_name, database_file_path, database_file_id = self._resolve_path(
            select_param_dict,
            chat_param.chat_session_id,
            self.fs_client,
            self._bucket,
            duckdb_path=duckdb_path,
        )

        use_cache_success = False
        if use_existing_db and duckdb_path:
            try:
                self.excel_reader = self._create_reader_from_duckdb(
                    chat_param.chat_session_id,
                    duckdb_path,
                    file_name,
                    duckdb_table_name,
                )
                # 使用 excel_reader 中实际获取到的表名（可能是从数据库查询得到的）
                table_name_from_reader = self.excel_reader.table_name
                # 检查是否为多表模式（表名包含逗号）
                if "," in table_name_from_reader:
                    table_list = [t.strip() for t in table_name_from_reader.split(",")]
                    self._all_table_names = table_list
                    self._is_multi_table_mode = True
                    self._curr_table = table_list[0]  # 使用第一个表名
                else:
                    self._all_table_names = [table_name_from_reader]
                    self._is_multi_table_mode = False
                    self._curr_table = table_name_from_reader
                use_cache_success = True
                # 确保 database_file_path 与实际使用的 duckdb_path 一致
                database_file_path = duckdb_path
                if self._is_multi_table_mode:
                    logger.info(f"成功使用DuckDB缓存，多表模式，表: {self._all_table_names}")
                else:
                    logger.info(f"成功使用DuckDB缓存，表名: {self._curr_table}")
            except Exception as e:
                logger.warning(f"使用DuckDB缓存失败，回退到重新导入: {e}")
                # 删除损坏的缓存文件
                try:
                    if os.path.exists(duckdb_path):
                        os.remove(duckdb_path)
                        logger.info(f"已删除损坏的缓存文件: {duckdb_path}")
                except Exception as del_e:
                    logger.warning(f"删除缓存文件失败: {del_e}")
                use_cache_success = False

        if not use_cache_success:
            # 如果有 duckdb_path 但缓存加载失败，使用原来的 duckdb_path
            actual_db_path = duckdb_path if duckdb_path else database_file_path
            
            # 检查数据库中是否已有表（可能是 excel_auto_register 创建的）
            existing_table_names = []
            if actual_db_path and os.path.exists(actual_db_path):
                try:
                    import duckdb as duckdb_check
                    check_conn = duckdb_check.connect(database=actual_db_path, read_only=True)
                    query = (
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'main'"
                    )
                    tables_result = check_conn.execute(query).fetchall()
                    check_conn.close()
                    if tables_result:
                        existing_table_names = [t[0] for t in tables_result]
                        logger.info(f"发现已存在的表: {existing_table_names}")
                except Exception as check_e:
                    logger.warning(f"检查已存在表失败: {check_e}")
            
            if existing_table_names:
                # 使用已存在的表，不需要重新导入
                # 多表模式：保存所有表名，_curr_table 使用第一个表名
                self._all_table_names = existing_table_names
                self._is_multi_table_mode = len(existing_table_names) > 1
                self._curr_table = existing_table_names[0]
                self.excel_reader = self._create_reader_from_duckdb(
                    chat_param.chat_session_id,
                    actual_db_path,
                    file_name,
                    self._curr_table,  # 使用第一个表名初始化
                )
                if self._is_multi_table_mode:
                    logger.info(f"使用多表模式，表: {existing_table_names}")
                else:
                    logger.info(f"使用已存在的DuckDB表: {self._curr_table}")
            else:
                # 没有已存在的表，需要重新创建
                self._all_table_names = ["data_analysis_table"]
                self._is_multi_table_mode = False
                self._curr_table = "data_analysis_table"
                self.excel_reader = ExcelReader(
                    chat_param.chat_session_id,
                    file_path,
                    file_name,
                    read_type="direct",
                    database_name=actual_db_path,
                    table_name=self._curr_table,
                    duckdb_extensions_dir=self.curr_config.duckdb_extensions_dir,
                    force_install=self.curr_config.force_install,
                )
            database_file_path = actual_db_path

        self._file_name = file_name
        self._database_file_path = database_file_path
        self._database_file_id = database_file_id
        self._query_rewrite_result = None
        self._last_sql_error = None
        self._current_suggested_questions = []

        self.api_call = ApiCall()
        super().__init__(chat_param=chat_param, system_app=system_app)

    def _create_reader_from_duckdb(
        self,
        conv_uid: str,
        duckdb_path: str,
        file_name: str,
        duckdb_table_name: str = None,
    ):
        import duckdb

        db_conn = duckdb.connect(database=duckdb_path, read_only=True)

        try:
            if not duckdb_table_name:
                query = (
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                )
                tables_result = db_conn.execute(query).fetchall()
                if tables_result:
                    duckdb_table_name = tables_result[0][0]
                else:
                    raise ValueError(f"在DuckDB数据库中未找到任何表: {duckdb_path}")

            reader = object.__new__(ExcelReader)
            reader.conv_uid = conv_uid
            reader.db = db_conn
            reader.temp_table_name = duckdb_table_name
            reader.table_name = duckdb_table_name
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

        if duckdb_path and os.path.exists(duckdb_path):
            database_file_path = duckdb_path
            database_file_id = None
        else:
            database_root_path = os.path.join(DATA_DIR, "_chat_excel_tmp")
            os.makedirs(database_root_path, exist_ok=True)
            # 使用 conv_uid 确保不同会话的数据库文件隔离，支持并发
            database_file_path = os.path.join(
                database_root_path, f"_chat_excel_{conv_uid}_{file_name}.duckdb"
            )
            database_file_id = None

        if file_path.startswith(_SCHEMA):
            file_path, file_meta = fs_client.download_file(file_path, dest_dir=DATA_DIR)
            file_name = os.path.basename(file_path)

            if not duckdb_path:
                # 使用 conv_uid 确保不同会话的数据库文件隔离，支持并发
                database_file_path = os.path.join(
                    database_root_path, f"_chat_excel_{conv_uid}_{file_name}.duckdb"
                )
                database_file_id = f"{file_meta.file_id}_{conv_uid}"
                db_files = fs_client.list_files(
                    bucket,
                    filters={"file_id": database_file_id},
                )
                if db_files:
                    fs_client.download_file(db_files[0].uri, database_file_path)

        return file_path, file_name, database_file_path, database_file_id

    @trace()
    async def generate_input_values(self) -> Dict:
        print("[DEBUG] ========== generate_input_values 开始 ==========")
        if (
            hasattr(self, "_cached_input_values")
            and self._cached_input_values is not None
        ):
            print("[DEBUG] 使用缓存的输入值")
            return self._cached_input_values

        print("[DEBUG] 确保数据表存在...")
        await self._ensure_data_analysis_table_exists()

        user_input = self.current_user_input.last_text
        detected_language = detect_language(user_input)
        print(f"[DEBUG] 用户输入: {user_input}, 语言: {detected_language}")
        self._detected_language = detected_language

        # 获取选中的表列表（优先使用 _selected_tables，否则使用 _all_table_names）
        selected_tables = getattr(self, "_selected_tables", None)
        if not selected_tables:
            if getattr(self, "_is_multi_table_mode", False) and hasattr(self, "_all_table_names"):
                selected_tables = self._all_table_names
            else:
                selected_tables = [self._curr_table]
        
        is_multi_table = len(selected_tables) > 1

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        # 根据是否多表模式获取对应的 prompt 模板
        prompt_templates = get_prompt_templates_by_language(detected_language, is_multi_table_mode=is_multi_table)

        # 只获取选中表的 schema 和样本数据
        all_schemas = []
        all_sample_data = []
        print(f"[DEBUG] 开始获取 {len(selected_tables)} 个表的schema和样本数据...")
        for tbl_name in selected_tables:
            try:
                print(f"[DEBUG] 获取表 {tbl_name} 的schema...")
                tbl_schema = await blocking_func_to_async(
                    self._executor, self.excel_reader.get_create_table_sql, tbl_name
                )
                all_schemas.append(f"-- 表: {tbl_name}\n{tbl_schema}")
                
                print(f"[DEBUG] 获取表 {tbl_name} 的样本数据...")
                tbl_cols, tbl_data = await blocking_func_to_async(
                    self._executor, self.excel_reader.get_sample_data, tbl_name
                )
                print(f"[DEBUG] 表 {tbl_name} 样本数据: {len(tbl_data)} 行")
                all_sample_data.append((tbl_name, tbl_cols, tbl_data))
            except Exception as e:
                print(f"[DEBUG] ⚠️ 获取表 {tbl_name} 的 schema 失败: {e}")
        
        table_schema = "\n\n".join(all_schemas)
        # 使用第一个表的示例数据
        if all_sample_data:
            _, colunms, datas = all_sample_data[0]
        else:
            colunms, datas = [], []
        logger.info(f"生成输入值：使用 {len(selected_tables)} 个表的 schema: {selected_tables}")

        data_time_range = await blocking_func_to_async(
            self._executor, self._get_data_time_range, self._curr_table
        )

        query_rewrite_info = ""
        relevant_columns_info = ""

        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                select_param_dict = json.loads(self.select_param)
            except Exception as e:
                logger.warning(f"解析select_param JSON失败: {e}")
                select_param_dict = None

        if select_param_dict and isinstance(select_param_dict, dict):
            data_schema_json = select_param_dict.get("data_schema_json")

            if data_schema_json:
                # 使用stream_call中完成的查询改写结果
                rewrite_result = self._query_rewrite_result

                if rewrite_result and rewrite_result.get("rewritten_query"):
                    # 构建 query_rewrite_info，包含多表策略信息
                    multi_table_strategy = rewrite_result.get("multi_table_strategy", {})
                    strategy_info = ""
                    if multi_table_strategy and is_multi_table:
                        strategy = multi_table_strategy.get("strategy", "")
                        tables_to_use = multi_table_strategy.get("tables_to_use", [])
                        join_condition = multi_table_strategy.get("join_condition", "")
                        strategy_info = f"""
多表查询策略：
  策略类型：{strategy}
  使用的表：{', '.join(tables_to_use) if tables_to_use else '未指定'}
  关联说明：{join_condition}
"""
                    
                    # 如果是多表模式，添加字段差异警告
                    schema_diff_warning = ""
                    if is_multi_table and len(selected_tables) > 1:
                        # 分析字段差异
                        all_columns_by_table = {}
                        all_tables_info = select_param_dict.get("all_tables_info", [])
                        
                        for table_info in all_tables_info:
                            tbl_name = table_info.get("table_name", "")
                            if tbl_name in selected_tables:
                                # 从建表SQL中提取字段
                                create_sql = table_info.get("create_table_sql", "")
                                if create_sql:
                                    import re
                                    col_matches = re.findall(r'"([^"]+)"', create_sql)
                                    all_columns_by_table[tbl_name] = set(col_matches)
                        
                        if len(all_columns_by_table) > 1:
                            diff_warning = self._analyze_schema_differences(
                                all_columns_by_table, selected_tables
                            )
                            if diff_warning:
                                schema_diff_warning = f"\n\n⚠️ 字段差异提醒：\n{diff_warning}\n"
                    
                    query_rewrite_info = f"""

用户的问题：{rewrite_result["rewritten_query"]}

相关字段：
{self._format_relevant_columns_with_table(rewrite_result.get("relevant_columns", []))}

分析建议：
{self._format_analysis_suggestions(rewrite_result.get("analysis_suggestions", []))}

分析逻辑：
{rewrite_result.get("analysis_logic", "")}
{strategy_info}{schema_diff_warning}
接下来请按照格式要求生成sql语句进行查询。
"""
                    extracted_knowledge = rewrite_result.get("_extracted_knowledge")
                    if extracted_knowledge:
                        await self._save_domain_knowledge(
                            extracted_knowledge, data_schema_json
                        )

                    # 构建重点关注字段信息（包含表名）
                    relevant_columns_info = await self._build_relevant_columns_info_with_table(
                        rewrite_result.get("relevant_columns", []),
                        select_param_dict,
                        selected_tables
                    )
                else:
                    # 如果查询改写失败或没有结果，设置默认值
                    query_rewrite_info = ""
                    relevant_columns_info = ""

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            _ANALYSIS_CONSTRAINTS_TEMPLATE,
        )

        # 根据是否多表模式选择约束条件
        if is_multi_table:
            from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
                _ANALYSIS_CONSTRAINTS_MULTI_TABLE_ZH,
                _ANALYSIS_CONSTRAINTS_MULTI_TABLE_EN,
            )
            # 根据检测到的语言选择模板
            multi_table_template = (
                _ANALYSIS_CONSTRAINTS_MULTI_TABLE_EN
                if detected_language == "en"
                else _ANALYSIS_CONSTRAINTS_MULTI_TABLE_ZH
            )
            table_names_str = ", ".join(selected_tables)
            analysis_constraints = multi_table_template.format(
                table_names=table_names_str, display_type=self._generate_numbered_list()
            )
            logger.info(f"多表模式：使用多表约束条件，表: {table_names_str}")
        else:
            analysis_constraints = _ANALYSIS_CONSTRAINTS_TEMPLATE.format(
                table_name=self._curr_table, display_type=self._generate_numbered_list()
            )

        # table_name 使用选中的表名
        table_name_for_prompt = ", ".join(selected_tables) if is_multi_table else self._curr_table

        input_values = {
            "user_input": self.current_user_input.last_text,
            "table_name": table_name_for_prompt,
            "display_type": self._generate_numbered_list(),
            "table_schema": table_schema,
            "data_example": json.dumps(
                datas, cls=EnhancedJSONEncoder, ensure_ascii=False
            ),
            "query_rewrite_info": query_rewrite_info,
            "relevant_columns_info": relevant_columns_info,
            "data_time_range": data_time_range or "",
            "duckdb_syntax_rules": prompt_templates["duckdb_rules"],
            "analysis_constraints": analysis_constraints,
            "examples": prompt_templates["examples"],
        }

        self._cached_input_values = input_values
        return input_values

    async def _build_model_request(self) -> ModelRequest:
        detected_language = getattr(self, "_detected_language", "zh")
        
        # 判断是否为多表模式
        selected_tables = getattr(self, "_selected_tables", None)
        if not selected_tables:
            if getattr(self, "_is_multi_table_mode", False) and hasattr(self, "_all_table_names"):
                selected_tables = self._all_table_names
            else:
                selected_tables = [self._curr_table]
        is_multi_table = len(selected_tables) > 1

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        prompt_templates = get_prompt_templates_by_language(detected_language, is_multi_table_mode=is_multi_table)

        dynamic_prompt = ChatPromptTemplate(
            messages=[
                SystemPromptTemplate.from_template(prompt_templates["system_prompt"]),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanPromptTemplate.from_template(
                    prompt_templates["user_prompt_template"]
                ),
            ]
        )

        original_prompt = self.prompt_template.prompt
        self.prompt_template.prompt = dynamic_prompt

        try:
            return await super()._build_model_request()
        finally:
            self.prompt_template.prompt = original_prompt

    async def _save_domain_knowledge(self, knowledge: dict, current_schema_json: str):
        try:
            import sqlite3
            from datetime import datetime

            column_name = knowledge.get("column_name")
            knowledge_text = knowledge.get("knowledge")

            if not column_name or not knowledge_text:
                return

            schema_obj = (
                json.loads(current_schema_json)
                if isinstance(current_schema_json, str)
                else current_schema_json
            )
            columns = schema_obj.get("columns", [])

            knowledge_saved = False
            for col in columns:
                if col.get("column_name") == column_name:
                    existing_knowledge = col.get("domain_knowledge", "")
                    if knowledge_text in existing_knowledge:
                        return

                    col["domain_knowledge"] = (
                        f"{existing_knowledge}\n• {knowledge_text}"
                        if existing_knowledge
                        else knowledge_text
                    )
                    knowledge_saved = True
                    break

            if not knowledge_saved:
                return

            current_file = Path(__file__)
            project_root = current_file
            for _ in range(9):
                project_root = project_root.parent
            meta_db_path = (
                project_root
                / "packages"
                / "pilot"
                / "data"
                / "excel_cache"
                / "excel_metadata.db"
            )

            if not meta_db_path.exists():
                return

            content_hash = getattr(self, "_content_hash", None)
            if not content_hash:
                return

            conn = sqlite3.connect(str(meta_db_path))
            cursor = conn.cursor()

            updated_schema_json = json.dumps(schema_obj, ensure_ascii=False, indent=2)

            cursor.execute(
                "UPDATE excel_metadata SET data_schema_json = ?, last_accessed = ? WHERE content_hash = ?",
                (updated_schema_json, datetime.now().isoformat(), content_hash),
            )

            conn.commit()
            conn.close()

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

    def _merge_multi_table_schemas(
        self, all_tables_info: List[Dict], selected_tables: List[str]
    ) -> str:
        """
        合并多个表的 schema 为一个完整的多表 schema JSON
        
        Args:
            all_tables_info: 所有表的信息列表
            selected_tables: 选中的表名列表
            
        Returns:
            合并后的 schema JSON 字符串
        """
        try:
            merged_schema = {
                "is_multi_table": True,
                "table_count": len(selected_tables),
                "table_names": selected_tables,
                "tables": [],
                "columns": [],  # 所有表的列合并（带表名前缀）
            }
            
            table_descriptions = []
            
            for table_info in all_tables_info:
                table_name = table_info.get("table_name")
                if table_name not in selected_tables:
                    continue
                    
                schema_json = table_info.get("data_schema_json")
                if not schema_json:
                    continue
                    
                try:
                    schema_obj = (
                        json.loads(schema_json)
                        if isinstance(schema_json, str)
                        else schema_json
                    )
                    
                    # 添加表信息
                    table_entry = {
                        "table_name": table_name,
                        "table_description": schema_obj.get("table_description", ""),
                        "row_count": schema_obj.get("row_count", table_info.get("row_count", 0)),
                        "column_count": schema_obj.get("column_count", table_info.get("column_count", 0)),
                        "columns": schema_obj.get("columns", []),
                    }
                    merged_schema["tables"].append(table_entry)
                    
                    # 收集表描述
                    desc = schema_obj.get("table_description", "")
                    if desc:
                        table_descriptions.append(f"- {table_name}: {desc}")
                    
                    # 合并列信息（带表名前缀，方便识别）
                    for col in schema_obj.get("columns", []):
                        col_with_table = col.copy()
                        col_with_table["source_table"] = table_name
                        merged_schema["columns"].append(col_with_table)
                        
                except Exception as e:
                    logger.warning(f"解析表 {table_name} 的 schema 失败: {e}")
            
            # 生成合并后的表描述
            merged_schema["table_description"] = (
                f"多表联合查询，包含以下 {len(selected_tables)} 个表：\n"
                + "\n".join(table_descriptions)
            )
            
            return json.dumps(merged_schema, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"合并多表 schema 失败: {e}")
            # 返回空 schema
            return json.dumps({"is_multi_table": True, "tables": []}, ensure_ascii=False)

    async def _build_multi_table_info_for_rewrite(
        self, selected_tables: List[str], all_tables_info: List[Dict]
    ) -> Dict:
        """
        为多表 Query 改写构建表信息
        不使用复杂的 data_schema_json，而是直接使用建表 SQL 和样本数据
        
        Args:
            selected_tables: 选中的表名列表
            all_tables_info: 所有表的信息
            
        Returns:
            包含多表信息的字典，格式：
            {
                "is_multi_table": True,
                "table_count": 2,
                "table_names": ["表1", "表2"],
                "table_description": "多表描述",
                "tables": [
                    {
                        "table_name": "表1",
                        "create_table_sql": "CREATE TABLE ...",
                        "sample_rows": [[...], [...]]
                    },
                    ...
                ],
                "schema_differences": "字段差异说明"
            }
        """
        result = {
            "is_multi_table": True,
            "table_count": len(selected_tables),
            "table_names": selected_tables,
            "tables": [],
        }
        
        table_descriptions = []
        all_columns_by_table = {}  # 记录每个表的字段列表，用于分析差异
        
        for table_name in selected_tables:
            # 从 all_tables_info 中找到对应的表信息
            table_info = None
            for info in all_tables_info:
                if info.get("table_name") == table_name:
                    table_info = info
                    break
            
            # 获取建表 SQL
            try:
                create_sql = await blocking_func_to_async(
                    self._executor, 
                    self.excel_reader.get_create_table_sql, 
                    table_name
                )
            except Exception as e:
                logger.warning(f"获取表 {table_name} 的建表SQL失败: {e}")
                create_sql = table_info.get("create_table_sql", "") if table_info else ""
            
            # 获取样本数据
            try:
                cols, rows = await blocking_func_to_async(
                    self._executor,
                    self.excel_reader.get_sample_data,
                    table_name,
                    2  # 只需要2行样本数据
                )
                # 将样本数据转换为列表格式
                sample_rows = []
                for row in rows:
                    sample_rows.append(list(row))
            except Exception as e:
                logger.warning(f"获取表 {table_name} 的样本数据失败: {e}")
                cols, sample_rows = [], []
            
            # 记录字段列表
            all_columns_by_table[table_name] = set(cols) if cols else set()
            
            # 获取表描述
            table_desc = ""
            if table_info and table_info.get("data_schema_json"):
                try:
                    schema_obj = json.loads(table_info["data_schema_json"]) if isinstance(
                        table_info["data_schema_json"], str
                    ) else table_info["data_schema_json"]
                    table_desc = schema_obj.get("table_description", "")
                except Exception:
                    pass
            
            table_entry = {
                "table_name": table_name,
                "table_description": table_desc,
                "create_table_sql": create_sql,
                "columns": list(cols) if cols else [],
                "sample_rows": sample_rows,
                "row_count": table_info.get("row_count", 0) if table_info else 0,
                "column_count": table_info.get("column_count", 0) if table_info else 0,
            }
            result["tables"].append(table_entry)
            
            if table_desc:
                table_descriptions.append(f"- {table_name}: {table_desc}")
        
        # 分析字段差异
        schema_diff_warning = self._analyze_schema_differences(all_columns_by_table, selected_tables)
        if schema_diff_warning:
            result["schema_differences"] = schema_diff_warning
            logger.warning(f"多表字段差异: {schema_diff_warning}")
        
        # 生成合并后的表描述
        result["table_description"] = (
            f"多表联合查询，包含以下 {len(selected_tables)} 个表：\n"
            + "\n".join(table_descriptions)
        )
        
        return result
    
    def _analyze_schema_differences(
        self, all_columns_by_table: Dict[str, set], table_names: List[str]
    ) -> str:
        """
        分析多个表之间的字段差异
        
        Args:
            all_columns_by_table: 每个表的字段集合
            table_names: 表名列表
            
        Returns:
            字段差异说明（如果有差异）
        """
        if len(table_names) < 2:
            return ""
        
        # 找出所有表的字段并集和交集
        all_columns = set()
        for cols in all_columns_by_table.values():
            all_columns.update(cols)
        
        common_columns = set(all_columns_by_table[table_names[0]])
        for table_name in table_names[1:]:
            common_columns &= all_columns_by_table[table_name]
        
        # 如果所有字段都相同，无需警告
        if len(common_columns) == len(all_columns):
            return ""
        
        # 找出每个表独有的字段
        unique_columns_by_table = {}
        for table_name in table_names:
            table_cols = all_columns_by_table[table_name]
            unique_cols = table_cols - common_columns
            if unique_cols:
                unique_columns_by_table[table_name] = unique_cols
        
        # 构建差异说明
        if not unique_columns_by_table:
            return ""
        
        diff_parts = [
            "⚠️ 重要提示：这些表的字段结构不完全相同！",
            f"共同字段数量：{len(common_columns)}",
            "\n各表独有的字段："
        ]
        
        for table_name, unique_cols in unique_columns_by_table.items():
            cols_str = ", ".join(list(unique_cols)[:])
                
            diff_parts.append(f"- {table_name}: {cols_str}")
        
        diff_parts.append("\n在编写SQL时，请注意：")
        diff_parts.append("1. 使用UNION时，确保SELECT的字段在所有表中都存在")
        diff_parts.append("2. 如果某个字段只存在于部分表，可以使用NULL或默认值填充")
        diff_parts.append("3. 或者分别查询各表，然后在应用层合并结果")
        
        return "\n".join(diff_parts)

    def _clean_history_content(self, content: str) -> str:
        import re

        # 去除 chart-view 标签
        content = re.sub(
            r"<chart-view[^>]*>.*?</chart-view>", "", content, flags=re.DOTALL
        )
        
        # 去除 SUGGESTED_QUESTIONS 标记
        content = re.sub(
            r"<!--SUGGESTED_QUESTIONS:.*?-->", "", content, flags=re.DOTALL
        )
        
        # 去除 vis-thinking 标签（匹配任意数量的反引号，从3个到6个）
        # 匹配格式：```vis-thinking ... ``` 或 ``````vis-thinking ... ``````
        content = re.sub(
            r"`{3,6}vis-thinking.*?`{3,6}", "", content, flags=re.DOTALL
        )
        
        # 清理多余的空行（将3个或更多连续换行符替换为2个）
        content = re.sub(r"\n{3,}", "\n\n", content)

        lines = content.split("\n")
        cleaned_lines = []
        skip_data = False

        for line in lines:
            if any(marker in line for marker in ["[{", '{"data":', '"rows":']):
                if len(line) > 200:
                    skip_data = True
                    continue

            if skip_data and ("}]" in line or line.strip() == "}"):
                skip_data = False
                continue

            if not skip_data:
                cleaned_lines.append(line)

        content = "\n".join(cleaned_lines)

        if len(content) > 1000:
            api_call_match = re.search(r"<api-call>.*?</api-call>", content, re.DOTALL)
            if api_call_match:
                before_api = content[: api_call_match.start()].strip()
                if len(before_api) > 200:
                    before_api = before_api[:200] + "..."
                content = before_api + "\n" + api_call_match.group(0)
            else:
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

    def _format_relevant_columns_with_table(self, columns: List[Dict]) -> str:
        """格式化相关列信息（包含表名）"""
        if not columns:
            return "未指定"

        formatted = []
        for col in columns:
            table_name = col.get("table_name", "")
            col_name = col.get("column_name", "")
            usage = col.get("usage", "")
            if table_name:
                formatted.append(f"  • {table_name}.{col_name}: {usage}")
            else:
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
        if not columns:
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"
            return (
                "No relevant column information found."
                if is_english
                else "未找到相关列信息。"
            )

        detected_language = getattr(self, "_detected_language", "zh")
        is_english = detected_language == "en"

        header = "Key fields to focus on:" if is_english else "你应该重点关注的字段为："
        formatted_parts = [header]

        for col in columns:
            col_name = col.get("column_name", "")
            data_type = col.get("data_type", "")
            description = col.get("description", "")
            analysis_usage = col.get("analysis_usage", [])
            domain_knowledge = col.get("domain_knowledge", "")

            col_text = f"  • {col_name}"
            if data_type:
                label = "Data type" if is_english else "数据类型"
                col_text += f"\n    {label}: {data_type}"
            if description:
                label = "Description" if is_english else "描述"
                col_text += f"\n    {label}: {description}"

            if domain_knowledge:
                label = "**Key Knowledge**" if is_english else "**关键知识**"
                col_text += f"\n    {label}: {domain_knowledge}"

            if analysis_usage:
                label = "Analysis usage" if is_english else "分析用途"
                col_text += f"\n    {label}: {', '.join(analysis_usage)}"

            if "statistics_summary" in col:
                label = "Statistics" if is_english else "统计信息"
                col_text += f"\n    {label}: {col['statistics_summary']}"

            if "unique_values_top20" in col:
                unique_vals = col["unique_values_top20"]
                label = "Possible values" if is_english else "可选值"
                label_partial = (
                    "Possible values (partial)" if is_english else "可选值(部分)"
                )
                if len(unique_vals) <= 20:
                    col_text += f"\n    {label}: {', '.join(map(str, unique_vals))}"
                else:
                    partial_vals = ", ".join(map(str, unique_vals[:20]))
                    col_text += f"\n    {label_partial}: {partial_vals}..."

            formatted_parts.append(col_text)

        return "\n\n".join(formatted_parts)

    async def _build_relevant_columns_info_with_table(
        self, 
        relevant_columns: List[Dict], 
        select_param_dict: Dict,
        selected_tables: List[str]
    ) -> str:
        """
        构建包含表名的重点关注字段信息
        
        Args:
            relevant_columns: Query 改写返回的相关字段列表
            select_param_dict: select_param 字典
            selected_tables: 选中的表列表
        
        Returns:
            格式化后的重点关注字段信息
        """
        if not relevant_columns:
            return ""
        
        detected_language = getattr(self, "_detected_language", "zh")
        is_english = detected_language == "en"
        
        # 获取所有表的 schema 信息
        all_tables_info = select_param_dict.get("all_tables_info", [])
        
        # 构建表名到 schema 的映射
        table_schema_map = {}
        for table_info in all_tables_info:
            tbl_name = table_info.get("table_name", "")
            if tbl_name in selected_tables:
                schema_json = table_info.get("data_schema_json")
                if schema_json:
                    try:
                        schema_obj = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                        table_schema_map[tbl_name] = schema_obj.get("columns", [])
                    except Exception:
                        pass
        
        # 如果没有多表信息，尝试从单表 data_schema_json 获取
        if not table_schema_map:
            data_schema_json = select_param_dict.get("data_schema_json")
            if data_schema_json:
                try:
                    schema_obj = json.loads(data_schema_json) if isinstance(data_schema_json, str) else data_schema_json
                    table_schema_map[self._curr_table] = schema_obj.get("columns", [])
                except Exception:
                    pass
        
        header = "Key fields to focus on:" if is_english else "你应该重点关注的字段为："
        formatted_parts = [header]
        
        for col in relevant_columns:
            table_name = col.get("table_name", "")
            col_name = col.get("column_name", "")
            usage = col.get("usage", "")
            
            # 查找列的详细信息
            col_details = None
            if table_name and table_name in table_schema_map:
                for col_info in table_schema_map[table_name]:
                    if col_info.get("column_name") == col_name:
                        col_details = col_info
                        break
            else:
                # 如果没有指定表名，在所有表中查找
                for tbl_cols in table_schema_map.values():
                    for col_info in tbl_cols:
                        if col_info.get("column_name") == col_name:
                            col_details = col_info
                            break
                    if col_details:
                        break
            
            # 格式化列信息
            if table_name:
                col_text = f"  • {table_name}.{col_name}"
            else:
                col_text = f"  • {col_name}"
            
            if col_details:
                data_type = col_details.get("data_type", "")
                description = col_details.get("description", "")
                domain_knowledge = col_details.get("domain_knowledge", "")
                
                if data_type:
                    label = "Data type" if is_english else "数据类型"
                    col_text += f"\n    {label}: {data_type}"
                if description:
                    label = "Description" if is_english else "描述"
                    col_text += f"\n    {label}: {description}"
                
                # 添加领域知识（如果有）
                if domain_knowledge:
                    label = "**Key Knowledge**" if is_english else "**关键知识**"
                    col_text += f"\n    {label}: {domain_knowledge}"
                
                if "statistics_summary" in col_details:
                    label = "Statistics" if is_english else "统计信息"
                    col_text += f"\n    {label}: {col_details['statistics_summary']}"
                
                if "unique_values_top20" in col_details:
                    unique_vals = col_details["unique_values_top20"]
                    label = "Possible values" if is_english else "可选值"
                    if len(unique_vals) <= 10:
                        col_text += f"\n    {label}: {', '.join(map(str, unique_vals))}"
                    else:
                        partial_vals = ", ".join(map(str, unique_vals[:10]))
                        col_text += f"\n    {label}(部分): {partial_vals}..."
            else:
                # 如果没有详细信息，至少显示用途
                if usage:
                    label = "Usage" if is_english else "用途"
                    col_text += f"\n    {label}: {usage}"
            
            formatted_parts.append(col_text)
        
        return "\n\n".join(formatted_parts)

    def _build_multi_table_schema_info_for_summary(
        self, 
        selected_tables: List[str], 
        all_tables_info: List[Dict],
        is_english: bool = False
    ) -> str:
        """
        为总结生成多表的 schema 信息
        
        Args:
            selected_tables: 选中的表名列表
            all_tables_info: 所有表的信息
            is_english: 是否使用英文
            
        Returns:
            格式化后的多表 schema 信息
        """
        tables_info_parts = []
        
        for table_name in selected_tables:
            # 从 all_tables_info 中找到对应的表信息
            table_info = None
            for info in all_tables_info:
                if info.get("table_name") == table_name:
                    table_info = info
                    break
            
            if not table_info:
                continue
            
            # 获取表的 schema
            schema_json = table_info.get("data_schema_json")
            if not schema_json:
                continue
            
            try:
                schema_obj = (
                    json.loads(schema_json)
                    if isinstance(schema_json, str)
                    else schema_json
                )
                
                table_description = schema_obj.get("table_description", "")
                columns_summary = []
                
                # 构建列信息
                for col in schema_obj.get("columns", []):
                    col_name = col.get("column_name", "")
                    description = col.get("description", "")
                    domain_knowledge = col.get("domain_knowledge", "")
                    
                    col_info_text = f"  - {col_name}: {description}"
                    
                    # 如果有领域知识，添加到描述中
                    if domain_knowledge:
                        if is_english:
                            col_info_text += f" [Key Knowledge: {domain_knowledge}]"
                        else:
                            col_info_text += f" [关键知识: {domain_knowledge}]"
                    
                    # 如果有unique_values_top20，补充显示前5个值
                    if "unique_values_top20" in col:
                        unique_vals = col["unique_values_top20"]
                        if isinstance(unique_vals, list) and len(unique_vals) > 0:
                            top_5_values = unique_vals[:5]
                            if is_english:
                                col_info_text += f" (Example values: {', '.join(map(str, top_5_values))})"
                            else:
                                col_info_text += f" (示例值: {', '.join(map(str, top_5_values))})"
                    
                    columns_summary.append(col_info_text)
                
                # 构建单个表的信息
                if is_english:
                    table_info_text = f"""
### Table: {table_name}
Description: {table_description}
Columns:
{chr(10).join(columns_summary)}
"""
                else:
                    table_info_text = f"""
### 表: {table_name}
表描述: {table_description}
字段:
{chr(10).join(columns_summary)}
"""
                tables_info_parts.append(table_info_text)
                
            except Exception as e:
                logger.warning(f"解析表 {table_name} 的 schema 失败: {e}")
        
        if not tables_info_parts:
            return ""
        
        # 组合所有表的信息
        if is_english:
            header = f"=== Data Table Information ({len(selected_tables)} tables) ==="
        else:
            header = f"=== 数据表信息（共 {len(selected_tables)} 个表）==="
        
        return f"{header}\n{''.join(tables_info_parts)}"

    def _get_data_time_range(self, table_name: str) -> str:
        try:
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

            date_col = date_columns[0]
            query = (
                f'SELECT MIN("{date_col}") as min_date, '
                f'MAX("{date_col}") as max_date FROM "{table_name}"'
            )
            result = self.excel_reader.db.sql(query).fetchone()

            if result and result[0] and result[1]:
                min_date = result[0]
                max_date = result[1]

                if isinstance(min_date, str):
                    min_date = min_date[:10]
                else:
                    min_date = str(min_date)[:10]

                if isinstance(max_date, str):
                    max_date = max_date[:10]
                else:
                    max_date = str(max_date)[:10]

                time_range = f"\n\n数据时间范围：{min_date} 至 {max_date}"
                time_range += (
                    "\n（注意：进行同比分析时，请确保SQL查询包含足够的历史数据）"
                )
                return time_range

        except Exception as e:
            logger.warning(f"获取数据时间范围失败: {e}")

        return ""

    async def prepare(self):
        logger.info(f"{self.chat_mode} prepare start!")
        if self.has_history_messages():
            return None

        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                select_param_dict = json.loads(self.select_param)
            except Exception:
                select_param_dict = {}

        if select_param_dict and isinstance(select_param_dict, dict):
            summary_prompt = select_param_dict.get("summary_prompt")

            if summary_prompt and isinstance(summary_prompt, str):
                if self._curr_table != "data_analysis_table":
                    logger.info(f"使用DuckDB缓存，表 {self._curr_table} 已存在")
                else:
                    try:
                        await blocking_func_to_async(
                            self._executor, self._create_simple_data_analysis_table
                        )
                    except Exception as e:
                        logger.warning(f"使用缓存创建表失败: {e}")

                await self._generate_and_save_excel_info(None)

                return ModelOutput(
                    error_code=0,
                    text="数据分析结构已加载（使用缓存）",
                    finish_reason="stop",
                )
        chat_param = ChatParam(
            chat_session_id=self.chat_session_id,
            current_user_input=f"[{self.excel_reader.excel_file_name}] Analyze！",
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

        await self._generate_and_save_excel_info(result)
        return result

    def _create_simple_data_analysis_table(self):
        try:
            tables = self.excel_reader.db.sql("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            # 多表模式：检查所有表是否都已存在
            if getattr(self, "_is_multi_table_mode", False) and hasattr(self, "_all_table_names"):
                all_exist = all(t in table_names for t in self._all_table_names)
                if all_exist:
                    logger.info(f"多表模式：所有表都已存在 {self._all_table_names}")
                    return
                # 如果有表不存在，记录警告但不抛出错误
                missing = [t for t in self._all_table_names if t not in table_names]
                if missing:
                    logger.warning(f"多表模式：部分表不存在 {missing}，但继续使用已存在的表")
                return

            if self._curr_table in table_names:
                return

            if (
                hasattr(self.excel_reader, "table_name")
                and self.excel_reader.table_name
            ):
                source_table = self.excel_reader.table_name
                # 处理多表名称（逗号分隔的情况）
                if "," in source_table:
                    # 多表模式下，source_table 可能是 "table1, table2, table3"
                    # 检查这些表是否都存在
                    source_tables = [t.strip() for t in source_table.split(",")]
                    existing_sources = [t for t in source_tables if t in table_names]
                    if existing_sources:
                        logger.info(f"多表模式：找到已存在的表 {existing_sources}")
                        # 设置 _curr_table 为第一个存在的表
                        self._curr_table = existing_sources[0]
                        self._all_table_names = existing_sources
                        self._is_multi_table_mode = len(existing_sources) > 1
                        return
                elif source_table in table_names and source_table != self._curr_table:
                    sql = (
                        f'CREATE TABLE "{self._curr_table}" '
                        f'AS SELECT * FROM "{source_table}";'
                    )
                    self.excel_reader.db.sql(sql)
                    return

            if "temp_table" in table_names:
                sql = f'CREATE TABLE "{self._curr_table}" AS SELECT * FROM temp_table;'
                self.excel_reader.db.sql(sql)
                return

            # 如果数据库中有任何表，使用第一个表
            if table_names:
                first_table = table_names[0]
                logger.info(f"使用数据库中已存在的第一个表: {first_table}")
                self._curr_table = first_table
                self._all_table_names = table_names
                self._is_multi_table_mode = len(table_names) > 1
                return

            table_name_attr = getattr(self.excel_reader, "table_name", None)
            logger.error(
                f"找不到源表：temp_table 不存在，且 "
                f"excel_reader.table_name ({table_name_attr}) 也不存在"
            )
            raise ValueError(f"找不到可用的源表来创建 {self._curr_table}")
        except Exception as e:
            logger.error(f"Failed to create {self._curr_table}: {e}")
            raise

    async def _ensure_data_analysis_table_exists(self):
        try:
            tables = await blocking_func_to_async(
                self._executor,
                lambda: self.excel_reader.db.sql("SHOW TABLES").fetchall(),
            )
            table_names = [t[0] for t in tables]

            # 多表模式：检查是否有任何表存在
            if getattr(self, "_is_multi_table_mode", False) and hasattr(self, "_all_table_names"):
                existing_tables = [t for t in self._all_table_names if t in table_names]
                if existing_tables:
                    # 更新 _curr_table 为第一个存在的表
                    self._curr_table = existing_tables[0]
                    logger.info(f"多表模式：使用已存在的表 {existing_tables}")
                    return
                # 如果指定的表都不存在，但数据库中有其他表，使用这些表
                if table_names:
                    self._all_table_names = table_names
                    self._curr_table = table_names[0]
                    logger.info(f"多表模式：使用数据库中的表 {table_names}")
                    return

            if self._curr_table not in table_names:
                await blocking_func_to_async(
                    self._executor, self._create_simple_data_analysis_table
                )
        except Exception as e:
            logger.error(f"确保表存在时失败: {e}")
            raise

    async def _generate_and_save_excel_info(self, learning_result: ModelOutput = None):
        try:
            columns, top_10_rows = await blocking_func_to_async(
                self._executor,
                self.excel_reader.get_sample_data,
                self._curr_table,
                10,
            )

            row_count, column_count = await blocking_func_to_async(
                self._executor, self._get_table_stats, self._curr_table
            )

            data_description = None
            data_schema_json = None

            if learning_result and learning_result.has_text:
                data_description = learning_result.text
            else:
                select_param_dict = self.select_param
                if isinstance(self.select_param, str):
                    try:
                        select_param_dict = json.loads(self.select_param)
                    except Exception:
                        select_param_dict = {}

                if isinstance(select_param_dict, dict):
                    data_description = select_param_dict.get("summary_prompt")
                    data_schema_json = select_param_dict.get("data_schema_json")

            suggested_questions = []
            if data_schema_json:
                try:
                    schema = json.loads(data_schema_json)
                    suggested_questions = schema.get("suggested_questions", [])
                except Exception as e:
                    logger.warning(f"从 data_schema_json 提取推荐问题失败: {e}")

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
        except Exception as e:
            logger.error(f"生成并保存 Excel 基本信息失败: {e}")
            import traceback

            traceback.print_exc()

    def _get_table_stats(self, table_name: str) -> tuple:
        try:
            row_result = self.excel_reader.db.sql(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            row_count = row_result[0] if row_result else 0

            columns, _ = self.excel_reader.get_columns(table_name)
            column_count = len(columns) if columns else 0

            return row_count, column_count
        except Exception as e:
            logger.error(f"获取表统计信息失败: {e}")
            return 0, 0

    def stream_plugin_call(self, text):
        with root_tracer.start_span(
            "ChatExcel.stream_plugin_call.run_display_sql", metadata={"text": text}
        ):
            result = self.api_call.display_sql_llmvis(
                text,
                self.excel_reader.get_df_by_sql_ex,
            )
            logger.info(f"stream_plugin_call 返回结果长度: {len(result) if result else 0}")
            logger.debug(f"stream_plugin_call 返回内容（前500字符）: {result[:500] if result else 'None'}")
            return result

    async def stream_call(self, text_output: bool = True, incremental: bool = False):
        # 先进行流式查询改写
        print("[DEBUG] ========== stream_call 开始 ==========")
        await self._ensure_data_analysis_table_exists()
        print("[DEBUG] ✅ 表检查完成")
        
        user_input = self.current_user_input.last_text
        detected_language = detect_language(user_input)
        self._detected_language = detected_language
        print(f"[DEBUG] 用户提问: {user_input}, 检测语言: {detected_language}")

        select_param_dict = self.select_param
        if isinstance(self.select_param, str):
            try:
                select_param_dict = json.loads(self.select_param)
            except Exception as e:
                logger.warning(f"解析select_param JSON失败: {e}")
                select_param_dict = None

        # 检查是否需要查询改写
        need_rewrite = False
        data_schema_json = None
        table_schema = None
        chat_history = []
        all_tables_info = None  # 多表模式下的所有表信息
        print("[DEBUG] 开始检查是否需要查询改写...")
        
        if select_param_dict and isinstance(select_param_dict, dict):
            data_schema_json = select_param_dict.get("data_schema_json")
            all_tables_info = select_param_dict.get("all_tables_info")  # 多表模式
            
            if data_schema_json:
                need_rewrite = True
                # 获取表结构
                table_schema = await blocking_func_to_async(
                    self._executor, self.excel_reader.get_create_table_sql, self._curr_table
                )
                # 构建聊天历史（限制最近3轮，约6条消息）
                if hasattr(self, "history_messages") and self.history_messages:
                    current_round_messages = []
                    last_role = None
                    
                    # 只取最近6条消息（约3轮对话）
                    recent_messages = self.history_messages[-6:]

                    for msg in recent_messages:
                        if not hasattr(msg, "content"):
                            continue

                        role = getattr(msg, "role", "user")
                        content = msg.content

                        if hasattr(content, "get_text"):
                            try:
                                content = content.get_text()
                            except Exception:
                                content = str(content)
                        elif isinstance(content, list):
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

                        content = self._clean_history_content(content)

                        if role == last_role and current_round_messages:
                            current_round_messages[-1]["content"] += "\n" + content
                        else:
                            current_round_messages.append(
                                {"role": role, "content": content}
                            )
                            last_role = role

                    chat_history = current_round_messages

        # 多表模式：先进行表选择
        selected_tables = None
        if (
            getattr(self, "_is_multi_table_mode", False) 
            and all_tables_info 
            and len(all_tables_info) > 1 
            and self.llm_client
        ):
            print(f"[DEBUG] 多表模式，开始表选择，总表数: {len(all_tables_info)}")
            try:
                from dbgpt_app.scene.chat_data.chat_excel.query_rewrite import (
                    TableSelectionAgent,
                )
                
                table_selection_agent = TableSelectionAgent(self.llm_client, self.llm_model)
                print("[DEBUG] 表选择Agent创建完成，开始流式选择...")
                
                # 流式表选择
                selection_result = None
                async for chunk in table_selection_agent.select_tables_stream(
                    self.current_user_input.last_text,
                    all_tables_info,
                    chat_history,
                ):
                    if isinstance(chunk, str):
                        # 流式输出表选择过程
                        if text_output:
                            yield chunk
                        else:
                            yield ModelOutput.build(
                                text=chunk,
                                error_code=0,
                                finish_reason="continue"
                            )
                    elif isinstance(chunk, dict):
                        selection_result = chunk
                        break
                
                if selection_result:
                    selected_tables = selection_result.get("selected_tables", [])
                    selection_reason = selection_result.get("selection_reason", "")
                    logger.info(f"表选择完成: {selected_tables}, 理由: {selection_reason}")
                    
                    # 保存选中的表列表，供后续流程使用
                    self._selected_tables = selected_tables
                    
                    # 输出表选择结果的思考过程
                    from dbgpt.vis.tags.vis_thinking import VisThinking
                    
                    is_english = detected_language == "en"
                    if is_english:
                        thinking_content = f"""### Table Selection Result
**Selected Tables**: {', '.join(selected_tables)}
**Reason**: {selection_reason}"""
                    else:
                        thinking_content = f"""### 表选择结果
**选中的表**: {', '.join(selected_tables)}
**选择理由**: {selection_reason}"""
                    
                    vis_thinking_output = VisThinking().sync_display(content=thinking_content)
                    if text_output:
                        yield vis_thinking_output
                    else:
                        yield ModelOutput.build(
                            text=vis_thinking_output, error_code=0, finish_reason="continue"
                        )
                    
                    # 更新当前使用的表
                    if len(selected_tables) == 1:
                        self._curr_table = selected_tables[0]
                        # 更新 data_schema_json 为选中表的 schema
                        for table_info in all_tables_info:
                            if table_info.get("table_name") == selected_tables[0]:
                                data_schema_json = table_info.get("data_schema_json")
                                break
                    else:
                        # 多表查询：合并选中表的 schema 为一个完整的多表 schema
                        self._curr_table = selected_tables[0]  # 设置第一个表为当前表
                        data_schema_json = self._merge_multi_table_schemas(
                            all_tables_info, selected_tables
                        )
                        logger.info(f"多表模式：已合并 {len(selected_tables)} 个表的 schema")
                    
            except Exception as e:
                logger.warning(f"表选择失败，使用所有表: {e}")
                import traceback
                traceback.print_exc()

        # 如果需要查询改写，先进行流式改写
        if need_rewrite and self.llm_client:
            try:
                from dbgpt_app.scene.chat_data.chat_excel.query_rewrite import (
                    QueryRewriteAgent,
                )

                rewrite_agent = QueryRewriteAgent(self.llm_client, self.llm_model)
                
                # 多表模式：获取所有选中表的样本数据和建表SQL
                selected_table_list = getattr(self, "_selected_tables", None)
                is_multi_table_rewrite = (
                    selected_table_list 
                    and len(selected_table_list) > 1 
                    and all_tables_info
                )
                
                if is_multi_table_rewrite:
                    # 多表模式：构建包含所有表信息的结构
                    multi_table_info = await self._build_multi_table_info_for_rewrite(
                        selected_table_list, all_tables_info
                    )
                    # 使用多表信息替代单表的 data_schema_json
                    rewrite_schema = json.dumps(multi_table_info, ensure_ascii=False)
                    rewrite_table_desc = multi_table_info.get("table_description", "")
                    rewrite_sample_data = None  # 样本数据已包含在 multi_table_info 中
                    logger.info(f"多表Query改写：使用 {len(selected_table_list)} 个表的信息")
                else:
                    # 单表模式：使用原有逻辑
                    rewrite_schema = data_schema_json
                    rewrite_table_desc = table_schema
                    try:
                        rewrite_sample_data = await blocking_func_to_async(
                            self._executor, 
                            self.excel_reader.get_sample_data, 
                            self._curr_table,
                            2  # 只需要2行样本数据
                        )
                    except Exception as e:
                        logger.warning(f"获取样本数据失败: {e}")
                        rewrite_sample_data = None
                
                # 流式输出查询改写结果
                rewrite_result = None
                
                async for chunk in rewrite_agent.rewrite_query_stream(
                    self.current_user_input.last_text,
                    rewrite_schema,
                    rewrite_table_desc,
                    chat_history,
                    sample_rows=rewrite_sample_data,
                ):
                    if isinstance(chunk, str):
                        # 流式输出原始文本（JSON格式）
                        # 输出完整内容，让前端看到逐渐增长的完整JSON
                        if text_output:
                            # 直接输出完整JSON文本，前端会实时显示完整内容
                            yield chunk
                        else:
                            # 使用ModelOutput格式，输出完整内容
                            yield ModelOutput.build(
                                text=chunk,
                                error_code=0,
                                finish_reason="continue"
                            )
                    elif isinstance(chunk, dict):
                        # 解析完成，保存结果
                        rewrite_result = chunk
                        self._query_rewrite_result = rewrite_result
                        
                        # 检查问题是否与数据表相关
                        is_relevant = rewrite_result.get("is_relevant", True)
                        if not is_relevant:
                            logger.info(f"检测到非数据分析问题，路由到通用对话: {self.current_user_input.last_text}")
                            # 路由到通用对话agent
                            async for response in self._handle_general_chat():
                                yield response
                            return
                        
                        # 保存领域知识
                        extracted_knowledge = rewrite_result.get("_extracted_knowledge")
                        if extracted_knowledge:
                            await self._save_domain_knowledge(
                                extracted_knowledge, data_schema_json
                            )
                        break
                
                # 输出完成后，展示格式化后的结果
                if rewrite_result:
                    thinking_stage1 = self._format_query_rewrite_thinking(
                        rewrite_result
                    )
                    if thinking_stage1:
                        from dbgpt.vis.tags.vis_thinking import VisThinking

                        vis_thinking_output = VisThinking().sync_display(
                            content=thinking_stage1
                        )
                        if text_output:
                            yield vis_thinking_output
                        else:
                            yield ModelOutput.build(
                                text=vis_thinking_output, error_code=0, finish_reason="continue"
                            )
            except Exception as e:
                logger.warning(f"流式Query改写失败，使用原始问题: {e}")
                self._query_rewrite_result = None

        # 生成输入值（如果已经完成查询改写，会使用缓存的结果）
        await self.generate_input_values()

        payload = await self._build_model_request()
        self._last_sql_error = None
        full_output = await self.call_llm_operator(payload)

        ai_response_text = ""
        view_message = ""
        
        if full_output:
            try:
                text_msg = full_output.text if full_output.has_text else ""
                view_msg = self.stream_plugin_call(text_msg)

                if self._last_sql_error:
                    logger.warning(f"SQL执行失败，跳过总结生成: {self._last_sql_error[:100]}")
                    view_msg = full_output.gen_text_with_thinking(new_text=view_msg)
                    ai_full_response = self._combine_thinking_and_text(full_output, view_msg)
                    ai_response_text = ai_full_response
                    view_message = view_msg
                else:
                    import re
                    
                    # 1. 提取思考结果
                    thinking_content = ""
                    if self._query_rewrite_result:
                        thinking_stage1 = self._format_query_rewrite_thinking(
                            self._query_rewrite_result
                        )
                        if thinking_stage1:
                            from dbgpt.vis.tags.vis_thinking import VisThinking
                            thinking_content = VisThinking().sync_display(
                                content=thinking_stage1
                            )
                    
                    # 2. 提取图表内容
                    chart_pattern = r"(<chart-view.*?</chart-view>)"
                    chart_matches = re.findall(chart_pattern, view_msg, re.DOTALL)
                    chart_content = "\n\n".join(chart_matches) if chart_matches else ""
                    
                    # 流式输出summary（引导语）
                    summary_result = None
                    async for chunk in self._generate_result_summary_stream(
                        text_msg, view_msg, text_output=text_output
                    ):
                        if isinstance(chunk, str):
                            # 流式输出：思考结果 + 引导语（图表稍后添加）
                            if thinking_content:
                                combined_output = f"{thinking_content}\n\n{chunk}"
                            else:
                                combined_output = chunk
                            if text_output:
                                yield combined_output
                            else:
                                yield ModelOutput.build(
                                    text=combined_output,
                                    error_code=0,
                                    finish_reason="continue"
                                )
                        elif isinstance(chunk, dict):
                            summary_result = chunk
                            break
                    
                    summary_text = summary_result.get("summary", "") if summary_result else ""
                    suggested_questions = summary_result.get("suggested_questions", []) if summary_result else []
                    
                    self._current_suggested_questions = suggested_questions
                    
                    # 组合最终输出：思考结果 + 引导语 + 图表（引导语在图表上方）
                    parts = []
                    if thinking_content:
                        parts.append(thinking_content)
                    if summary_text:
                        parts.append(summary_text)
                    elif not chart_content:
                        # 没有引导语也没有图表
                        detected_language = getattr(self, "_detected_language", "zh")
                        is_english = detected_language == "en"
                        no_result_msg = "Query completed." if is_english else "查询已完成。"
                        parts.append(no_result_msg)
                    if chart_content:
                        parts.append(chart_content)
                    
                    view_message = "\n\n".join(parts)
                    
                    # 如果有推荐问题，在view_msg末尾添加特殊标记
                    if suggested_questions:
                        questions_json = json.dumps({"suggested_questions": suggested_questions}, ensure_ascii=False)
                        view_message += f"\n\n<!--SUGGESTED_QUESTIONS:{questions_json}-->"
                    
                    view_message = full_output.gen_text_with_thinking(new_text=view_message)
                    ai_response_text = self._combine_thinking_and_text(full_output, view_message)

                if text_output:
                    yield view_message
                else:
                    yield ModelOutput.build(
                        view_message,
                        "",
                        error_code=full_output.error_code if full_output else 0,
                        finish_reason=(
                            full_output.finish_reason if full_output else "stop"
                        ),
                    )
            except Exception as e:
                logger.error(f"处理输出时出错: {e}")
                error_msg = f"数据查询失败：{str(e)}"
                ai_response_text = error_msg
                view_message = error_msg
                if text_output:
                    yield error_msg
                else:
                    yield ModelOutput.build(
                        error_msg, "", error_code=1, finish_reason="error"
                    )
        else:
            error_msg = "生成SQL失败，请重试"
            ai_response_text = error_msg
            view_message = error_msg
            if text_output:
                yield error_msg
            else:
                yield ModelOutput.build(
                    error_msg, "", error_code=1, finish_reason="error"
                )
        
        # 保存对话历史
        try:
            if ai_response_text:
                self.current_message.add_ai_message(ai_response_text)
            if view_message:
                self.current_message.add_view_message(view_message)
            await blocking_func_to_async(
                self._executor, self.current_message.end_current_round
            )
        except Exception as save_error:
            logger.error(f"保存对话历史失败: {save_error}", exc_info=True)

    async def _handle_final_output(
        self,
        final_output: ModelOutput,
        incremental: bool = False,
        check_error: bool = True,
    ):
        text_msg = final_output.text if final_output.has_text else ""
        view_msg = self.stream_plugin_call(text_msg)

        if check_error and self._last_sql_error:
            logger.warning(f"SQL执行失败，跳过总结生成: {self._last_sql_error[:100]}")
            view_msg = final_output.gen_text_with_thinking(new_text=view_msg)
            ai_full_response = self._combine_thinking_and_text(final_output, view_msg)
            return ai_full_response, view_msg

        summary_result = await self._generate_result_summary(text_msg, view_msg)
        
        summary_text = summary_result.get("summary", "") if isinstance(summary_result, dict) else ""
        suggested_questions = summary_result.get("suggested_questions", []) if isinstance(summary_result, dict) else []
        
        # 保存推荐问题到实例变量，供前端获取
        self._current_suggested_questions = suggested_questions

        if summary_text:
            import re

            chart_pattern = r"(<chart-view.*?</chart-view>)"
            chart_matches = re.findall(chart_pattern, view_msg, re.DOTALL)

            if chart_matches:
                all_chart_views = "\n\n".join(chart_matches)
                view_msg = f"{summary_text}\n\n{all_chart_views}"
            else:
                view_msg = f"{summary_text}\n\n{view_msg}"
            
            # 如果有推荐问题，在view_msg末尾添加特殊标记（前端可以解析）
            if suggested_questions:
                questions_json = json.dumps({"suggested_questions": suggested_questions}, ensure_ascii=False)
                view_msg += f"\n\n<!--SUGGESTED_QUESTIONS:{questions_json}-->"

        view_msg = final_output.gen_text_with_thinking(new_text=view_msg)
        ai_full_response = self._combine_thinking_and_text(final_output, view_msg)
        return ai_full_response, view_msg

    def _combine_thinking_and_text(
        self, final_output: ModelOutput, view_msg: str
    ) -> str:
        thinking_text = getattr(final_output, "thinking", None)

        if thinking_text:
            text_content = final_output.text if final_output.has_text else ""
            return f"{thinking_text}\n{text_content}".strip()
        else:
            return final_output.text if final_output.has_text else ""

    async def _handle_general_chat(self):
        """
        处理通用对话（非数据分析问题）
        使用LLM进行简单的对话回复
        """
        try:
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"
            
            if is_english:
                system_prompt = """You are a friendly AI assistant. The user is currently in a data analysis session but has asked a general question unrelated to data analysis. Please provide a brief, helpful response and gently remind them that you're here primarily to help with data analysis tasks."""
            else:
                system_prompt = """你是一个友好的AI助手。用户当前处于数据分析会话中，但提出了一个与数据分析无关的一般性问题。请提供简短、有帮助的回复，并温和地提醒他们你主要是用来帮助数据分析任务的。"""
            
            request = ModelRequest(
                model=self.llm_model,
                messages=[
                    ModelMessage(role=ModelMessageRoleType.SYSTEM, content=system_prompt),
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN,
                        content=self.current_user_input.last_text
                    )
                ],
                temperature=0.7,
                max_new_tokens=512,
                context=ModelRequestContext(stream=True),
            )
            
            if self.llm_client:
                import inspect
                
                stream_response = self.llm_client.generate_stream(request)
                full_text = ""
                
                if inspect.isasyncgen(stream_response):
                    async for chunk in stream_response:
                        try:
                            if hasattr(chunk, "has_text") and chunk.has_text:
                                chunk_text = chunk.text
                            elif hasattr(chunk, "text"):
                                try:
                                    chunk_text = chunk.text
                                except ValueError:
                                    continue
                            else:
                                continue
                            
                            if chunk_text:
                                full_text = chunk_text
                                yield chunk_text
                        except Exception as e:
                            logger.debug(f"获取chunk.text失败: {e}")
                            continue
                elif inspect.isgenerator(stream_response):
                    for chunk in stream_response:
                        try:
                            if hasattr(chunk, "has_text") and chunk.has_text:
                                chunk_text = chunk.text
                            elif hasattr(chunk, "text"):
                                try:
                                    chunk_text = chunk.text
                                except ValueError:
                                    continue
                            else:
                                continue
                            
                            if chunk_text:
                                full_text = chunk_text
                                yield chunk_text
                        except Exception as e:
                            logger.debug(f"获取chunk.text失败: {e}")
                            continue
                
                # 保存对话历史
                if full_text:
                    self.current_message.add_ai_message(full_text)
                    self.current_message.add_view_message(full_text)
                    await blocking_func_to_async(
                        self._executor, self.current_message.end_current_round
                    )
            else:
                if is_english:
                    fallback_msg = "I'm here to help you with data analysis. If you have any questions about the data, feel free to ask!"
                else:
                    fallback_msg = "我是数据分析助手，如果您有关于数据的问题，随时可以问我！"
                yield fallback_msg
                
        except Exception as e:
            logger.error(f"处理通用对话失败: {e}", exc_info=True)
            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"
            
            if is_english:
                error_msg = "I'm here to help with data analysis. Do you have any questions about the data?"
            else:
                error_msg = "我主要负责数据分析，您有什么关于数据的问题吗？"
            yield error_msg
    
    def _calculate_result_statistics(self, result_data: list, is_english: bool = False) -> str:
        """
        计算查询结果的统计信息，特别是数值字段的总和、平均值等
        
        Args:
            result_data: 查询结果数据列表
            is_english: 是否使用英文
            
        Returns:
            格式化的统计信息字符串
        """
        try:
            if not result_data or not isinstance(result_data, list) or len(result_data) == 0:
                return ""
            
            # 识别数值字段（通过字段名或值的类型判断）
            numeric_fields = {}
            sample_row = result_data[0] if result_data else {}
            
            for field_name, field_value in sample_row.items():
                # 判断字段是否为数值类型
                if isinstance(field_value, (int, float)):
                    numeric_fields[field_name] = {
                        'sum': 0,
                        'count': 0,
                        'min': float('inf'),
                        'max': float('-inf'),
                        'values': []
                    }
            
            # 如果没有数值字段，返回空
            if not numeric_fields:
                return ""
            
            # 遍历所有数据行，计算统计值
            for row in result_data:
                for field_name in numeric_fields.keys():
                    value = row.get(field_name)
                    if value is not None and isinstance(value, (int, float)):
                        numeric_fields[field_name]['sum'] += value
                        numeric_fields[field_name]['count'] += 1
                        numeric_fields[field_name]['min'] = min(numeric_fields[field_name]['min'], value)
                        numeric_fields[field_name]['max'] = max(numeric_fields[field_name]['max'], value)
                        numeric_fields[field_name]['values'].append(value)
            
            # 构建统计信息文本
            stats_parts = []
            
            if is_english:
                stats_parts.append("**Statistical Summary** (Pre-calculated, use these exact numbers):")
            else:
                stats_parts.append("**统计汇总**（已为你计算好，如有需要,请直接使用这些数值）：")
            
            for field_name, stats in numeric_fields.items():
                if stats['count'] == 0:
                    continue
                
                avg_value = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
                
                if is_english:
                    field_stats = f"  - {field_name}:"
                    field_stats += f"\n    * Total: {stats['sum']}"
                    field_stats += f"\n    * Average: {avg_value:.2f}"
                    field_stats += f"\n    * Min: {stats['min']}, Max: {stats['max']}"
                    field_stats += f"\n    * Record Count: {stats['count']}"
                else:
                    field_stats = f"  - {field_name}："
                    field_stats += f"\n    * 总计：{stats['sum']}"
                    field_stats += f"\n    * 平均值：{avg_value:.2f}"
                    field_stats += f"\n    * 最小值：{stats['min']}，最大值：{stats['max']}"
                    field_stats += f"\n    * 记录条数：{stats['count']}"
                
                stats_parts.append(field_stats)
            
            if len(stats_parts) <= 1:
                return ""
            
            return "\n".join(stats_parts) + "\n"
            
        except Exception as e:
            logger.warning(f"计算结果统计信息失败: {e}")
            return ""
    
    def _format_query_rewrite_thinking(self, rewrite_result: dict) -> str:
        try:
            if not rewrite_result:
                return ""

            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"

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

            thinking_parts = [title]

            rewritten_query = rewrite_result.get("rewritten_query", "")
            if rewritten_query:
                thinking_parts.append(f"{label_question}{rewritten_query}\n")

            relevant_columns = rewrite_result.get("relevant_columns", [])
            if relevant_columns:
                thinking_parts.append(label_columns)
                for col in relevant_columns[:5]:
                    col_name = col.get("column_name", "")
                    usage = col.get("usage", "")
                    if col_name:
                        thinking_parts.append(f"  • {col_name}")
                        if usage:
                            thinking_parts.append(f"{separator}{usage}")
                        thinking_parts.append("\n")

            analysis_suggestions = rewrite_result.get("analysis_suggestions", [])
            if analysis_suggestions:
                thinking_parts.append(label_suggestions)
                for suggestion in analysis_suggestions[:5]:
                    thinking_parts.append(f"  • {suggestion}\n")

            return "".join(thinking_parts)

        except Exception as e:
            logger.warning(f"格式化Query改写thinking失败: {e}")
            return ""

    async def _generate_result_summary_stream(
        self, original_text: str, view_msg: str, text_output: bool = True
    ):
        """
        流式生成结果总结
        
        Yields:
            str: 流式输出的summary文本（只输出summary部分，不输出suggested_questions）
            Dict: 最终完整结果（包含summary和suggested_questions）
        """
        try:
            import html
            import json
            import re

            chart_pattern = r'<chart-view content="([^"]+)">'
            matches = re.findall(chart_pattern, view_msg)

            all_sql_results = []
            total_result_count = 0
            
            for match_str in matches:
                content_str = html.unescape(match_str)
                content_data = json.loads(content_str)

                sql = content_data.get("sql", "").strip()
                query_data = content_data.get("data", [])
                total_result_count += len(query_data) if isinstance(query_data, list) else 0

                all_sql_results.append({"sql": sql, "result": query_data})

            # 如果没有找到chart-view标签，尝试从原始文本中提取SQL
            if not all_sql_results:
                api_call_pattern = r'<api-call>.*?<sql>(.*?)</sql>.*?</api-call>'
                api_matches = re.findall(api_call_pattern, original_text, re.DOTALL)
                if api_matches:
                    for sql in api_matches:
                        sql = sql.strip()
                        if sql:
                            all_sql_results.append({"sql": sql, "result": []})

            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"
            
            # 检查数据量是否过大（超过20条记录），如果过大则使用轻量级prompt
            use_lightweight_summary = total_result_count > 40
            
            # 提前初始化 selected_tables 和 is_multi_table，以便后续使用
            selected_tables = None
            is_multi_table = False
            
            # 获取 select_param_dict 以确定是否为多表模式
            select_param_dict = self.select_param
            if isinstance(self.select_param, str):
                try:
                    select_param_dict = json.loads(self.select_param)
                except Exception:
                    select_param_dict = {}
            
            if isinstance(select_param_dict, dict):
                # 获取选中的表列表
                selected_tables = getattr(self, "_selected_tables", None)
                if not selected_tables:
                    if getattr(self, "_is_multi_table_mode", False) and hasattr(self, "_all_table_names"):
                        selected_tables = self._all_table_names
                    else:
                        selected_tables = [self._curr_table]
                
                is_multi_table = len(selected_tables) > 1
            else:
                # 如果 select_param_dict 不是字典，使用默认值
                selected_tables = [self._curr_table]
                is_multi_table = False
            
            # 构建多表场景的引导信息
            multi_table_guidance = ""
            if is_multi_table and selected_tables:
                table_names_str = "、".join(selected_tables) if not is_english else ", ".join(selected_tables)
                if is_english:
                    multi_table_guidance = f"""

**IMPORTANT - Multi-Table Scenario**:
This data analysis session involves multiple data tables: {table_names_str}.

**Special Requirements for Recommended Questions**:
- **Questions MUST explicitly specify which table(s) to use**, avoid vague expressions
- **Good Examples** (explicitly specify tables):
  * "Who are in {selected_tables[0]}" (specify a specific table)
  * "Who are all employees" (use "all", "total", "overall" to indicate querying all tables)
  * "Who has the highest bonus in {selected_tables[0]}" (specify a specific table)
- **Bad Examples** (no table specified):
  * "Who are there" (too vague, no table specified)
  * "Who has the highest bonus" (no table specified, unclear)
- **Question Distribution**:
  * Include some questions specific to individual tables (explicitly mention table names)
  * Include some questions involving "all" or "total" tables (use words like "all", "total", "overall")

"""
                else:
                    multi_table_guidance = f"""

**重要提示 - 多表场景**：
当前数据分析会话涉及多个数据表：{table_names_str}。

**推荐问题的特殊要求**：
- **必须在问题中明确指出要使用哪个表或哪些表**，避免模糊表达

- **推荐问题的分配建议**：
  * 可以包含几个针对特定表的问题（明确提到表名）
  * 可以包含几个涉及"所有"或"全部"表的问题（使用"所有"、"全部"、"总共"等词）

"""

            history_context = ""
            if self.history_messages and len(self.history_messages) > 0:
                history_context = "\n=== 历史对话 ===\n"
                for msg in self.history_messages[-9:]:
                    if not hasattr(msg, "content"):
                        continue

                    role = getattr(msg, "role", "user")
                    content = msg.content

                    if hasattr(content, "get_text"):
                        try:
                            content = content.get_text()
                        except Exception:
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

                    content = self._clean_history_content(content)

                    detected_language = getattr(self, "_detected_language", "zh")
                    is_english = detected_language == "en"
                    
                    # 正确识别角色
                    if role == "human":
                        role_display = "User" if is_english else "用户"
                    else:  # role == "ai" or "assistant" or "view"
                        role_display = "Assistant" if is_english else "助手"
                    
                    history_context += f"{role_display}: {content}\n\n"

            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"

            sql_results_text = ""
            if all_sql_results:
                for i, sql_result in enumerate(all_sql_results, 1):
                    sql_label = f"Executed SQL {i}" if is_english else f"执行的SQL {i}"
                    sql_results_text += f"\n{sql_label}：\n{sql_result['sql']}\n\n"
                    
                    result_data = sql_result["result"]
                    result_count = len(result_data) if isinstance(result_data, list) else 0
                    
                    # 计算统计信息（数值字段的总和、平均值等）
                    stats_info = self._calculate_result_statistics(result_data, is_english)
                    
                    if use_lightweight_summary:
                        # 数据量过大时，只传入摘要信息
                        if result_count > 0:
                            # 获取字段名
                            field_names = list(result_data[0].keys()) if result_data else []
                            if is_english:
                                sql_results_text += f"Query Result: {result_count} records returned\n"
                                sql_results_text += f"Fields: {', '.join(field_names)}\n"
                            else:
                                sql_results_text += f"查询结果：返回 {result_count} 条记录\n"
                                sql_results_text += f"包含字段：{', '.join(field_names)}\n"
                            
                            # 添加统计信息
                            if stats_info:
                                sql_results_text += stats_info
                        else:
                            if is_english:
                                sql_results_text += "Query Result: No records found (empty result set)\n"
                            else:
                                sql_results_text += "查询结果：未找到匹配的数据（结果为空）\n"
                    else:
                        # 数据量较小时，传入完整数据
                        result_label = f"Query Result {i}" if is_english else f"查询结果 {i}"
                        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                        sql_results_text += f"{result_label}：\n{result_json}\n"
                        
                        # 添加统计信息
                        if stats_info:
                            sql_results_text += f"\n{stats_info}\n"
                        
                        if not result_data:
                            if is_english:
                                sql_results_text += "\n**Note**: The query returned no results (empty result set).\n"
                            else:
                                sql_results_text += "\n**注意**：查询结果为空（未找到匹配的数据）。\n"
            else:
                if is_english:
                    sql_results_text = "\n**Note**: No SQL query was executed or no query results were found.\n"
                else:
                    sql_results_text = "\n**注意**：未执行SQL查询或未找到查询结果。\n"

            # 获取数据schema信息，用于生成推荐问题
            data_schema_info = ""
            # select_param_dict 已在前面获取，这里直接使用
            
            if isinstance(select_param_dict, dict):
                # selected_tables 和 is_multi_table 已在前面初始化，这里直接使用
                all_tables_info = select_param_dict.get("all_tables_info", [])
                
                if is_multi_table and all_tables_info:
                    # 多表模式：构建包含所有选中表的信息
                    data_schema_info = self._build_multi_table_schema_info_for_summary(
                        selected_tables, all_tables_info, is_english
                    )
                else:
                    # 单表模式：使用原有逻辑
                    data_schema_json = select_param_dict.get("data_schema_json")
                    if data_schema_json:
                        try:
                            schema_obj = (
                                json.loads(data_schema_json)
                                if isinstance(data_schema_json, str)
                                else data_schema_json
                            )
                            table_description = schema_obj.get("table_description", "")
                            columns_summary = []
                            # 包含所有列，不限制数量
                            for col in schema_obj.get("columns", []):
                                col_name = col.get("column_name", "")
                                description = col.get("description", "")
                                domain_knowledge = col.get("domain_knowledge", "")
                                
                                # 构建列信息文本
                                col_info_text = f"- {col_name}: {description}"
                                
                                # 如果有领域知识，添加到描述中
                                if domain_knowledge:
                                    if is_english:
                                        col_info_text += f" [Key Knowledge: {domain_knowledge}]"
                                    else:
                                        col_info_text += f" [关键知识: {domain_knowledge}]"
                                
                                # 如果有unique_values_top20，补充显示前5个值
                                if "unique_values_top20" in col:
                                    unique_vals = col["unique_values_top20"]
                                    if isinstance(unique_vals, list) and len(unique_vals) > 0:
                                        # 取前5个值
                                        top_5_values = unique_vals[:5]
                                        if is_english:
                                            col_info_text += f" (Example values: {', '.join(map(str, top_5_values))})"
                                        else:
                                            col_info_text += f" (示例值: {', '.join(map(str, top_5_values))})"
                                
                                columns_summary.append(col_info_text)
                            
                            # 根据语言切换标签
                            if is_english:
                                data_schema_info = f"""
=== Data Table Information ===
Table Description: {table_description}
All Columns:
{chr(10).join(columns_summary)}
"""
                            else:
                                data_schema_info = f"""
=== 数据表信息 ===
表描述: {table_description}
所有字段:
{chr(10).join(columns_summary)}
"""
                        except Exception as e:
                            logger.warning(f"解析data_schema_json失败: {e}")

            if use_lightweight_summary:
                # 数据量过大时，使用简化的prompt（不传入完整数据，但仍生成推荐问题）
                # 构建数据来源说明
                table_source_info = ""
                if is_multi_table and selected_tables:
                    table_names_str = ", ".join(selected_tables) if is_english else "、".join(selected_tables)
                    if is_english:
                        table_source_info = f"\n**Data Source**: Queried from {len(selected_tables)} tables: {table_names_str}\n"
                    else:
                        table_source_info = f"\n**数据来源**：查询了 {len(selected_tables)} 个表的数据：{table_names_str}\n"
                elif selected_tables and len(selected_tables) == 1:
                    if is_english:
                        table_source_info = f"\n**Data Source**: Queried from table: {selected_tables[0]}\n"
                    else:
                        table_source_info = f"\n**数据来源**：查询了表 {selected_tables[0]} 的数据\n"
                
                if is_english:
                    summary_prompt = f"""User's Question: {self.current_user_input.last_text}
{sql_results_text}
{table_source_info}{data_schema_info}{multi_table_guidance}
Task: Generate:
1. A brief, natural introduction sentence for the query results (MUST mention which table(s) were queried)
2. 9 follow-up questions based on the data schema

Output Format:
```json
{{
  "summary": "Brief introduction mentioning data source (under 40 words, no data analysis)",
  "suggested_questions": [
    "Question 1 (simple)", "Question 2 (simple)", "Question 3 (simple)",
    "Question 4 (simple)", "Question 5 (simple)", "Question 6 (simple)",
    "Question 7 (medium)", "Question 8 (medium)", "Question 9 (medium)"
  ]
}}
```

Requirements:
- Summary: naturally introduce the data results, MUST clearly state which table(s) were queried, concise, NO analysis
- Questions: first 6 simple with standard answers, last 3 medium difficulty
- All questions must be based on actual fields in the data table
- **If multi-table scenario: Questions MUST explicitly specify which table(s) to use**
- Must be in ENGLISH

Output JSON only:"""
                else:
                    # 构建数据来源说明
                    table_source_info = ""
                    if is_multi_table and selected_tables:
                        table_names_str = "、".join(selected_tables)
                        table_source_info = f"\n**数据来源**：查询了 {len(selected_tables)} 个表的数据：{table_names_str}\n"
                    elif selected_tables and len(selected_tables) == 1:
                        table_source_info = f"\n**数据来源**：查询了表 {selected_tables[0]} 的数据\n"
                    
                    summary_prompt = f"""用户问题：{self.current_user_input.last_text}
{sql_results_text}
{table_source_info}{data_schema_info}{multi_table_guidance}
任务：生成：
1. 一句简短、自然的引导语来介绍查询结果（必须说明查询了哪些表）
2. 9个基于数据表的推荐问题

输出格式：
```json
{{
  "summary": "简短引导语，需明确说明数据来源（不超过40字，不要分析数据）",
  "suggested_questions": [
    "问题1（简单）", "问题2（简单）", "问题3（简单）",
    "问题4（简单）", "问题5（简单）", "问题6（简单）",
    "问题7（中等）", "问题8（中等）", "问题9（中等）"
  ]
}}
```

要求：
- 引导语：自然引出数据结果，必须清楚说明查询了哪些表的数据，简洁，不要分析数据
- 问题：前6个简单问题有标准答案，后3个中等难度
- 所有问题必须基于数据表中的实际字段
- **如果是多表场景：必须在问题中明确指出要使用哪个表或哪些表**
- 必须使用中文

直接输出JSON："""
            elif is_english:
                # 构建数据来源说明
                table_source_info = ""
                if is_multi_table and selected_tables:
                    table_names_str = ", ".join(selected_tables)
                    table_source_info = f"\n**Data Source**: Queried from {len(selected_tables)} tables: {table_names_str}\n"
                elif selected_tables and len(selected_tables) == 1:
                    table_source_info = f"\n**Data Source**: Queried from table: {selected_tables[0]}\n"
                
                summary_prompt = f"""{history_context}
=== User's Current Question ===
{self.current_user_input.last_text}
{sql_results_text}
{table_source_info}{data_schema_info}{multi_table_guidance}
**IMPORTANT - Language Requirement**:
- The user's question is in ENGLISH
- You MUST respond in ENGLISH
- Your answer MUST be in ENGLISH, not Chinese

**Task**:
Based on the conversation history, current question, SQL query results, and data schema information above, please generate:
1. An objective and accurate summary (MUST mention which table(s) were queried)
2. 9 follow-up questions that would help users explore the data further based on the current analysis results

**Output Format**:
Please output a JSON object with the following structure:
```json
{{
  "summary": "Your objective summary based on SQL logic, clearly stating which table(s) were queried",
  "suggested_questions": [
    "Question 1 (simple question with standard answer)",
    "Question 2 (simple question with standard answer)",
    "Question 3 (simple question with standard answer)",
    "Question 4 (simple question with standard answer)",
    "Question 5 (simple question with standard answer)",
    "Question 6 (simple question with standard answer)",
    "Question 7 (open-ended question)",
    "Question 8 (open-ended question)",
    "Question 9 (open-ended question)"
  ]
}}
```

**Requirements for summary**:
- Explain the final query result
- **MUST clearly state which table(s) were queried** (e.g., "Based on data from table X..." or "Queried data from tables X and Y...")
- **IMPORTANT - Use Pre-calculated Statistics**:
  * If "Statistical Summary" is provided above, you MUST use those exact numbers directly
  * DO NOT calculate totals, averages, or any statistics yourself
  * DO NOT add up numbers from individual records - use the provided totals
  * The statistical values have been pre-calculated for accuracy
- **Constraints**:
  * Do NOT speculate, extrapolate, or provide subjective interpretations
- **Output Requirements**:
  * One objective and accurate summary
  * Must be in **ENGLISH**

**Requirements for suggested_questions**:
- **First 6 questions**: Simple questions with clear standard answers 
- **Last 3 questions**: Medium difficulty questions that require some thinking and analysis
- **IMPORTANT**: All questions MUST be based on actual fields and data in the data table, DO NOT fabricate non-existent fields or data
- **If multi-table scenario: Questions MUST explicitly specify which table(s) to use**
- Questions should be based on the current analysis results and conversation context, you can think more broadly, but do not deviate from the topic
- All questions must be in ENGLISH

Please output the JSON directly, without any other text:"""  # noqa: E501
            else:
                # 构建数据来源说明
                table_source_info = ""
                if is_multi_table and selected_tables:
                    table_names_str = "、".join(selected_tables)
                    table_source_info = f"\n**数据来源**：查询了 {len(selected_tables)} 个表的数据：{table_names_str}\n"
                elif selected_tables and len(selected_tables) == 1:
                    table_source_info = f"\n**数据来源**：查询了表 {selected_tables[0]} 的数据\n"
                
                summary_prompt = f"""{history_context}
=== 用户当前问题 ===
{self.current_user_input.last_text}
{sql_results_text}
{table_source_info}{data_schema_info}{multi_table_guidance}
**重要 - 语言要求**：
- 用户的问题是**中文**
- 你必须用**中文**回答

**任务**：
根据上述历史对话、当前问题、SQL查询结果和数据表信息，请生成：
1. 一段客观、准确的总结（必须说明查询了哪些表）
2. 9个基于当前分析结果的推荐问题，帮助用户进一步探索数据

**输出格式**：
请输出一个JSON对象，格式如下：
```json
{{
  "summary": "基于SQL逻辑的客观总结，需明确说明查询了哪些表",
  "suggested_questions": [
    "问题1（简单问题，有标准答案）",
    "问题2（简单问题，有标准答案）",
    "问题3（简单问题，有标准答案）",
    "问题4（简单问题，有标准答案）",
    "问题5（简单问题，有标准答案）",
    "问题6（简单问题，有标准答案）",
    "问题7（中等难度问题）",
    "问题8（中等难度问题）",
    "问题9（中等难度问题）"
  ]
}}
```

**总结的要求**：
- 阐述最终查询结果
- **必须清楚说明查询了哪些表的数据**（例如："根据表X的数据..."或"查询了表X和表Y的数据..."）
- **重要 - 使用预计算的统计值**：
  * 如果上面提供了"统计汇总"，你必须直接使用这些准确的数值
  * 不要自己计算总数、平均值或任何统计值
  * 不要把各条记录的数字相加 - 直接使用提供的总计值
  * 这些统计值已经预先计算好，确保准确性
- **约束条件**：
  * 不要进行推测、延伸或主观解读
- **输出要求**：
  * 一段客观、准确的总结
  * 必须使用**中文**

**推荐问题的要求**：
- **前6个问题**：简单的问题，有明确的标准答案
- **后3个问题**：中等难度问题，需要一定的思考和分析
- **重要**：所有问题必须基于数据表中的实际字段和数据，可以围绕具体的分类值进行分析，不能凭空捏造不存在的字段或数据
- **如果是多表场景：必须在问题中明确指出要使用哪个表或哪些表**
- 问题应该基于当前分析结果和对话上下文，可以适当发散思维，但不要偏离主题
- 所有问题必须使用**中文**

请直接输出JSON，不要有其他文字："""

            summary_request = ModelRequest(
                model=self.llm_model,
                messages=[
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN, content=summary_prompt
                    )
                ],
                temperature=0.3,
                max_new_tokens=2048,
                context=ModelRequestContext(stream=True),
            )

            if self.llm_client:
                import inspect
                
                # 获取流式响应
                stream_response = self.llm_client.generate_stream(summary_request)
                
                full_text = ""
                previous_summary = ""
                
                if inspect.isasyncgen(stream_response):
                    async for chunk in stream_response:
                        chunk_text = ""
                        try:
                            if hasattr(chunk, "has_text") and chunk.has_text:
                                chunk_text = chunk.text
                            elif hasattr(chunk, "text"):
                                try:
                                    chunk_text = chunk.text
                                except ValueError:
                                    continue
                        except Exception as e:
                            logger.debug(f"获取chunk.text失败: {e}")
                            continue
                        
                        if chunk_text:
                            full_text = chunk_text
                            
                            # 尝试从流式JSON中提取summary部分
                            try:
                                # 查找 "summary" 字段的开始位置
                                summary_start = full_text.find('"summary"')
                                if summary_start >= 0:
                                    # 找到summary字段值开始的位置（冒号后的引号）
                                    value_start = full_text.find('"', summary_start + len('"summary"'))
                                    if value_start >= 0:
                                        value_start += 1  # 跳过开始引号
                                        # 查找summary值的结束位置（考虑转义字符）
                                        value_end = value_start
                                        while value_end < len(full_text):
                                            if full_text[value_end] == '"' and full_text[value_end - 1] != '\\':
                                                break
                                            value_end += 1
                                        
                                        if value_end > value_start:
                                            current_summary = full_text[value_start:value_end]
                                            # 输出完整的summary内容（逐渐增长）
                                            if len(current_summary) > len(previous_summary):
                                                previous_summary = current_summary
                                                
                                                if text_output:
                                                    yield current_summary
                                                else:
                                                    yield ModelOutput.build(
                                                        text=current_summary,
                                                        error_code=0,
                                                        finish_reason="continue"
                                                    )
                            except Exception as parse_err:
                                # 如果解析失败，继续等待更多内容
                                logger.debug(f"解析summary流式输出失败: {parse_err}")
                                continue
                elif inspect.isgenerator(stream_response):
                    for chunk in stream_response:
                        chunk_text = ""
                        try:
                            if hasattr(chunk, "has_text") and chunk.has_text:
                                chunk_text = chunk.text
                            elif hasattr(chunk, "text"):
                                try:
                                    chunk_text = chunk.text
                                except ValueError:
                                    continue
                        except Exception as e:
                            logger.debug(f"获取chunk.text失败: {e}")
                            continue
                        
                        if chunk_text:
                            full_text = chunk_text
                            
                            # 尝试从流式JSON中提取summary部分
                            try:
                                summary_start = full_text.find('"summary"')
                                if summary_start >= 0:
                                    value_start = full_text.find('"', summary_start + len('"summary"'))
                                    if value_start >= 0:
                                        value_start += 1
                                        value_end = value_start
                                        while value_end < len(full_text):
                                            if full_text[value_end] == '"' and full_text[value_end - 1] != '\\':
                                                break
                                            value_end += 1
                                        
                                        if value_end > value_start:
                                            current_summary = full_text[value_start:value_end]
                                            # 输出完整的summary内容（逐渐增长）
                                            if len(current_summary) > len(previous_summary):
                                                previous_summary = current_summary
                                                
                                                if text_output:
                                                    yield current_summary
                                                else:
                                                    yield ModelOutput.build(
                                                        text=current_summary,
                                                        error_code=0,
                                                        finish_reason="continue"
                                                    )
                            except Exception as parse_err:
                                logger.debug(f"解析summary流式输出失败: {parse_err}")
                                continue
                
                # 流式输出完成后，解析完整JSON并返回结果
                if full_text:
                    try:
                        # 提取JSON部分
                        json_str = full_text
                        if "```json" in json_str.lower():
                            start_idx = json_str.lower().find("```json")
                            if start_idx >= 0:
                                content_start = json_str.find("\n", start_idx) + 1
                                if content_start > 0:
                                    end_idx = json_str.find("```", content_start)
                                    if end_idx > content_start:
                                        json_str = json_str[content_start:end_idx].strip()
                        
                        if "```" in json_str and "```json" not in json_str.lower():
                            start_idx = json_str.find("```")
                            content_start = json_str.find("\n", start_idx) + 1
                            if content_start > 0:
                                end_idx = json_str.find("```", content_start)
                                if end_idx > content_start:
                                    json_str = json_str[content_start:end_idx].strip()
                        
                        # 尝试提取JSON对象
                        if "{" in json_str and "}" in json_str:
                            start = json_str.find("{")
                            end = json_str.rfind("}") + 1
                            if start >= 0 and end > start:
                                json_str = json_str[start:end]
                        
                        result = json.loads(json_str)
                        summary_text = result.get("summary", "")
                        suggested_questions = result.get("suggested_questions", [])
                        
                        if not summary_text:
                            summary_text = full_text
                        
                        yield {
                            "summary": summary_text,
                            "suggested_questions": suggested_questions[:9] if isinstance(suggested_questions, list) else [],
                        }
                        return
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析总结JSON失败: {e}，使用原始文本")
                        yield {
                            "summary": full_text,
                            "suggested_questions": [],
                        }
                        return
            else:
                logger.warning("llm_client未初始化，无法生成总结")

            yield {"summary": "", "suggested_questions": []}

        except Exception as e:
            logger.warning(f"生成结果总结失败: {e}", exc_info=True)
            yield {"summary": "", "suggested_questions": []}
