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
                        cursor.execute(
                            "SELECT data_schema_json FROM excel_metadata WHERE content_hash = ?",
                            (self._content_hash,),
                        )
                        result = cursor.fetchone()
                        conn.close()

                        if result and result[0]:
                            select_param_dict["data_schema_json"] = result[0]
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

        if use_existing_db and duckdb_path:
            self._curr_table = duckdb_table_name or "data_analysis_table"
            self.excel_reader = self._create_reader_from_duckdb(
                chat_param.chat_session_id,
                duckdb_path,
                file_name,
                duckdb_table_name,
            )
        else:
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
            database_file_path = os.path.join(
                database_root_path, f"_chat_excel_{file_name}.duckdb"
            )
            database_file_id = None

        if file_path.startswith(_SCHEMA):
            file_path, file_meta = fs_client.download_file(file_path, dest_dir=DATA_DIR)
            file_name = os.path.basename(file_path)

            if not duckdb_path:
                database_file_path = os.path.join(
                    database_root_path, f"_chat_excel_{file_name}.duckdb"
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
        if (
            hasattr(self, "_cached_input_values")
            and self._cached_input_values is not None
        ):
            return self._cached_input_values

        await self._ensure_data_analysis_table_exists()

        user_input = self.current_user_input.last_text
        detected_language = detect_language(user_input)
        self._detected_language = detected_language

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        prompt_templates = get_prompt_templates_by_language(detected_language)

        table_schema = await blocking_func_to_async(
            self._executor, self.excel_reader.get_create_table_sql, self._curr_table
        )
        colunms, datas = await blocking_func_to_async(
            self._executor, self.excel_reader.get_sample_data, self._curr_table
        )

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
                try:
                    from dbgpt_app.scene.chat_data.chat_excel.query_rewrite import (
                        QueryRewriteAgent,
                    )

                    rewrite_agent = QueryRewriteAgent(self.llm_client, self.llm_model)

                    chat_history = []
                    if hasattr(self, "history_messages") and self.history_messages:
                        current_round_messages = []
                        last_role = None

                        for msg in self.history_messages:
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

                    rewrite_result = await blocking_func_to_async(
                        self._executor,
                        rewrite_agent.rewrite_query,
                        self.current_user_input.last_text,
                        data_schema_json,
                        table_schema,
                        chat_history,
                    )

                    if rewrite_result and rewrite_result.get("rewritten_query"):
                        query_rewrite_info = f"""


用户的问题：{rewrite_result["rewritten_query"]}

相关字段：
{self._format_relevant_columns(rewrite_result.get("relevant_columns", []))}

分析建议：
{self._format_analysis_suggestions(rewrite_result.get("analysis_suggestions", []))}

分析逻辑：
{rewrite_result.get("analysis_logic", "")}

接下来请按照格式要求生成sql语句进行查询。
"""
                        self._query_rewrite_result = rewrite_result

                        extracted_knowledge = rewrite_result.get("_extracted_knowledge")
                        if extracted_knowledge:
                            await self._save_domain_knowledge(
                                extracted_knowledge, data_schema_json
                            )

                        try:
                            schema_obj = (
                                json.loads(data_schema_json)
                                if isinstance(data_schema_json, str)
                                else data_schema_json
                            )
                            all_columns = schema_obj.get("columns", [])

                            relevant_col_names = [
                                col.get("column_name", "")
                                for col in rewrite_result.get("relevant_columns", [])
                            ]

                            relevant_columns_details = []
                            for col_name in relevant_col_names:
                                for col_info in all_columns:
                                    if col_info.get("column_name") == col_name:
                                        relevant_columns_details.append(col_info)
                                        break

                            if relevant_columns_details:
                                relevant_columns_info = (
                                    self._format_relevant_columns_for_prompt(
                                        relevant_columns_details
                                    )
                                )
                            else:
                                relevant_columns_info = "未找到相关列的详细信息。"

                        except Exception as col_err:
                            logger.warning(f"提取列详细信息失败: {col_err}")
                            relevant_columns_info = ""

                except Exception as e:
                    logger.warning(f"Query改写失败，使用原始问题: {e}")
                    self._query_rewrite_result = None

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            _ANALYSIS_CONSTRAINTS_TEMPLATE,
        )

        analysis_constraints = _ANALYSIS_CONSTRAINTS_TEMPLATE.format(
            table_name=self._curr_table, display_type=self._generate_numbered_list()
        )

        input_values = {
            "user_input": self.current_user_input.last_text,
            "table_name": self._curr_table,
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

        from dbgpt_app.scene.chat_data.chat_excel.excel_analyze.prompt import (
            get_prompt_templates_by_language,
        )

        prompt_templates = get_prompt_templates_by_language(detected_language)

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

    def _clean_history_content(self, content: str) -> str:
        import re

        content = re.sub(
            r"<chart-view[^>]*>.*?</chart-view>", "", content, flags=re.DOTALL
        )

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

            if self._curr_table in table_names:
                return

            if (
                hasattr(self.excel_reader, "table_name")
                and self.excel_reader.table_name
            ):
                source_table = self.excel_reader.table_name
                if source_table in table_names and source_table != self._curr_table:
                    sql = (
                        f"CREATE TABLE {self._curr_table} "
                        f"AS SELECT * FROM {source_table};"
                    )
                    self.excel_reader.db.sql(sql)
                    return

            if "temp_table" in table_names:
                sql = f"CREATE TABLE {self._curr_table} AS SELECT * FROM temp_table;"
                self.excel_reader.db.sql(sql)
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
            return result

    async def stream_call(self, text_output: bool = True, incremental: bool = False):
        await self.generate_input_values()

        if self._query_rewrite_result:
            thinking_stage1 = self._format_query_rewrite_thinking(
                self._query_rewrite_result
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

        payload = await self._build_model_request()
        self._last_sql_error = None
        full_output = await self.call_llm_operator(payload)

        ai_response_text = ""
        view_message = ""
        
        if full_output:
            try:
                ai_response_text, view_message = await self._handle_final_output(
                    full_output, incremental=incremental
                )

                final_output_parts = []

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

                final_output_parts.append(view_message)
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

    async def _generate_result_summary(
        self, original_text: str, view_msg: str
    ) -> Dict[str, Any]:
        try:
            import html
            import json
            import re

            chart_pattern = r'<chart-view content="([^"]+)">'
            matches = re.findall(chart_pattern, view_msg)

            if not matches:
                return ""

            all_sql_results = []
            for match_str in matches:
                content_str = html.unescape(match_str)
                content_data = json.loads(content_str)

                sql = content_data.get("sql", "").strip()
                query_data = content_data.get("data", [])

                if query_data:
                    all_sql_results.append({"sql": sql, "result": query_data})

            if not all_sql_results:
                return ""

            history_context = ""
            if self.history_messages and len(self.history_messages) > 0:
                history_context = "\n=== 历史对话 ===\n"
                for msg in self.history_messages[-6:]:
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
                    role_display = (
                        "User"
                        if role == "human"
                        else (
                            "Assistant"
                            if is_english
                            else "用户"
                            if role == "human"
                            else "助手"
                        )
                    )
                    history_context += f"{role_display}: {content}\n\n"

            detected_language = getattr(self, "_detected_language", "zh")
            is_english = detected_language == "en"

            sql_results_text = ""
            for i, sql_result in enumerate(all_sql_results, 1):
                sql_label = f"Executed SQL {i}" if is_english else f"执行的SQL {i}"
                result_label = f"Query Result {i}" if is_english else f"查询结果 {i}"
                sql_results_text += f"\n{sql_label}：\n{sql_result['sql']}\n\n"
                result_json = json.dumps(
                    sql_result["result"], ensure_ascii=False, indent=2
                )
                sql_results_text += f"{result_label}：\n{result_json}\n"

            # 获取数据schema信息，用于生成推荐问题
            data_schema_info = ""
            select_param_dict = self.select_param
            if isinstance(self.select_param, str):
                try:
                    select_param_dict = json.loads(self.select_param)
                except Exception:
                    select_param_dict = {}
            
            if isinstance(select_param_dict, dict):
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
                            
                            # 构建列信息文本
                            col_info_text = f"- {col_name}: {description}"
                            
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

            if is_english:
                summary_prompt = f"""{history_context}
=== User's Current Question ===
{self.current_user_input.last_text}
{sql_results_text}
{data_schema_info}
**IMPORTANT - Language Requirement**:
- The user's question is in ENGLISH
- You MUST respond in ENGLISH
- Your answer MUST be in ENGLISH, not Chinese

**Task**:
Based on the conversation history, current question, SQL query results, and data schema information above, please generate:
1. A concise summary answering the user's current question (at least 100 words)
2. 9 follow-up questions that would help users explore the data further based on the current analysis results

**Output Format**:
Please output a JSON object with the following structure:
```json
{{
  "summary": "Your concise summary answering the user's question",
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

**Requirements for suggested_questions**:
- **First 6 questions**: Simple questions with clear standard answers 
- **Last 3 questions**: Medium difficulty questions that require some thinking and analysis
- **IMPORTANT**: All questions MUST be based on actual fields and data in the data table, DO NOT fabricate non-existent fields or data
- Questions should be based on the current analysis results and conversation context
- All questions must be in ENGLISH

Please output the JSON directly, without any other text:"""  # noqa: E501
            else:
                summary_prompt = f"""{history_context}
=== 用户当前问题 ===
{self.current_user_input.last_text}
{sql_results_text}
{data_schema_info}
**重要 - 语言要求**：
- 用户的问题是**中文**
- 你必须用**中文**回答

**任务**：
根据上述历史对话、当前问题、SQL查询结果和数据表信息，请生成：
1. 一句话总结，完整回答用户的当前问题(至少100字)
2. 9个基于当前分析结果的推荐问题，帮助用户进一步探索数据

**输出格式**：
请输出一个JSON对象，格式如下：
```json
{{
  "summary": "您的一句话总结，完整回答用户的问题",
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

**推荐问题的要求**：
- **前6个问题**：简单的问题，有明确的标准答案
- **后3个问题**：中等难度问题，需要一定的思考和分析
- **重要**：所有问题必须基于数据表中的实际字段和数据，可以围绕具体的分类值进行分析，不能凭空捏造不存在的字段或数据
- 问题应该基于当前分析结果和对话上下文
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
            )

            if self.llm_client:
                summary_output = await self.llm_client.generate(summary_request)
                if summary_output and summary_output.text:
                    output_text = summary_output.text.strip()
                    logger.info(f"生成结果总结和推荐问题: {output_text[:200]}...")
                    
                    # 解析JSON响应
                    try:
                        # 提取JSON部分
                        json_str = output_text
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
                            # 如果没有summary，使用原始输出
                            summary_text = output_text
                        
                        return {
                            "summary": summary_text,
                            "suggested_questions": suggested_questions[:9] if isinstance(suggested_questions, list) else [],
                        }
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析总结JSON失败: {e}，使用原始文本")
                        # 如果解析失败，返回原始文本作为summary
                        return {
                            "summary": output_text,
                            "suggested_questions": [],
                        }
            else:
                logger.warning("llm_client未初始化，无法生成总结")

            return {"summary": "", "suggested_questions": []}

        except Exception as e:
            logger.warning(f"生成结果总结失败: {e}", exc_info=True)
            return {"summary": "", "suggested_questions": []}
