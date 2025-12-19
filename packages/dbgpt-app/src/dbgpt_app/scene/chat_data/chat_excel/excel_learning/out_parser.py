import json
import logging
from typing import Dict, List, NamedTuple

from dbgpt.core.interface.output_parser import BaseOutputParser
from dbgpt_app.scene.chat_data.chat_excel.excel_reader import TransformedExcelResponse


class ExcelResponse(NamedTuple):
    desciption: str
    clounms: List
    plans: List


logger = logging.getLogger(__name__)


class LearningExcelOutputParser(BaseOutputParser):
    def __init__(self, is_stream_out: bool = False, **kwargs):
        super().__init__(is_stream_out=is_stream_out, **kwargs)
        self.is_downgraded = False

    def parse_prompt_response(self, model_out_text):
        description = ""
        columns = []
        plans = []
        try:
            clean_str = super().parse_prompt_response(model_out_text)
            # 只记录解析结果的简要信息，不输出完整内容
            logger.debug(f"parse_prompt_response: 解析完成，长度={len(model_out_text)}")
            response = json.loads(clean_str)
            for key in sorted(response):
                if key.strip() == "data_analysis":
                    description = response[key]
                if key.strip() == "column_analysis":
                    columns = response[key]
                if key.strip() == "analysis_program":
                    plans = response[key]
            logger.info(f"✅ Excel Schema 解析成功: {len(columns)} 列")
            return TransformedExcelResponse(
                description=description, columns=columns, plans=plans
            )
        except Exception as e:
            logger.error(f"parse_prompt_response failed: {e}")
            for name in self.data_schema:
                columns.append({name: "-"})
            return TransformedExcelResponse(
                description=model_out_text, columns=columns, plans=plans
            )

    def _build_columns_html(self, columns: List[Dict[str, str]]) -> str:
        html_columns = "### **Data Structure**\n"
        column_index = 0
        for item in columns:
            column_index += 1
            column_name = item.get("new_column_name", "")
            old_column_name = item.get("old_column_name", "")
            column_description = item.get("column_description", "")
            html_columns += (
                f"- **{column_index}. {column_name}({old_column_name})**: "
                f"_{column_description}_\n"
            )
        return html_columns

    def __build_plans_html(self, plans_data):
        html_plans = "### **Analysis plans**\n"
        index = 0
        if plans_data:
            for item in plans_data:
                index += 1
                html_plans = html_plans + f"{item} \n"
        return html_plans

    def parse_view_response(
        self, speak, data: TransformedExcelResponse, prompt_response
    ) -> str:
        # 不再展示 Data Summary 给前端，只保存到后台缓存
        # 返回简单的成功消息即可
        if data and not isinstance(data, str):
            return "数据分析结构已生成并缓存，可以开始查询数据。"
        else:
            return speak
