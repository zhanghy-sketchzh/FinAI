"""自定义 Ollama 代理客户端，支持非标准的 Ollama API。"""

import json
import logging
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Iterator, Optional, Type, Union

import requests

from dbgpt.core import MessageConverter, ModelMetadata, ModelOutput, ModelRequest
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    ResourceCategory,
    auto_register_resource,
)
from dbgpt.core.interface.parameter import LLMDeployModelParameters
from dbgpt.model.proxy.base import (
    AsyncGenerateStreamFunction,
    GenerateStreamFunction,
    ProxyLLMClient,
    register_proxy_model_adapter,
)
from dbgpt.model.proxy.llms.proxy_model import ProxyModel, parse_model_request
from dbgpt.util.i18n_utils import _

from ...utils.parse_utils import parse_chat_message

logger = logging.getLogger(__name__)


@auto_register_resource(
    label=_("Custom Ollama Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("Custom Ollama proxy LLM configuration for non-standard APIs."),
    show_in_ui=False,
)
@dataclass
class CustomOllamaDeployModelParameters(LLMDeployModelParameters):
    """自定义 Ollama 部署模型参数。"""

    provider: str = "proxy/custom_ollama"

    api_base: Optional[str] = field(
        default="http://localhost:11434/api/generate",
        metadata={
            "help": _("The full API endpoint URL (including /api/generate)."),
        },
    )

    auth_token: Optional[str] = field(
        default="",
        metadata={
            "help": _("Authorization token for the API (e.g., 'token-user1')."),
        },
    )


def custom_ollama_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=4096
):
    """自定义 Ollama 流式生成函数。"""
    client: CustomOllamaLLMClient = model.proxy_llm_client
    request = parse_model_request(params, client.default_model, stream=True)
    for r in client.sync_generate_stream(request):
        yield r


class CustomOllamaLLMClient(ProxyLLMClient):
    """自定义 Ollama LLM 客户端，支持非标准的 Ollama API。
    
    该客户端支持：
    1. 使用 /api/generate 端点（而不是 /api/chat）
    2. 自定义 Authorization header
    3. 使用 prompt 字段（而不是 messages）
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        auth_token: Optional[str] = None,
        model_alias: Optional[str] = "llama2",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        if not model:
            model = "llama2"
        if not api_base:
            api_base = "http://localhost:11434/api/generate"
        
        self._model = model
        self._api_base = self._resolve_env_vars(api_base)
        self._auth_token = auth_token or ""

        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )

    @classmethod
    def new_client(
        cls,
        model_params: CustomOllamaDeployModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "CustomOllamaLLMClient":
        return cls(
            model=model_params.real_provider_model_name,
            api_base=model_params.api_base,
            auth_token=model_params.auth_token,
            model_alias=model_params.real_provider_model_name,
            context_length=model_params.context_length,
            executor=default_executor,
        )

    @classmethod
    def param_class(cls) -> Type[CustomOllamaDeployModelParameters]:
        return CustomOllamaDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        return custom_ollama_generate_stream

    @property
    def default_model(self) -> str:
        return self._model

    def _convert_messages_to_prompt(self, messages: list) -> str:
        """将消息列表转换为单个 prompt 字符串。"""
        prompt_parts = []
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # 处理列表格式的 content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "".join(text_parts)
                
                # 根据角色添加前缀
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                else:  # user
                    prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(str(msg))
        
        return "\n\n".join(prompt_parts)

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        """同步流式生成响应。"""
        request = self.local_covert_message(request, message_converter)
        
        # 转换消息为 prompt
        messages = request.to_common_messages()
        prompt = self._convert_messages_to_prompt(messages)
        
        model = request.model or self._model
        # DeepSeek-R1 系列模型需要特殊处理推理内容
        is_reasoning_model = (
            getattr(request.context, "is_reasoning_model", False) or
            "deepseek-r1" in model.lower() or
            "r1" in model.lower()
        )
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json"
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        
        # 构建请求数据
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        logger.info(
            f"Custom Ollama API 请求 - 模型: {model}, "
            f"API地址: {self._api_base}, "
            f"Prompt长度: {len(prompt)}, "
            f"推理模型: {is_reasoning_model}"
        )
        logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)[:200]}...")
        
        try:
            with requests.post(
                self._api_base,
                json=data,
                headers=headers,
                stream=True,
                timeout=60
            ) as response:
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"Custom Ollama API 错误: {error_msg}")
                    yield ModelOutput.build(
                        text=f"**API 请求失败**: {error_msg}",
                        error_code=-1,
                    )
                    return
                
                # 处理流式响应
                full_content = ""
                last_yielded_content = ""
                last_yielded_thinking = ""
                
                # DeepSeek-R1 的推理模式标记
                # 参考: https://github.com/deepseek-ai/DeepSeek-R1
                reasoning_patterns = [
                    {"start": "<think>", "end": "</think>"},
                    {"start": "<reasoning>", "end": "</reasoning>"},
                    {"start": "<思考>", "end": "</思考>"},
                ]
                
                for line in response.iter_lines():
                    if line:
                        try:
                            # 解析 JSON 响应
                            obj = json.loads(line)
                            
                            # 提取响应文本
                            if "response" in obj:
                                chunk_text = obj["response"]
                                full_content += chunk_text
                                
                                # 解析推理内容（如果是推理模型）
                                if is_reasoning_model:
                                    msg = parse_chat_message(
                                        full_content, 
                                        extract_reasoning=True,
                                        reasoning_patterns=reasoning_patterns
                                    )
                                else:
                                    # 非推理模型，直接使用全部内容
                                    msg = parse_chat_message(
                                        full_content, 
                                        extract_reasoning=False
                                    )
                                
                                # 只在内容有变化时才输出
                                # 这样可以避免重复输出相同的内容
                                content_changed = msg.content != last_yielded_content
                                thinking_changed = msg.reasoning_content != last_yielded_thinking
                                
                                if content_changed or thinking_changed:
                                    # 确保 text 参数始终有值，避免 "The content type is not text" 错误
                                    # 即使只有 thinking 内容，也要提供一个空字符串作为 text
                                    text_to_yield = msg.content if msg.content else ""
                                    thinking_to_yield = msg.reasoning_content if msg.reasoning_content else None
                                    
                                    yield ModelOutput.build(
                                        text=text_to_yield,
                                        thinking=thinking_to_yield,
                                        error_code=0,
                                        is_reasoning_model=is_reasoning_model
                                    )
                                    
                                    last_yielded_content = msg.content
                                    last_yielded_thinking = msg.reasoning_content
                            
                            # 检查是否完成
                            if obj.get("done", False):
                                logger.info(
                                    f"生成完成 - 总长度: {len(full_content)}, "
                                    f"思考内容长度: {len(last_yielded_thinking)}, "
                                    f"回答内容长度: {len(last_yielded_content)}"
                                )
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失败: {line.decode('utf-8')}, 错误: {e}")
                            continue
                        
        except requests.RequestException as e:
            error_msg = f"请求异常: {str(e)}"
            logger.error(f"Custom Ollama API 请求异常: {error_msg}")
            yield ModelOutput.build(
                text=f"**API 请求异常**: {error_msg}",
                error_code=-1,
            )
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.exception(f"Custom Ollama API 未知错误: {error_msg}")
            yield ModelOutput.build(
                text=f"**处理错误**: {error_msg}",
                error_code=-1,
            )


# 注册自定义代理模型适配器
register_proxy_model_adapter(
    CustomOllamaLLMClient,
    supported_models=[
        ModelMetadata(
            model="deepseek-r1:70b",
            context_length=64 * 1024,
            max_output_length=8 * 1024,
            description="DeepSeek-R1 (Custom Ollama API)",
            function_calling=False,
        ),
    ],
)

