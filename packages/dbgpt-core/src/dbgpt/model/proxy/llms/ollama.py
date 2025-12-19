import logging
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Iterator, Optional, Type, Union

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

from ...utils.parse_utils import (
    parse_chat_message,
)

logger = logging.getLogger(__name__)


@auto_register_resource(
    label=_("Ollama Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("Ollama proxy LLM configuration."),
    documentation_url="https://ollama.com/library",
    show_in_ui=False,
)
@dataclass
class OllamaDeployModelParameters(LLMDeployModelParameters):
    """Deploy model parameters for Ollama."""

    provider: str = "proxy/ollama"

    api_base: Optional[str] = field(
        default="${env:OLLAMA_API_BASE:-http://localhost:11434}",
        metadata={
            "help": _("The base url of the Ollama API."),
        },
    )


def ollama_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=4096
):
    client: OllamaLLMClient = model.proxy_llm_client
    request = parse_model_request(params, client.default_model, stream=True)
    for r in client.sync_generate_stream(request):
        yield r


class OllamaLLMClient(ProxyLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        model_alias: Optional[str] = "llama2",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        if not model:
            model = "llama2"
        if not api_base:
            api_base = "http://localhost:11434"
        self._model = model
        self._api_base = self._resolve_env_vars(api_base)

        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )

    @classmethod
    def new_client(
        cls,
        model_params: OllamaDeployModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "OllamaLLMClient":
        return cls(
            model=model_params.real_provider_model_name,
            api_base=model_params.api_base,
            model_alias=model_params.real_provider_model_name,
            context_length=model_params.context_length,
            executor=default_executor,
        )

    @classmethod
    def param_class(cls) -> Type[OllamaDeployModelParameters]:
        return OllamaDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        return ollama_generate_stream

    @property
    def default_model(self) -> str:
        return self._model

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        try:
            import ollama
            from ollama import Client
        except ImportError as e:
            raise ValueError(
                "Could not import python package: ollama "
                "Please install ollama by command `pip install ollama"
            ) from e
        request = self.local_covert_message(request, message_converter)
        # Log original messages for debugging
        # Ensure messages are in the correct format for Ollama
        processed_messages = []
        for msg in request.to_common_messages():
            if hasattr(msg, "object") and hasattr(msg.object, "data"):
                # Handle MediaContent objects
                processed_messages.append({"content": msg.object.data, "role": "user"})
            elif isinstance(msg, dict):
                # Handle dict messages
                if "content" in msg:
                    content = msg["content"]
                    # If content is a list, convert it to string
                    if isinstance(content, list):
                        # Try to extract text from list of dicts
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        content = "".join(text_parts)
                    processed_messages.append({"content": content, "role": msg.get("role", "user")})
                elif "text" in msg:
                    processed_messages.append({"content": msg["text"], "role": msg.get("role", "user")})
                else:
                    processed_messages.append(msg)
            else:
                processed_messages.append(msg)
        
        messages = processed_messages

        model = request.model or self._model
        is_reasoning_model = getattr(request.context, "is_reasoning_model", False)
        client = Client(self._api_base)
        
        # 记录请求信息以便调试
        logger.info(f"Ollama API 请求 - 模型: {model}, API地址: {self._api_base}")
        logger.debug(f"Ollama 消息数量: {len(messages)}")
        
        try:
            stream = client.chat(
                model=model,
                messages=messages,
                stream=True,
            )
            content = ""
            for chunk in stream:
                content = content + chunk["message"]["content"]
                msg = parse_chat_message(content, extract_reasoning=is_reasoning_model)
                yield ModelOutput.build(
                    text=msg.content, thinking=msg.reasoning_content, error_code=0
                )
        except ollama.ResponseError as e:
            error_msg = str(e)
            logger.error(
                f"Ollama API 错误 - 模型: {model}, API地址: {self._api_base}, "
                f"错误: {error_msg}"
            )
            
            # 提供更详细的错误提示
            if "405" in error_msg or "Method Not Allowed" in error_msg:
                error_detail = (
                    f"**Ollama API 错误 (405 Method Not Allowed)**\n\n"
                    f"当前配置的 API 地址 `{self._api_base}` 不支持 Ollama 的 chat 接口。\n\n"
                    f"可能的原因：\n"
                    f"1. 该地址不是标准的 Ollama 服务（标准端口是 11434）\n"
                    f"2. 该服务不支持 `/api/chat` 端点\n"
                    f"3. 需要检查 API 地址配置是否正确\n\n"
                    f"原始错误信息: {error_msg}"
                )
            else:
                error_detail = f"**Ollama Response Error, Please CheckErrorInfo.**: {error_msg}"
            
            yield ModelOutput.build(
                text=error_detail,
                error_code=-1,
            )


register_proxy_model_adapter(
    OllamaLLMClient,
    supported_models=[
        ModelMetadata(
            model="deepseek-v3",
            context_length=64 * 1024,
            max_output_length=8 * 1024,
            description="DeepSeek-V3 by DeepSeek",
            link="https://ollama.com/library/deepseek-v3",
            function_calling=True,
        ),
        ModelMetadata(
            model="deepseek-r1:671b",
            context_length=64 * 1024,
            max_output_length=8 * 1024,
            description="DeepSeek-R1 by DeepSeek",
            link="https://ollama.com/library/deepseek-r1",
            function_calling=True,
        ),
        # More models see: https://ollama.com/search
    ],
)
