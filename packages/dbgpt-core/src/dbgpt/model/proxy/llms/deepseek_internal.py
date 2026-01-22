import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Type, Union

import httpx

if TYPE_CHECKING:
    from concurrent.futures import Executor

from dbgpt.core import ModelMetadata, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    ResourceCategory,
    auto_register_resource,
)
from dbgpt.core.interface.parameter import LLMDeployModelParameters
from dbgpt.model.proxy.llms.proxy_model import ProxyModel, parse_model_request
from dbgpt.util.i18n_utils import _

from ..base import (
    AsyncGenerateStreamFunction,
    GenerateStreamFunction,
    ProxyLLMClient,
    register_proxy_model_adapter,
)

if TYPE_CHECKING:
    from httpx._types import ProxiesTypes

logger = logging.getLogger(__name__)

_DEEPSEEK_INTERNAL_DEFAULT_MODEL = "DeepSeek-R1-Online-64K"
_DEEPSEEK_INTERNAL_DEFAULT_API_BASE = "http://api.taiji.woa.com/openapi"


@auto_register_resource(
    label=_("DeepSeek Internal Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("DeepSeek Internal (Taiji) proxy LLM configuration."),
    documentation_url="http://api.taiji.woa.com",
    show_in_ui=False,
)
@dataclass
class DeepSeekInternalDeployModelParameters(LLMDeployModelParameters):
    """Deploy model parameters for DeepSeek Internal (Taiji)."""

    provider: str = "proxy/deepseek-internal"

    api_base: Optional[str] = field(
        default="${env:DEEPSEEK_INTERNAL_API_BASE:-http://api.taiji.woa.com/openapi}",
        metadata={
            "help": _("The base url of the DeepSeek Internal API."),
        },
    )

    api_key: Optional[str] = field(
        default="${env:DEEPSEEK_INTERNAL_API_KEY}",
        metadata={
            "help": _("The API key (Authorization) of the DeepSeek Internal API."),
            "tags": "privacy",
        },
    )

    wsid: Optional[str] = field(
        default="${env:DEEPSEEK_INTERNAL_WSID}",
        metadata={
            "help": _("The Wsid header value for DeepSeek Internal API."),
            "tags": "privacy",
        },
    )

    model: Optional[str] = field(
        default="${env:DEEPSEEK_INTERNAL_MODEL:-DeepSeek-R1-Online-64K}",
        metadata={
            "help": _("The model name to use."),
        },
    )

    timeout: Optional[int] = field(
        default=240,
        metadata={
            "help": _("Request timeout in seconds."),
        },
    )

    http_proxy: Optional[str] = field(
        default=None,
        metadata={"help": _("The http or https proxy to use")},
    )


async def deepseek_internal_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    client: DeepSeekInternalLLMClient = model.proxy_llm_client
    request = parse_model_request(params, client.default_model, stream=True)
    async for r in client.generate_stream(request):
        yield r


class DeepSeekInternalLLMClient(ProxyLLMClient):
    """DeepSeek Internal (Taiji) LLM Client.

    Uses SSE (Server-Sent Events) format for streaming responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        wsid: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = _DEEPSEEK_INTERNAL_DEFAULT_MODEL,
        timeout: Optional[int] = 240,
        http_proxy: Optional[str] = None,
        proxies: Optional["ProxiesTypes"] = None,
        model_alias: Optional[str] = None,
        context_length: Optional[int] = 64 * 1024,  # 64K context
        **kwargs,
    ):
        api_base = (
            api_base
            or os.getenv("DEEPSEEK_INTERNAL_API_BASE")
            or _DEEPSEEK_INTERNAL_DEFAULT_API_BASE
        )
        api_key = api_key or os.getenv("DEEPSEEK_INTERNAL_API_KEY")
        wsid = wsid or os.getenv("DEEPSEEK_INTERNAL_WSID")
        model = model or _DEEPSEEK_INTERNAL_DEFAULT_MODEL
        model_alias = model_alias or model

        if not api_key:
            raise ValueError(
                "DeepSeek Internal API key is required, please set 'DEEPSEEK_INTERNAL_API_KEY' in "
                "environment or pass it as an argument."
            )
        if not wsid:
            raise ValueError(
                "DeepSeek Internal Wsid is required, please set 'DEEPSEEK_INTERNAL_WSID' in "
                "environment or pass it as an argument."
            )

        self._api_key = api_key
        self._wsid = wsid
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._model_alias = model_alias
        self._timeout = timeout
        self._context_length = context_length or 64 * 1024

        # Setup HTTP client
        proxy_config = http_proxy or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy_config:
            self._client = httpx.AsyncClient(
                proxies=proxy_config,
                timeout=timeout,
            )
        else:
            self._client = httpx.AsyncClient(timeout=timeout)

        # Call parent class __init__ with required model_names parameter
        super().__init__(
            model_names=[model_alias or model],
            context_length=context_length or 64 * 1024,
            executor=kwargs.get("executor"),
            proxy_tokenizer=kwargs.get("proxy_tokenizer"),
        )

    @property
    def default_model(self) -> str:
        return self._model_alias or self._model

    def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse SSE line format: 'data: {...}'"""
        line = line.strip()
        if not line:
            return None
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                return {"done": True}
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE data: {data_str}")
                return None
        return None

    async def generate_stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelOutput]:
        """Generate stream response from DeepSeek Internal API using SSE."""
        url = f"{self._api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key,
            "Wsid": self._wsid,
        }

        # Convert messages to DeepSeek format
        messages = []
        for msg in request.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content
            if isinstance(content, list):
                # Handle multimodal content
                content = " ".join(
                    str(item) if not isinstance(item, dict) else item.get("text", "")
                    for item in content
                )
            messages.append({"role": role, "content": str(content)})

        # Generate query_id (UUID)
        query_id = f"query_{uuid.uuid4()}"

        json_data = {
            "model": request.model or self._model,
            "query_id": query_id,
            "messages": messages,
            "stream": True,
        }

        # Add optional parameters
        if request.temperature is not None:
            json_data["temperature"] = request.temperature
        else:
            json_data["temperature"] = 1.0

        if request.max_new_tokens is not None:
            json_data["max_tokens"] = request.max_new_tokens
        else:
            json_data["max_tokens"] = 1024

        if request.top_p is not None:
            json_data["top_p"] = request.top_p
        else:
            json_data["top_p"] = 1.0

        try:
            async with self._client.stream(
                "POST", url, headers=headers, json=json_data
            ) as response:
                response.raise_for_status()
                
                # Parse SSE stream
                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="ignore")
                    
                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        data = self._parse_sse_line(line)
                        
                        if data is None:
                            continue
                        
                        if data.get("done"):
                            break
                        
                        # Extract content from SSE data
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield ModelOutput(
                                    text=content,
                                    error_code=0,
                                    finish_reason=None,
                                )
                        
                        # Check for finish reason
                        if choices and choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                            yield ModelOutput(
                                text="",
                                error_code=0,
                                finish_reason=finish_reason,
                            )
                            break
                            
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek Internal API request failed: {e}")
            yield ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=f"HTTP error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"DeepSeek Internal API error: {e}")
            yield ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=str(e),
            )

    async def generate(self, request: ModelRequest) -> ModelOutput:
        """Generate non-stream response from DeepSeek Internal API."""
        url = f"{self._api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key,
            "Wsid": self._wsid,
        }

        # Convert messages to DeepSeek format
        messages = []
        for msg in request.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    str(item) if not isinstance(item, dict) else item.get("text", "")
                    for item in content
                )
            messages.append({"role": role, "content": str(content)})

        # Generate query_id (UUID)
        query_id = f"query_{uuid.uuid4()}"

        json_data = {
            "model": request.model or self._model,
            "query_id": query_id,
            "messages": messages,
            "stream": False,
        }

        # Add optional parameters
        if request.temperature is not None:
            json_data["temperature"] = request.temperature
        else:
            json_data["temperature"] = 1.0

        if request.max_new_tokens is not None:
            json_data["max_tokens"] = request.max_new_tokens
        else:
            json_data["max_tokens"] = 1024

        if request.top_p is not None:
            json_data["top_p"] = request.top_p
        else:
            json_data["top_p"] = 1.0

        try:
            response = await self._client.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                finish_reason = choices[0].get("finish_reason", "stop")
                return ModelOutput(
                    text=content,
                    error_code=0,
                    finish_reason=finish_reason,
                )
            return ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error="No response from DeepSeek Internal API",
            )
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek Internal API request failed: {e}")
            return ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=f"HTTP error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"DeepSeek Internal API error: {e}")
            return ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=str(e),
            )

    async def count_token(self, model: str, prompt: str) -> int:
        """Count tokens (placeholder, DeepSeek Internal API may not support this)."""
        # Simple estimation: ~4 characters per token
        return len(prompt) // 4

    async def models(self) -> list:
        """List available models."""
        return [self._model]

    @classmethod
    def new_client(
        cls,
        model_params: DeepSeekInternalDeployModelParameters,
        default_executor: Optional["Executor"] = None,
    ) -> "DeepSeekInternalLLMClient":
        """Create a new client with the model parameters."""
        return cls(
            api_key=model_params.api_key,
            wsid=model_params.wsid,
            api_base=model_params.api_base,
            model=model_params.real_provider_model_name,
            timeout=model_params.timeout,
            http_proxy=model_params.http_proxy,
            model_alias=model_params.real_provider_model_name,
            context_length=model_params.context_length or 64 * 1024,
            executor=default_executor or ThreadPoolExecutor(),
        )

    @classmethod
    def param_class(cls) -> Type[DeepSeekInternalDeployModelParameters]:
        return DeepSeekInternalDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        return deepseek_internal_generate_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()


register_proxy_model_adapter(
    DeepSeekInternalLLMClient,
    supported_models=[
        ModelMetadata(
            model=["DeepSeek-R1-Online-64K"],
            context_length=64 * 1024,
            max_output_length=32 * 1024,
            description="DeepSeek-R1-Online-64K by DeepSeek (Internal Taiji API)",
            link="http://api.taiji.woa.com",
            function_calling=False,
        ),
    ],
)

