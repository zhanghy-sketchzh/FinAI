import json
import logging
import os
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

_HUNYUAN_DEFAULT_MODEL = "hunyuan-t1-latest"
_HUNYUAN_DEFAULT_API_BASE = "http://hunyuanapi.woa.com/openapi/v1"


@auto_register_resource(
    label=_("Hunyuan Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("Hunyuan proxy LLM configuration."),
    documentation_url="http://hunyuanapi.woa.com",
    show_in_ui=False,
)
@dataclass
class HunyuanDeployModelParameters(LLMDeployModelParameters):
    """Deploy model parameters for Hunyuan."""

    provider: str = "proxy/hunyuan"

    api_base: Optional[str] = field(
        default="${env:HUNYUAN_API_BASE:-http://hunyuanapi.woa.com/openapi/v1}",
        metadata={
            "help": _("The base url of the Hunyuan API."),
        },
    )

    api_key: Optional[str] = field(
        default="${env:HUNYUAN_API_KEY}",
        metadata={
            "help": _("The API key of the Hunyuan API."),
            "tags": "privacy",
        },
    )

    model: Optional[str] = field(
        default="${env:HUNYUAN_MODEL:-hunyuan-t1-latest}",
        metadata={
            "help": _("The model name to use."),
        },
    )

    enable_enhancement: Optional[bool] = field(
        default=False,
        metadata={
            "help": _("Enable enhancement feature."),
        },
    )

    sensitive_business: Optional[bool] = field(
        default=True,
        metadata={
            "help": _("Sensitive business flag."),
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


async def hunyuan_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    client: HunyuanLLMClient = model.proxy_llm_client
    request = parse_model_request(params, client.default_model, stream=True)
    async for r in client.generate_stream(request):
        yield r


class HunyuanLLMClient(ProxyLLMClient):
    """Hunyuan LLM Client.

    Hunyuan API uses OpenAI-compatible format but requires custom HTTP requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = _HUNYUAN_DEFAULT_MODEL,
        enable_enhancement: Optional[bool] = False,
        sensitive_business: Optional[bool] = True,
        timeout: Optional[int] = 240,
        http_proxy: Optional[str] = None,
        proxies: Optional["ProxiesTypes"] = None,
        model_alias: Optional[str] = None,
        context_length: Optional[int] = 8192,
        **kwargs,
    ):
        api_base = (
            api_base
            or os.getenv("HUNYUAN_API_BASE")
            or _HUNYUAN_DEFAULT_API_BASE
        )
        api_key = api_key or os.getenv("HUNYUAN_API_KEY")
        model = model or _HUNYUAN_DEFAULT_MODEL
        model_alias = model_alias or model

        if not api_key:
            raise ValueError(
                "Hunyuan API key is required, please set 'HUNYUAN_API_KEY' in "
                "environment or pass it as an argument."
            )

        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._model_alias = model_alias
        self._enable_enhancement = enable_enhancement
        self._sensitive_business = sensitive_business
        self._timeout = timeout
        self._context_length = context_length or 8192

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
            context_length=context_length,
            executor=kwargs.get("executor"),
            proxy_tokenizer=kwargs.get("proxy_tokenizer"),
        )

    @property
    def default_model(self) -> str:
        return self._model_alias or self._model

    def _convert_messages_to_hunyuan_format(self, messages):
        """Convert messages to Hunyuan API format.
        
        Hunyuan API only supports: system, user, assistant, tool
        Maps "human" to "user" and filters out empty messages.
        
        Hunyuan API message order requirements:
        - system messages can be at the beginning (multiple allowed)
        - After system, messages must alternate between user/tool and assistant
        - The last message must be user/tool (not assistant)
        
        Args:
            messages: List of message objects from ModelRequest
            
        Returns:
            List of dicts with "role" and "content" keys, ordered correctly
        """
        role_mapping = {
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "ai": "assistant",  # Map "ai" to "assistant"
            "system": "system",
            "tool": "tool",
        }
        
        # Step 1: Convert and filter messages
        converted_messages = []
        for i, msg in enumerate(messages):
            # Extract role - handle both string and object types
            if isinstance(msg.role, str):
                role_str = msg.role
            elif hasattr(msg.role, "value"):
                role_str = msg.role.value
            else:
                role_str = str(msg.role)
            
            # Normalize role to lowercase and map to Hunyuan supported roles
            role = role_mapping.get(role_str.lower(), "user")
            
            # Extract and process content
            content = msg.content
            if isinstance(content, list):
                # Handle multimodal content
                content = " ".join(
                    str(item) if not isinstance(item, dict) else item.get("text", "")
                    for item in content
                )
            
            content_str = str(content) if content else ""
            # Skip empty messages as Hunyuan API requires non-empty content
            if not content_str.strip():
                logger.warning(
                    f"Skipping message {i} with empty content, role: {role} "
                    f"(original role: {role_str})"
                )
                continue
            
            converted_messages.append({"role": role, "content": content_str})
        
        if not converted_messages:
            logger.warning("No valid messages after conversion to Hunyuan format")
            return []
        
        # Step 2: Separate system messages and non-system messages
        system_messages = [msg for msg in converted_messages if msg["role"] == "system"]
        non_system_messages = [msg for msg in converted_messages if msg["role"] != "system"]
        
        # Step 3: Validate and fix non-system message order
        # Hunyuan requires: user/tool <-> assistant alternating, ending with user/tool
        user_tool_roles = {"user", "tool"}
        assistant_role = "assistant"
        
        if non_system_messages:
            # Fix 1: First non-system message must be user/tool
            if non_system_messages[0]["role"] == assistant_role:
                logger.warning(
                    f"First non-system message is assistant, should be user/tool. "
                    f"Auto-fixing by changing to user."
                )
                non_system_messages[0]["role"] = "user"
            
            # Fix 2: Ensure proper alternating pattern
            # After assistant, must be user/tool
            # After user/tool, can be assistant or user/tool (multiple allowed)
            fixed_messages = []
            last_role = None
            
            for i, msg in enumerate(non_system_messages):
                current_role = msg["role"]
                
                if last_role == assistant_role:
                    # After assistant, must be user/tool
                    if current_role not in user_tool_roles:
                        logger.warning(
                            f"Message {i} role '{current_role}' after assistant should be user/tool. "
                            f"Auto-fixing to user."
                        )
                        msg["role"] = "user"
                        current_role = "user"
                elif last_role and last_role in user_tool_roles:
                    # After user/tool, can be assistant or user/tool
                    # This is allowed, no fix needed
                    pass
                
                fixed_messages.append(msg)
                last_role = current_role
            
            non_system_messages = fixed_messages
            
            # Fix 3: Ensure last message is user/tool
            if non_system_messages and non_system_messages[-1]["role"] == assistant_role:
                logger.warning(
                    f"Last message is assistant, should be user/tool. "
                    f"Auto-fixing by changing to user."
                )
                non_system_messages[-1]["role"] = "user"
        
        # Step 4: Combine system and non-system messages
        hunyuan_messages = system_messages + non_system_messages
        
        # Log final message format for debugging
        if hunyuan_messages:
            roles_sequence = [msg["role"] for msg in hunyuan_messages]
            logger.debug(
                f"Converted {len(messages)} messages to {len(hunyuan_messages)} Hunyuan messages. "
                f"Role sequence: {roles_sequence}"
            )
        
        return hunyuan_messages

    async def generate_stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelOutput]:
        """Generate stream response from Hunyuan API."""
        url = f"{self._api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Convert messages to Hunyuan format
        messages = self._convert_messages_to_hunyuan_format(request.messages)

        json_data = {
            "model": request.model or self._model,
            "messages": messages,
            "enable_enhancement": self._enable_enhancement,
            "sensitive_business": self._sensitive_business,
            "stream": True,
        }

        # Add optional parameters
        if request.temperature is not None:
            json_data["temperature"] = request.temperature
        if request.max_new_tokens is not None:
            json_data["max_tokens"] = request.max_new_tokens
        if request.top_p is not None:
            json_data["top_p"] = request.top_p

        try:
            async with self._client.stream(
                "POST", url, headers=headers, json=json_data, timeout=self._timeout
            ) as response:
                # Check status code before processing
                if response.status_code != 200:
                    # Read error response body
                    error_text = ""
                    try:
                        async for chunk in response.aiter_bytes():
                            error_text += chunk.decode("utf-8", errors="ignore")
                    except:
                        pass
                    
                    error_msg = error_text
                    try:
                        error_json = json.loads(error_text)
                        error_msg = error_json.get("error", {}).get("message", error_text)
                    except:
                        pass
                    
                    logger.error(
                        f"Hunyuan API request failed: {response.status_code} - {error_msg}\n"
                        f"Request URL: {url}\n"
                        f"Request data: {json.dumps(json_data, ensure_ascii=False, indent=2)}"
                    )
                    yield ModelOutput(
                        text="",
                        error_code=response.status_code,
                        finish_reason="error",
                        error=f"HTTP {response.status_code}: {error_msg}",
                    )
                    return
                
                # Accumulate content for streaming output
                accumulated_text = ""
                finish_reason = None
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                # Check for finish_reason
                                if choices[0].get("finish_reason"):
                                    finish_reason = choices[0].get("finish_reason")
                                
                                if content:
                                    # Accumulate the content
                                    accumulated_text += content
                                    # Yield the accumulated text (not just delta)
                                    yield ModelOutput(
                                        text=accumulated_text,
                                        error_code=0,
                                        finish_reason=finish_reason,
                                    )
                                elif finish_reason:
                                    # Yield final output with finish_reason
                                    yield ModelOutput(
                                        text=accumulated_text,
                                        error_code=0,
                                        finish_reason=finish_reason,
                                    )
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE data: {data_str}")
                            continue
        except httpx.HTTPStatusError as e:
            # Try to get error details from response
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_body = e.response.read()
                    error_text = error_body.decode("utf-8", errors="ignore")
                    try:
                        error_json = json.loads(error_text)
                        error_msg = error_json.get("error", {}).get("message", error_text)
                    except:
                        error_msg = error_text
                except:
                    pass
            
            logger.error(
                f"Hunyuan API request failed: {e.response.status_code if e.response else 'Unknown'} - {error_msg}\n"
                f"Request URL: {url}\n"
                f"Request data: {json.dumps(json_data, ensure_ascii=False, indent=2)}"
            )
            yield ModelOutput(
                text="",
                error_code=e.response.status_code if e.response else 1,
                finish_reason="error",
                error=f"HTTP error: {error_msg}",
            )
        except httpx.HTTPError as e:
            logger.error(f"Hunyuan API request failed: {e}")
            yield ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=f"HTTP error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Hunyuan API error: {e}")
            yield ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=str(e),
            )

    async def generate(self, request: ModelRequest) -> ModelOutput:
        """Generate non-stream response from Hunyuan API."""
        url = f"{self._api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Convert messages to Hunyuan format
        messages = self._convert_messages_to_hunyuan_format(request.messages)

        json_data = {
            "model": request.model or self._model,
            "messages": messages,
            "enable_enhancement": self._enable_enhancement,
            "sensitive_business": self._sensitive_business,
            "stream": False,
        }

        # Add optional parameters
        if request.temperature is not None:
            json_data["temperature"] = request.temperature
        if request.max_new_tokens is not None:
            json_data["max_tokens"] = request.max_new_tokens
        if request.top_p is not None:
            json_data["top_p"] = request.top_p

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
                error="No response from Hunyuan API",
            )
        except httpx.HTTPStatusError as e:
            # Try to get error details from response
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_text = e.response.text
                    try:
                        error_json = json.loads(error_text)
                        error_msg = error_json.get("error", {}).get("message", error_text)
                    except:
                        error_msg = error_text
                except:
                    pass
            
            logger.error(
                f"Hunyuan API request failed: {e.response.status_code if e.response else 'Unknown'} - {error_msg}\n"
                f"Request URL: {url}\n"
                f"Request data: {json.dumps(json_data, ensure_ascii=False, indent=2)}"
            )
            return ModelOutput(
                text="",
                error_code=e.response.status_code if e.response else 1,
                finish_reason="error",
                error=f"HTTP {e.response.status_code if e.response else 'Unknown'}: {error_msg}",
            )
        except httpx.HTTPError as e:
            logger.error(f"Hunyuan API request failed: {e}")
            return ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=f"HTTP error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Hunyuan API error: {e}")
            return ModelOutput(
                text="",
                error_code=1,
                finish_reason="error",
                error=str(e),
            )

    async def count_token(self, model: str, prompt: str) -> int:
        """Count tokens (placeholder, Hunyuan API may not support this)."""
        # Simple estimation: ~4 characters per token
        return len(prompt) // 4

    async def models(self) -> list:
        """List available models."""
        return [self._model]

    @classmethod
    def new_client(
        cls,
        model_params: HunyuanDeployModelParameters,
        default_executor: Optional["Executor"] = None,
    ) -> "HunyuanLLMClient":
        """Create a new client with the model parameters."""
        return cls(
            api_key=model_params.api_key,
            api_base=model_params.api_base,
            model=model_params.real_provider_model_name,
            enable_enhancement=model_params.enable_enhancement,
            sensitive_business=model_params.sensitive_business,
            timeout=model_params.timeout,
            http_proxy=model_params.http_proxy,
            model_alias=model_params.real_provider_model_name,
            context_length=model_params.context_length or 8192,
            executor=default_executor or ThreadPoolExecutor(),
        )

    @classmethod
    def param_class(cls) -> Type[HunyuanDeployModelParameters]:
        return HunyuanDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        return hunyuan_generate_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()


register_proxy_model_adapter(
    HunyuanLLMClient,
    supported_models=[
        ModelMetadata(
            model=["hunyuan-t1-latest"],
            context_length=32 * 1024,
            max_output_length=16 * 1024,
            description="Hunyuan T1 Latest by Tencent",
            link="http://hunyuanapi.woa.com",
            function_calling=False,
        ),
    ],
)

