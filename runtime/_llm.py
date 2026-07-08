"""
LLM — Production-grade multi-provider LLM client.

Supports OpenAI-compatible APIs and Nebula cluster with automatic
failover, retry logic, streaming, and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("aicp.llm")


def _log(msg: str) -> None:
    """Structured debug logging."""
    logger.debug(msg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default timeout values (seconds)
DEFAULT_REQUEST_TIMEOUT: float = 60.0
DEFAULT_STREAM_TIMEOUT: float = 300.0
DEFAULT_CONNECT_TIMEOUT: float = 10.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_MAX_CONCURRENT: int = 10

# HTTP timeout configuration
HTTP_TIMEOUT_CONFIG = {
    "connect": DEFAULT_CONNECT_TIMEOUT,
    "read": 90.0,
    "write": 30.0,
    "pool": DEFAULT_CONNECT_TIMEOUT,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Validated model configuration."""
    client_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_streaming: bool = True
    high_quality: bool = False


@dataclass
class ProviderConfig:
    """Provider configuration container."""
    name: str
    type: str  # "openai" or "cluster"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    models: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NebulaResponse:
    """Parsed Nebula cluster response."""
    content: Optional[str] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    success: bool = False


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMNotConfiguredError(LLMError):
    """Raised when no LLM client is configured."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""
    pass


class LLMEmptyResponseError(LLMError):
    """Raised when LLM returns an empty response."""
    pass


# ---------------------------------------------------------------------------
# Safe fallback constants
# ---------------------------------------------------------------------------

# Messages returned to users when LLM fails (never None)
FALLBACK_MESSAGES = {
    "not_configured": "[LLM 未配置]",
    "nebula_unavailable": "[Nebula 不可用且无降级模型]",
    "empty_response": "[模型返回空响应，请稍后重试]",
    "timeout": "[服务请求超时，请稍后重试]",
    "max_retries": "[LLM 达到最大重试次数]",
    "system_error": "[系统错误]",
    "empty": "[空响应]",
}


# ---------------------------------------------------------------------------
# Nebula HTTP client (shared, synchronous)
# ---------------------------------------------------------------------------


class NebulaClient:
    """Thread-safe HTTP client for Nebula cluster API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session: Optional[Any] = None  # requests.Session

    def _ensure_session(self) -> Any:
        """Lazy-initialize requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "AICP-LLM/1.0",
            })
        return self._session

    def chat(
        self,
        messages: List[Dict[str, Any]],
        timeout: float = 180.0,
    ) -> NebulaResponse:
        """Send chat request to Nebula cluster.

        Args:
            messages: Chat messages (supports text, image, file)
            timeout: Request timeout in seconds

        Returns:
            NebulaResponse with content or error
        """
        import requests

        t0 = time.time()

        try:
            # Parse messages for multimodal content
            image_path, file_path, text_messages = self._parse_messages(messages)

            # Build payload
            if file_path:
                payload = {
                    "action": "chat_with_file",
                    "messages": text_messages,
                    "file_path": file_path,
                }
            elif image_path:
                payload = {
                    "action": "chat_with_image",
                    "messages": text_messages,
                    "image_path": image_path,
                }
            else:
                payload = {
                    "action": "chat",
                    "messages": text_messages,
                }

            session = self._ensure_session()
            resp = session.post(
                self._base_url,
                json={"payload": payload},
                timeout=(10, timeout),
            )
            resp.raise_for_status()

            data = resp.json()

            if data.get("error"):
                return NebulaResponse(
                    error=str(data["error"]),
                    elapsed=time.time() - t0,
                )

            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                elapsed = time.time() - t0
                if content and content.strip():
                    _log(f"nebula ← {len(content)} chars, {elapsed:.1f}s")
                    return NebulaResponse(
                        content=content,
                        elapsed=elapsed,
                        success=True,
                    )

            return NebulaResponse(
                content=None,
                elapsed=time.time() - t0,
                success=False,
            )

        except requests.exceptions.ConnectionError as exc:
            _log(f"nebula connection error: {exc}")
            return NebulaResponse(
                error=f"连接失败: {exc}",
                elapsed=time.time() - t0,
            )
        except requests.exceptions.Timeout as exc:
            _log(f"nebula timeout: {exc}")
            return NebulaResponse(
                error=f"请求超时",
                elapsed=time.time() - t0,
            )
        except Exception as exc:
            _log(f"nebula error: {type(exc).__name__}: {str(exc)[:100]}")
            return NebulaResponse(
                error=f"{type(exc).__name__}: {str(exc)[:100]}",
                elapsed=time.time() - t0,
            )

    def generate_image(
        self,
        prompt: str,
        n: int = 4,
        timeout: float = 180.0,
    ) -> List[str]:
        """Generate images via Nebula cluster.

        Args:
            prompt: Image description
            n: Number of images
            timeout: Request timeout

        Returns:
            List of image URLs
        """
        import requests

        try:
            payload = {
                "action": "image",
                "prompt": prompt,
                "n": n,
            }

            session = self._ensure_session()
            resp = session.post(
                self._base_url,
                json={"payload": payload},
                timeout=(10, timeout),
            )
            resp.raise_for_status()

            data = resp.json()

            if data.get("error"):
                _log(f"nebula image error: {data['error']}")
                return []

            images = data.get("data", [])
            urls = [img.get("url") for img in images if img.get("url")]
            _log(f"nebula image ← {len(urls)} 张")
            return urls

        except Exception as exc:
            _log(f"nebula image error: {type(exc).__name__}: {str(exc)[:100]}")
            return []

    @staticmethod
    def _parse_messages(
        messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Parse messages to extract image/file paths and build text messages.

        Args:
            messages: Raw chat messages

        Returns:
            Tuple of (image_path, file_path, text_messages)
        """
        image_path: Optional[str] = None
        file_path: Optional[str] = None
        text_messages: List[Dict[str, Any]] = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts: List[str] = []
                for part in content:
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        image_path = url[7:] if url.startswith("file://") else url
                    elif part_type == "file":
                        file_path = part.get("file", {}).get("file_path", "")
                    else:
                        text_parts.append(str(part.get("text", "")))
                msg = {
                    "role": msg.get("role", "user"),
                    "content": " ".join(text_parts),
                }
            text_messages.append(msg)

        return image_path, file_path, text_messages


# ---------------------------------------------------------------------------
# Main LLM class
# ---------------------------------------------------------------------------


class LLM:
    """Production-grade multi-provider LLM client.

    Supports:
    - OpenAI-compatible APIs (via openai library)
    - Nebula cluster (via HTTP)
    - Automatic failover from Nebula to OpenAI-compatible
    - Streaming responses
    - Image generation
    - JSON structured output
    - Concurrency control via semaphore
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LLM client from configuration.

        Args:
            config: Configuration dictionary with 'models' section
        """
        models_cfg = config.get("models", {})

        # Role-based model selection
        self._roles: Dict[str, str] = models_cfg.get("roles", {})
        self.default_model: str = self._roles.get(
            "default",
            models_cfg.get("default", "gpt-3.5-turbo"),
        )

        # Providers
        self.providers: Dict[str, Any] = models_cfg.get("providers", {})

        # Internal state
        self._clients: Dict[str, Any] = {}
        self._model_to_client: Dict[str, str] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._nebula_client: Optional[NebulaClient] = None
        self.high_quality_model: Optional[str] = None

        # Retry & timeout configuration
        self.max_retries: int = models_cfg.get("max_retries", DEFAULT_MAX_RETRIES)
        self.request_timeout: float = models_cfg.get(
            "request_timeout", DEFAULT_REQUEST_TIMEOUT
        )
        self.stream_timeout: float = models_cfg.get(
            "stream_timeout", DEFAULT_STREAM_TIMEOUT
        )
        self.connect_timeout: float = models_cfg.get(
            "connect_timeout", DEFAULT_CONNECT_TIMEOUT
        )

        # Concurrency control
        max_concurrent: int = models_cfg.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Fallback model for Nebula failures
        self._fallback_model: Optional[str] = None

        _log(f"📋 Config models keys: {list(models_cfg.keys())}")
        _log(f"⏱️ request_timeout: {self.request_timeout}s")

        # Initialize all providers
        self._init_clients()

    # ======================================================================
    # Provider initialization
    # ======================================================================

    def _init_clients(self) -> None:
        """Initialize all configured providers."""
        for name, cfg in self.providers.items():
            provider = ProviderConfig(
                name=name,
                type=cfg.get("type", "openai"),
                api_key=cfg.get("api_key", ""),
                base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                models=cfg.get("models", []),
            )

            if provider.type == "cluster":
                self._init_cluster_provider(provider)
            else:
                self._init_openai_provider(provider)

        # Validate default model
        self._validate_default_model()

        # Find fallback model (first non-Nebula model)
        self._find_fallback_model()

        # Log summary
        _log(f"🤖 Default: {self.default_model} | Models: {len(self._model_to_client)}")
        if self._roles:
            _log(f"📋 Roles: {self._roles}")

    def _init_cluster_provider(self, provider: ProviderConfig) -> None:
        """Initialize a Nebula cluster provider."""
        self._clients[provider.name] = {
            "type": "cluster",
            "base_url": provider.base_url,
        }

        # Initialize shared Nebula client
        if self._nebula_client is None:
            self._nebula_client = NebulaClient(provider.base_url)

        for model_cfg in provider.models:
            model_id = model_cfg.get("id")
            if not model_id:
                continue

            self._model_configs[model_id] = ModelConfig(
                client_name=provider.name,
                max_tokens=model_cfg.get("max_tokens", 4096),
                temperature=model_cfg.get("temperature", 0.7),
                supports_streaming=False,  # Nebula does not support streaming natively
                high_quality=model_cfg.get("high_quality", False),
            )
            self._model_to_client[model_id] = provider.name

            if model_cfg.get("high_quality"):
                self.high_quality_model = model_id

            _log(f"✅ {model_id} ({provider.name}) [cluster]")

    def _init_openai_provider(self, provider: ProviderConfig) -> None:
        """Initialize an OpenAI-compatible provider."""
        # Skip if API key is not set
        if not provider.api_key or provider.api_key.startswith("${"):
            _log(f"⚠️ Skip {provider.name}: API key not configured")
            return

        try:
            timeout = httpx.Timeout(
                timeout=120.0,
                connect=self.connect_timeout,
                read=HTTP_TIMEOUT_CONFIG["read"],
                write=HTTP_TIMEOUT_CONFIG["write"],
                pool=self.connect_timeout,
            )

            client = AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=timeout,
                http_client=httpx.AsyncClient(
                    timeout=timeout,
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                    ),
                ),
                max_retries=0,  # We handle retries ourselves
            )
            self._clients[provider.name] = client

            for model_cfg in provider.models:
                model_id = model_cfg.get("id")
                if not model_id:
                    continue

                self._model_configs[model_id] = ModelConfig(
                    client_name=provider.name,
                    max_tokens=model_cfg.get("max_tokens", 4096),
                    temperature=model_cfg.get("temperature", 0.7),
                    supports_streaming=model_cfg.get("supports_streaming", True),
                    high_quality=model_cfg.get("high_quality", False),
                )
                self._model_to_client[model_id] = provider.name

                if model_cfg.get("high_quality"):
                    self.high_quality_model = model_id

                _log(f"✅ {model_id} ({provider.name})")

        except Exception as exc:
            _log(f"❌ Failed to init {provider.name}: {exc}")

    def _validate_default_model(self) -> None:
        """Ensure default_model exists, or fall back to first available."""
        if self.default_model not in self._model_to_client:
            if self._model_to_client:
                self.default_model = next(iter(self._model_to_client))
                _log(f"⚠️ roles.default 模型不可用，改用: {self.default_model}")
            else:
                _log("❌ 没有任何可用模型！")

    def _find_fallback_model(self) -> None:
        """Find first non-Nebula model for failover."""
        for model_id, client_name in self._model_to_client.items():
            client = self._clients.get(client_name)
            is_nebula = isinstance(client, dict) and client.get("type") == "cluster"
            if not is_nebula:
                self._fallback_model = model_id
                _log(f"🔄 降级模型: {self._fallback_model}")
                break

    # ======================================================================
    # Model resolution
    # ======================================================================

    def _resolve_model(self, role: Optional[str] = None) -> str:
        """Resolve model from role or return default.

        Args:
            role: Role name (e.g., "coding", "reasoning")

        Returns:
            Resolved model ID (never None)
        """
        if not role:
            return self.default_model

        model = self._roles.get(role)
        if model and model in self._model_to_client:
            return model

        if model:
            _log(f"⚠️ roles.{role}={model} 不可用，降级到 default")

        return self.default_model

    def _get_client(self, model: Optional[str] = None) -> tuple[Optional[Any], str]:
        """Get client and actual model name.

        Args:
            model: Requested model ID

        Returns:
            Tuple of (client, resolved_model)
        """
        resolved_model = model or self.default_model
        client_name = self._model_to_client.get(resolved_model)

        if client_name and client_name in self._clients:
            return self._clients[client_name], resolved_model

        # Fallback to first available client
        if self._clients:
            name = next(iter(self._clients))
            return self._clients[name], resolved_model

        return None, resolved_model

    def _is_nebula_client(self, client: Any) -> bool:
        """Check if client is a Nebula cluster client."""
        return isinstance(client, dict) and client.get("type") == "cluster"

    # ======================================================================
    # Nebula integration
    # ======================================================================

    async def _call_nebula(
        self,
        messages: List[Dict[str, Any]],
        timeout: float = 180.0,
    ) -> Optional[str]:
        """Call Nebula cluster asynchronously.

        Args:
            messages: Chat messages
            timeout: Request timeout

        Returns:
            Response content or None on failure
        """
        if self._nebula_client is None:
            return None

        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._nebula_client.chat,
                    messages,
                    timeout,
                ),
                timeout=timeout + 10,  # Slightly longer than HTTP timeout
            )

            if response.success and response.content:
                return response.content.strip()

            if response.error:
                _log(f"nebula error: {response.error}")

            return None

        except asyncio.TimeoutError:
            _log("nebula timeout")
            return None
        except Exception as exc:
            _log(f"nebula error: {type(exc).__name__}: {str(exc)[:100]}")
            return None

    async def _try_nebula_first(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Try Nebula cluster before falling back to default model.

        Args:
            messages: Chat messages

        Returns:
            Response content or None
        """
        if self._nebula_client is None:
            return None

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._nebula_client.chat,
                messages,
                30.0,  # Short timeout for preflight
            )

            if (
                response.success
                and response.content
                and response.content.strip()
                and not response.content.startswith("[")
            ):
                _log("✅ Nebula 命中！")
                return response.content.strip()

        except Exception:
            pass

        return None

    # ======================================================================
    # Retry logic
    # ======================================================================

    @staticmethod
    def _calculate_backoff(attempt: int, max_wait: int = 10) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-based)
            max_wait: Maximum wait time in seconds

        Returns:
            Wait time in seconds
        """
        import random
        wait = min(2 ** attempt, max_wait)
        jitter = random.uniform(0, wait * 0.5)
        return wait + jitter

    # ======================================================================
    # Core chat implementation
    # ======================================================================

    async def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Internal chat implementation. Guaranteed to return a string.

        Args:
            messages: Chat messages
            model: Model ID (optional)
            **kwargs: Additional parameters

        Returns:
            Response string (never None)
        """
        client, actual_model = self._get_client(model)

        # No client available
        if client is None:
            return FALLBACK_MESSAGES["not_configured"]

        # ==================================================================
        # Nebula cluster path
        # ==================================================================
        if self._is_nebula_client(client):
            _log(f"nebula cluster path, model={actual_model}")

            result = await self._call_nebula(messages)

            # Nebula failed → fallback to non-Nebula model
            if result is None and self._fallback_model:
                _log(f"🔄 Nebula 不可用，降级到 {self._fallback_model}")
                return await self._chat_impl(messages, self._fallback_model, **kwargs)

            return result if result else FALLBACK_MESSAGES["nebula_unavailable"]

        # ==================================================================
        # Try Nebula preflight for default model
        # ==================================================================
        if actual_model == self.default_model:
            nebula_result = await self._try_nebula_first(messages)
            if nebula_result is not None:
                return nebula_result

        # ==================================================================
        # OpenAI-compatible client path
        # ==================================================================
        config = self._model_configs.get(actual_model, ModelConfig(client_name="unknown"))
        max_tokens = kwargs.pop("max_tokens", config.max_tokens)
        temperature = kwargs.pop("temperature", config.temperature)

        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                _log(f"🔄 API call {attempt + 1}/{self.max_retries} → {actual_model}")
                t0 = time.time()

                task = asyncio.create_task(
                    client.chat.completions.create(
                        model=actual_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                )

                try:
                    response = await asyncio.wait_for(
                        task,
                        timeout=self.request_timeout,
                    )
                except asyncio.TimeoutError:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise

                elapsed = time.time() - t0

                # Validate response
                if (
                    response
                    and response.choices
                    and response.choices[0].message.content
                ):
                    result = response.choices[0].message.content.strip()
                    if result:
                        _log(f"✅ 成功 ({elapsed:.1f}s, {len(result)} chars)")
                        return result

                _log(f"⚠️ 空响应 ({elapsed:.1f}s)")
                last_error = "empty_response"

            except asyncio.TimeoutError:
                _log(f"⏱️ 超时 attempt {attempt + 1}")
                last_error = "timeout"
            except Exception as exc:
                error_msg = str(exc)[:200]
                _log(f"❌ 错误 attempt {attempt + 1}: {type(exc).__name__}: {error_msg[:100]}")
                last_error = error_msg

            # Don't sleep on last attempt
            if attempt < self.max_retries - 1:
                wait = self._calculate_backoff(attempt)
                _log(f"⏳ 等待 {wait:.1f}s 后重试...")
                await asyncio.sleep(wait)

        # All retries exhausted
        if last_error == "timeout":
            return FALLBACK_MESSAGES["timeout"]
        elif last_error == "empty_response":
            return FALLBACK_MESSAGES["empty_response"]
        else:
            return f"[LLM请求失败: {last_error[:200]}]"

    # ======================================================================
    # Public API
    # ======================================================================

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Chat completion interface.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Direct model ID (highest priority)
            role: Role-based model selection ("default", "coding", "reasoning")
            **kwargs: Additional parameters passed to API

        Returns:
            Response string (guaranteed non-None, non-empty)
        """
        # Resolve model
        if model:
            resolved_model = model
        elif role:
            resolved_model = self._resolve_model(role)
        else:
            resolved_model = self.default_model

        _log(f"💬 chat: {resolved_model}" + (f" (role={role})" if role else ""))

        try:
            async with self._semaphore:
                result = await self._chat_impl(messages, resolved_model, **kwargs)

                # Guarantee a valid string return
                if result is None:
                    return FALLBACK_MESSAGES["system_error"]
                if not isinstance(result, str):
                    result = str(result)
                if not result.strip():
                    return FALLBACK_MESSAGES["empty"]

                return result

        except Exception as exc:
            _log(f"❌ chat 异常: {type(exc).__name__}: {str(exc)[:100]}")
            return f"[系统错误: {str(exc)[:100]}]"

    # ======================================================================
    # Streaming
    # ======================================================================

    async def _chat_stream_impl(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Internal streaming implementation.

        Args:
            messages: Chat messages
            model: Model ID
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        client, actual_model = self._get_client(model)

        # No client available
        if client is None:
            yield FALLBACK_MESSAGES["not_configured"]
            return

        # ==================================================================
        # Nebula cluster (simulated streaming)
        # ==================================================================
        if self._is_nebula_client(client):
            result = await self._chat_impl(messages, model, **kwargs)
            if result and not result.startswith("["):
                # Simulate streaming by chunking
                chunk_size = 10
                for i in range(0, len(result), chunk_size):
                    yield result[i : i + chunk_size]
                    await asyncio.sleep(0.01)
            else:
                yield FALLBACK_MESSAGES["empty"]
            return

        # ==================================================================
        # Try Nebula preflight for default model
        # ==================================================================
        if actual_model == self.default_model:
            nebula_result = await self._try_nebula_first(messages)
            if nebula_result is not None:
                chunk_size = 10
                for i in range(0, len(nebula_result), chunk_size):
                    yield nebula_result[i : i + chunk_size]
                    await asyncio.sleep(0.01)
                return

        # ==================================================================
        # OpenAI-compatible streaming
        # ==================================================================
        config = self._model_configs.get(actual_model, ModelConfig(client_name="unknown"))

        # If streaming not supported, fall back to non-streaming
        if not config.supports_streaming:
            result = await self._chat_impl(messages, model, **kwargs)
            if result and not result.startswith("["):
                yield result
            else:
                yield FALLBACK_MESSAGES["empty"]
            return

        max_tokens = kwargs.pop("max_tokens", config.max_tokens)
        temperature = kwargs.pop("temperature", config.temperature)

        for attempt in range(self.max_retries):
            stream = None
            try:
                _log(f"📡 stream {attempt + 1}/{self.max_retries} → {actual_model}")

                stream = await client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    **kwargs,
                )

                collected: List[str] = []
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        if token:
                            collected.append(token)
                            yield token

                full_text = "".join(collected)
                _log(f"✅ stream 完成: {len(full_text)} chars")
                return

            except asyncio.TimeoutError:
                _log(f"⏱️ stream 超时 attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    yield FALLBACK_MESSAGES["timeout"]
                    return
            except Exception as exc:
                error_msg = str(exc)[:200]
                _log(f"❌ stream 错误: {type(exc).__name__}: {error_msg[:100]}")
                if attempt == self.max_retries - 1:
                    yield f"[LLM stream error: {error_msg[:200]}]"
                    return
            finally:
                # Always close stream to release resources
                if stream is not None:
                    try:
                        await stream.close()
                    except Exception:
                        pass

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self._calculate_backoff(attempt, max_wait=10))

        yield FALLBACK_MESSAGES["max_retries"]

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming chat interface.

        Args:
            messages: List of message dicts
            model: Direct model ID (highest priority)
            role: Role-based model selection
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        if model:
            resolved_model = model
        elif role:
            resolved_model = self._resolve_model(role)
        else:
            resolved_model = self.default_model

        async with self._semaphore:
            async for token in self._chat_stream_impl(
                messages,
                resolved_model,
                **kwargs,
            ):
                if token:
                    yield token

    # ======================================================================
    # Image generation
    # ======================================================================

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: int = 4,
        **kwargs: Any,
    ) -> List[str]:
        """Image generation interface.

        Args:
            prompt: Image description
            model: Model ID
            n: Number of images to generate
            **kwargs: Additional parameters

        Returns:
            List of image URLs (empty list on failure)
        """
        resolved_model = model or self.default_model
        client, actual_model = self._get_client(resolved_model)

        if client is None:
            _log("❌ generate_image: 没有可用客户端")
            return []

        if self._is_nebula_client(client) and self._nebula_client is not None:
            _log(f"🎨 generate_image: nebula cluster, prompt={prompt[:50]}...")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._nebula_client.generate_image,
                prompt,
                n,
            )

        _log(f"⚠️ generate_image: {actual_model} 不支持生图")
        return []

    # ======================================================================
    # JSON output
    # ======================================================================

    async def chat_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """JSON-formatted chat interface with automatic retry.

        Args:
            messages: List of message dicts
            model: Direct model ID
            role: Role-based model selection
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dict (or dict with error/parse_error keys)
        """
        max_json_attempts = 3
        local_messages = list(messages)  # Don't mutate original

        for attempt in range(max_json_attempts):
            raw = await self.chat(
                local_messages,
                model=model,
                role=role,
                **kwargs,
            )

            # Check for error responses
            if raw.startswith("[") and raw.endswith("]"):
                if attempt == max_json_attempts - 1:
                    return {"error": raw}
                continue

            # Try to extract and parse JSON
            cleaned = self._extract_json(raw)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                _log(f"JSON 解析失败 {attempt + 1}: {exc}")
                if attempt == max_json_attempts - 1:
                    return {"content": cleaned, "parse_error": str(exc)}

                # Add hint for retry
                local_messages.append({
                    "role": "system",
                    "content": "请只返回有效的 JSON 格式，不要包含任何其他文本。",
                })

        return {"content": raw, "error": "max_json_attempts_exceeded"}

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Extract JSON from markdown code blocks or raw text.

        Args:
            raw: Raw response text

        Returns:
            Cleaned JSON string
        """
        cleaned = raw.strip()

        # Remove ```json blocks
        if "```json" in cleaned:
            parts = cleaned.split("```json", 1)
            if len(parts) > 1:
                cleaned = parts[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) > 1:
                cleaned = parts[1].split("```", 1)[0].strip()

        return cleaned

    # ======================================================================
    # Health check
    # ======================================================================

    async def health_check(self) -> bool:
        """Check if any provider is reachable.

        Returns:
            True if at least one provider is healthy
        """
        for name, client in self._clients.items():
            if isinstance(client, dict) and client.get("type") == "cluster":
                continue

            try:
                await asyncio.wait_for(
                    client.models.list(),
                    timeout=5.0,
                )
                _log(f"✅ health check: {name} OK")
                return True
            except Exception as exc:
                _log(f"⚠️ health check: {name} failed: {exc}")
                continue

        return False