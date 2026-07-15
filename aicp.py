#!/usr/bin/env python3
"""
AICP CLI — Production-grade command-line interface.

Provides both local and remote chat modes with robust error handling,
comprehensive logging, and clean architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import ssl
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from runtime._config import load_config
from runtime._aicp_llm import AICP_LLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "aicp.yaml"
DEFAULT_ENDPOINTS_PATH: Path = Path(__file__).resolve().parent / "endpoints.yaml"
DEFAULT_TIMEOUT_SECONDS: int = 120
EXIT_COMMANDS: frozenset[str] = frozenset({"/exit", "/quit", "exit"})

logger = logging.getLogger("aicp.cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Terminal spinner
# ---------------------------------------------------------------------------

def _show_progress(stop_event, retry_count=0):
    frames = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        msg = f"\r⏳ {frames[i % len(frames)]}"
        if retry_count > 0:
            msg += f" (重试 {retry_count}/5)"
        sys.stdout.write(msg + " ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteConfig:
    endpoint: str
    token: str = ""

    @classmethod
    def from_endpoints_config(cls, config: Dict[str, Any], node_name: Optional[str] = None) -> Optional[RemoteConfig]:
        """从 endpoints.yaml 读取远端节点配置"""
        endpoints_path = config.get("endpoints_config", "")
        if not endpoints_path:
            if DEFAULT_ENDPOINTS_PATH.exists():
                endpoints_path = str(DEFAULT_ENDPOINTS_PATH)
            else:
                return None

        path = Path(endpoints_path)
        if not path.exists():
            return None

        if _yaml is None:
            logger.error("PyYAML is required to read endpoints config. Install it with: pip install pyyaml")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
        except Exception:
            logger.exception("Failed to load endpoints config: %s", path)
            return None

        endpoints = data.get("endpoints", [])
        if not endpoints:
            return None

        for ep in endpoints:
            if node_name is None or ep.get("name") == node_name:
                return cls(
                    endpoint=ep.get("url", ""),
                    token=ep.get("token", ""),
                )

        if node_name:
            logger.warning("Node '%s' not found in endpoints.yaml, using first available", node_name)
            ep = endpoints[0]
            return cls(endpoint=ep.get("url", ""), token=ep.get("token", ""))

        return None


@dataclass(frozen=True)
class ChatResponse:
    success: bool
    content: str = ""
    error: str = ""

    @classmethod
    def ok(cls, content: str) -> ChatResponse:
        return cls(success=True, content=content)

    @classmethod
    def fail(cls, error: str) -> ChatResponse:
        return cls(success=False, error=error)


class ChatMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"


# ---------------------------------------------------------------------------
# HTTP client abstraction
# ---------------------------------------------------------------------------


class HttpClient(ABC):
    @abstractmethod
    def post_json(self, url: str, payload: Dict[str, Any], token: str = "") -> Dict[str, Any]:
        ...


class HttpxClient(HttpClient):
    def __init__(self, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if httpx is None:
            raise ImportError("httpx is required for remote mode. Install it with: pip install httpx")
        self._timeout = timeout

    def post_json(self, url: str, payload: Dict[str, Any], token: str = "") -> Dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %d calling %s: %s", exc.response.status_code, url, exc)
            return {"ok": False, "error": f"HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return {"ok": False, "error": str(exc)}


class StdlibHttpClient(HttpClient):
    def __init__(self, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def post_json(self, url: str, payload: Dict[str, Any], token: str = "") -> Dict[str, Any]:
        import http.client

        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"

        if parsed.scheme == "https":
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(host, port, timeout=self._timeout, context=ctx)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=self._timeout)

        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            return json.loads(data or "{}")
        except (http.client.HTTPException, OSError, json.JSONDecodeError) as exc:
            logger.error("HTTP request failed for %s: %s", url, exc)
            return {"ok": False, "error": str(exc)}
        finally:
            conn.close()


def create_http_client() -> HttpClient:
    if httpx is not None:
        return HttpxClient()
    return StdlibHttpClient()


# ---------------------------------------------------------------------------
# Chat engines
# ---------------------------------------------------------------------------


class ChatEngine(ABC):
    @abstractmethod
    def send(self, user_message: str) -> ChatResponse:
        ...

    @property
    @abstractmethod
    def banner(self) -> str:
        ...


class RemoteChatEngine(ChatEngine):
    def __init__(self, config: RemoteConfig, http: HttpClient) -> None:
        self._endpoint = config.endpoint
        self._token = config.token
        self._http = http

    @property
    def banner(self) -> str:
        return f"=== AICP 远端模式 | {self._endpoint} | /exit 退出 ===\n"

    def send(self, user_message: str) -> ChatResponse:
        stop = threading.Event()
        t = threading.Thread(target=_show_progress, args=(stop,), daemon=True)
        t.start()
        try:
            raw = self._http.post_json(
                self._endpoint,
                {"messages": [{"role": "user", "content": user_message}]},
                self._token,
            )
        finally:
            stop.set()
            t.join(timeout=0.5)
        if raw.get("ok"):
            return ChatResponse.ok(raw.get("data", ""))
        return ChatResponse.fail(raw.get("error", "未知错误"))


class LocalChatEngine(ChatEngine):
    def __init__(self, config: Dict[str, Any], stream: bool = True) -> None:
        self._aicp = AICP_LLM(config)
        self._stream = stream

    @property
    def banner(self) -> str:
        return "=== AICP CLI (local) | /exit 退出 ===\n"

    def send(self, user_message: str) -> ChatResponse:
        stop = threading.Event()
        t = None
        if not self._stream:
            t = threading.Thread(target=_show_progress, args=(stop,), daemon=True)
            t.start()
        try:
            result = asyncio.run(
                self._aicp.chatEnvelop(
                    [{"role": "user", "content": user_message}],
                    stream=self._stream
                )
            )
        except Exception as exc:
            logger.exception("Local chat failed")
            return ChatResponse.fail(str(exc))
        finally:
            stop.set()
            if t is not None:
                t.join(timeout=0.5)
        if result.payload.get("ok"):
            return ChatResponse.ok(result.payload.get("data", ""))
        return ChatResponse.fail(result.payload.get("error", "未知错误"))


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------


class ChatCLI:
    def __init__(self, engine: ChatEngine) -> None:
        self._engine = engine
        self._running = True

    def run(self) -> None:
        self._setup_signal_handlers()
        print(self._engine.banner)

        while self._running:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                self._shutdown("👋 再见")
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in EXIT_COMMANDS:
                self._shutdown("👋 再见")
                break

            response = self._engine.send(stripped)

            if response.success:
                print(f"\r🤖 {response.content}")
            else:
                print(f"\r❌ {response.error}")
            print()

    def _setup_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down", signum)
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass

    @staticmethod
    def _shutdown(message: str) -> None:
        print(f"\n{message}")
        logger.info("CLI session ended")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


class Application:
    def __init__(
        self,
        config_path: Path,
        mode: ChatMode,
        node_name: Optional[str] = None,
        stream: bool = True,
    ) -> None:
        self._config = self._load_configuration(config_path)
        self._mode = mode
        self._node_name = node_name
        self._stream = stream

    @staticmethod
    def _load_configuration(path: Path) -> Dict[str, Any]:
        try:
            return load_config(str(path))
        except FileNotFoundError:
            logger.critical("Configuration file not found: %s", path)
            sys.exit(1)
        except Exception as exc:
            logger.critical("Failed to load configuration: %s", exc)
            sys.exit(1)

    def create_engine(self) -> ChatEngine:
        if self._mode == ChatMode.REMOTE:
            remote_cfg = RemoteConfig.from_endpoints_config(self._config, self._node_name)
            if remote_cfg is None:
                logger.critical(
                    "Remote mode selected but no endpoints found in endpoints.yaml. "
                    "Please create endpoints.yaml with at least one endpoint."
                )
                sys.exit(1)
            return RemoteChatEngine(remote_cfg, create_http_client())
        return LocalChatEngine(self._config, stream=self._stream)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> Dict[str, Any]:
    flags = {
        "remote": False,
        "verbose": False,
        "stream": True,
        "config": str(DEFAULT_CONFIG_PATH),
        "node": None,
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-r", "--remote"):
            flags["remote"] = True
        elif arg in ("-v", "--verbose"):
            flags["verbose"] = True
        elif arg in ("--no-stream",):
            flags["stream"] = False
        elif arg in ("-n", "--node"):
            if i + 1 < len(argv):
                flags["node"] = argv[i + 1]
                i += 1
        elif arg.startswith("-c"):
            if "=" in arg:
                flags["config"] = arg.split("=", 1)[1]
            else:
                if i + 1 < len(argv):
                    flags["config"] = argv[i + 1]
                    i += 1
        i += 1
    return flags


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args(sys.argv)
    setup_logging(verbose=args["verbose"])
    logger.info(
        "Starting AICP CLI | config=%s | remote=%s | node=%s | stream=%s",
        args["config"], args["remote"], args["node"], args["stream"],
    )
    mode = ChatMode.REMOTE if args["remote"] else ChatMode.LOCAL
    app = Application(
        Path(args["config"]),
        mode,
        node_name=args["node"],
        stream=args["stream"],
    )
    engine = app.create_engine()
    cli = ChatCLI(engine)
    cli.run()


if __name__ == "__main__":
    main()