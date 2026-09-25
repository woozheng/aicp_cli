#!/usr/bin/env python3
"""
AICP CLI — 极简命令行接口。

模式:
  python cli.py               本地模式（直接调用 AICP 引擎，endpoints.yaml 中非 studio 端点作为能力补充）
  python cli.py -r            远端模式（完全依赖远端节点，从 endpoints.yaml 读取）
  python cli.py --studio      Studio 模式（连接 main_agent，使用 content 格式，session_id 由 token 决定）
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

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "aicp.yaml"
DEFAULT_ENDPOINTS_PATH: Path = Path(__file__).resolve().parent / "endpoints.yaml"
DEFAULT_TIMEOUT_SECONDS: int = 300
EXIT_COMMANDS: frozenset[str] = frozenset({"/exit", "/quit", "exit"})

logger = logging.getLogger("aicp.cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def _show_progress(stop_event: threading.Event) -> None:
    frames = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r⏳ {frames[i % len(frames)]} ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


@dataclass(frozen=True)
class RemoteConfig:
    endpoint: str
    token: str = ""

    @classmethod
    def from_endpoints_config(cls, config: Dict[str, Any], node_name: Optional[str] = None,
                              exclude_studio: bool = False, prefer_studio: bool = False) -> Optional[RemoteConfig]:
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
            logger.error("PyYAML is required to read endpoints config. Install: pip install pyyaml")
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

        if exclude_studio:
            endpoints = [ep for ep in endpoints if "studio" not in ep.get("name", "").lower()]

        if node_name:
            for ep in endpoints:
                if ep.get("name") == node_name:
                    return cls(endpoint=ep.get("url", ""), token=ep.get("token", ""))
            logger.warning("Node '%s' not found", node_name)
            return None

        if prefer_studio:
            for ep in endpoints:
                if "studio" in ep.get("name", "").lower():
                    return cls(endpoint=ep.get("url", ""), token=ep.get("token", ""))

        if endpoints:
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
    STUDIO = "studio"


class HttpClient(ABC):
    @abstractmethod
    def post_json(self, url: str, payload: Dict[str, Any], token: str = "") -> Dict[str, Any]:
        ...


class HttpxClient(HttpClient):
    def __init__(self, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if httpx is None:
            raise ImportError("httpx is required for remote mode. Install: pip install httpx")
        self._timeout = timeout

    def post_json(self, url: str, payload: Dict[str, Any], token: str = "") -> Dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
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


class ChatEngine(ABC):
    @abstractmethod
    def send(self, user_message: str) -> ChatResponse:
        ...

    @property
    @abstractmethod
    def banner(self) -> str:
        ...


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
                    stream=self._stream,
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


class RemoteChatEngine(ChatEngine):
    def __init__(
        self,
        config: RemoteConfig,
        http: HttpClient,
        studio_mode: bool = False,
    ) -> None:
        self._url = config.endpoint
        self._token = config.token
        self._http = http
        self._studio_mode = studio_mode
        # ✅ token 固定 → session_id 固定，main_agent 能恢复会话
        self._session_id = config.token or "cli_default"

    @property
    def banner(self) -> str:
        if self._studio_mode:
            session_display = self._session_id[:8] + "..." if len(self._session_id) > 8 else self._session_id
            return f"=== AICP Studio 模式 (main_agent) | session={session_display} | /exit 退出 ===\n"
        return f"=== AICP 远端模式 | {self._url} | /exit 退出 ===\n"

    def send(self, user_message: str) -> ChatResponse:
        stop = threading.Event()
        t = threading.Thread(target=_show_progress, args=(stop,), daemon=True)
        t.start()
        try:
            if self._studio_mode:
                payload = {
                    "content": user_message,
                    "session_id": self._session_id,
                    "channel": "cli",
                }
            else:
                payload = {"messages": [{"role": "user", "content": user_message}]}

            raw = self._http.post_json(self._url, payload, self._token)
        finally:
            stop.set()
            t.join(timeout=0.5)

        # studio 模式：main_agent 返回 {"type": "chat", "content": "..."}
        if self._studio_mode:
            if "error" in raw:
                return ChatResponse.fail(raw.get("error", "未知错误"))
            if "content" in raw:
                return ChatResponse.ok(raw["content"])
            if "message" in raw:
                return ChatResponse.ok(raw["message"])
            if "data" in raw:
                return ChatResponse.ok(raw["data"])
            if raw:
                return ChatResponse.ok(json.dumps(raw, ensure_ascii=False))
            return ChatResponse.fail("无响应")

        # 非 studio 模式：标准格式
        if raw.get("ok"):
            return ChatResponse.ok(raw.get("message", raw.get("data", "")))
        return ChatResponse.fail(raw.get("error", "未知错误"))


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
                sys.stdout.buffer.write(f"🤖 {response.content}\n".encode("utf-8"))
            else:
                sys.stdout.buffer.write(f"❌ {response.error}\n".encode("utf-8"))
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
                logger.critical("Remote mode requires an endpoint in endpoints.yaml. Use -n to specify one.")
                sys.exit(1)
            return RemoteChatEngine(remote_cfg, create_http_client(), studio_mode=False)

        if self._mode == ChatMode.STUDIO:
            remote_cfg = RemoteConfig.from_endpoints_config(self._config, self._node_name, prefer_studio=True)
            if remote_cfg is None:
                logger.critical("Studio mode requires a studio endpoint in endpoints.yaml. Use -n to specify one.")
                sys.exit(1)
            return RemoteChatEngine(remote_cfg, create_http_client(), studio_mode=True)

        return LocalChatEngine(self._config, stream=self._stream)


def parse_args(argv: List[str]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {
        "remote": False,
        "studio": False,
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
        elif arg in ("-s", "--studio"):
            flags["studio"] = True
        elif arg in ("-v", "--verbose"):
            flags["verbose"] = True
        elif arg == "--no-stream":
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


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

    args = parse_args(sys.argv)
    setup_logging(verbose=args["verbose"])

    if args["studio"]:
        mode = ChatMode.STUDIO
    elif args["remote"]:
        mode = ChatMode.REMOTE
    else:
        mode = ChatMode.LOCAL

    logger.info(
        "Starting AICP CLI | config=%s | mode=%s | node=%s | stream=%s",
        args["config"], mode.value, args["node"], args["stream"],
    )

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