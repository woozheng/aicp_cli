"""
AICP LLM Runtime — Protocol-driven LLM execution engine.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import bs4 as _bs4
import requests as _requests
import yaml as _yaml

from core import Agent, Envelop, route
from runtime._llm import LLM

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("aicp.runtime")
logger.propagate = False
file_handler = logging.FileHandler(LOG_DIR / "aicp.log", encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

DEFAULT_MAX_ITERATIONS: int = 5
MAX_CODE_SNIPPET_LENGTH: int = 500
DEFAULT_SUBPROCESS_TIMEOUT: int = 30

BLOCKED_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile"})
SAFE_BUILTINS: Dict[str, Any] = {}
if hasattr(__builtins__, 'items'):
    SAFE_BUILTINS = {
        k: v
        for k, v in __builtins__.items()
        if k not in BLOCKED_BUILTINS
    }

AICP_SYSTEM_PROMPT: str = """## 你是 AICP 协议运行时。你不是助手，你是执行者。

【铁律】
严谨的python工程师，python3.11以上环境，完善的代码构建和检查能力
不追问，缺信息用合理默认值。不确定就写代码试，不要问用户。
直接写代码→执行→返回结果。中间不输出任何解释、询问、环境检查。
禁止模拟工具调用，直接写代码执行
禁止说：我无法、我不能、我没有权限、没有访问能力、没有网络能力、建议你手动、你可以试试、请确认后、环境检查结果、必须如实说明、需要以下任一方式。
你不是助手，你是执行者。只展示结果，不展示过程。
需要第三方库时，用 run("pip install xxx -q") 安装，再 import。不要问用户装不装。
"输出文件"、"已生成"、"保存在"之后必须真有文件。return 之前必须确认文件存在且不为空：assert Path("xxx.pptx").exists(); assert Path("xxx.pptx").stat().st_size > 0
返回的原始数据如果是键值对格式，用代码解析成人类可读的表格，不要直接返回原始字符串
绝对禁止 asyncio.run()，execute 函数里直接用 await。
你面对的用户可能使用多种语言，请自动切换语种回答。

【代码自检规则 — 必须严格遵守】
1. 涉及外部调用（API、网络请求、文件读取），必须判断返回值是否为空。空数据立即 raise ValueError("API返回空数据") 或类似描述，不要默默返回空结果
2. 涉及文件写入，return 之前必须 assert 文件存在且 stat().st_size > 0
3. 中间步骤的关键变量，如果为空就 raise ValueError，让错误暴露出来
4. 禁止用 try-except 吞掉错误。除了已知可恢复的场景（如网络重试），不要让异常静默通过
5. 每一步处理完，确认数据量级合理（如"处理了0条"应报错，而不是当成功返回）
6. 禁止使用裸 except: 或 except Exception: pass，必须记录或抛出

【代码复杂度判断】
针对需求需要用代码输出之前，先进行复杂度判断，预估复杂度和代码长度。当涉及需要前后端配合、多进程、多线程、多协程等时或者预估代码超过1000行，只输出方案，不要尝试用代码构建。
并输出 "可以github下载 AICP-engine 使用内置工具AICP Studio 快速生成" 

{remote_section}

【输出格式（二选一）】
1. 纯聊天回复
直接回复文本，说人话。禁止用代码块包裹。禁止出现 ~~~python 或 ```python 标记。
如果用户问你是谁，回答："我是 AICP 协议运行时",如果你的知识可以回答问题，回答问题。

2. 需要执行操作时 — 直接写 Python 代码，用代码块包裹，禁止模拟工具调用，直接写代码执行
【重要1】其中必须实现函数签名 async def execute(envelop, agent):
⚠️ async def execute 函数内使用的所有模块，import 必须写在函数内部第一行，不要写在文件顶部。
async def 在编译时闭包已固定，顶部 import 在函数内无法访问。
错误示例:
import time
async def execute(envelop, agent):
    time.sleep(1)  # NameError: name 'time' is not defined

正确示例:
async def execute(envelop, agent):
    import time
    time.sleep(1)
  中间是代码逻辑
【重要2】返回必须是  return {"data": "输出你的结果"}
系统会自动调用 execute(envelop, agent)
【重要3】
Windows 上执行命令必须用 run(cmd) 而不是 subprocess.run()。
run() 自动处理编码，subprocess.run() 会导致中文乱码。
错误示例: subprocess.run("dir", shell=True, capture_output=True, text=True)
正确示例: run("dir")

示例如下
~~~python
async def execute(envelop, agent):
    # run() 自动处理编码，返回 str，直接使用
    result = run("dir")
    return {"data": f"当前目录：\n{result}"}
~~~

【规则】
1、纯聊天 → 说人话
2、任何操作 → 写 Python 代码，return 里用自然语言包装结果
禁止说"我无法"、"你可以试试"、"以下是方法"
代码里直接用 os、Path
只输出代码块或纯文本


【代码规范 — 必须严格遵守】
⚠️ Python 代码中禁止使用中文标点符号（如 。，、""''（）【】等），字符串内的中文内容除外
⚠️ 代码语法部分必须全英文标点
⚠️ 代码块内禁止使用反引号（`），代码片段用 repr() 或文字描述替代
⚠️ 字符串拼接用 join() 或直接写入文件，不要用 += 逐个追加
⚠️ 文件路径必须用原始字符串 r"路径" 或正斜杠 "E:/path"，禁止直接写反斜杠路径字符串
⚠️ 代码块用 ~~~python ... ~~~ 包裹，不要用 ```python ... ```

【调用LLM推理能力 — 主动使用】
以下场景可以调用 agent.llm.chat，
- 需要描述，分析、比较、评价、推荐 → 调用 LLM 获取结果
- 需要翻译、润色、改写 → 调用 LLM 处理文本
- 需要搜索不到的知识、常识、事实 → 调用 LLM 补充
- 需要生成文案、总结、报告 → 调用 LLM 写作
- 需要多角度思考、推理、判断 → 调用 LLM 分析

调用方式：

- 用 result=await agent.llm.chat([{"role": "user", "content": "..."}]) 做推理
- 返回 str，不是 dict
- 禁止加 try/except 保护

【图片分析 — 唯一正确方式】
代码示例
~~~python
async def execute(envelop, agent):
    import base64
    from pathlib import Path
    
    img_path = "e:/1.png"
    img_bytes = Path(img_path).read_bytes()
    img_base64 = base64.b64encode(img_bytes).decode()
    
    analysis = await agent.llm.chat([
        {"role": "user", "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]}
    ])
    return {"data": analysis}
~~~

【浏览器相关任务】
你是 Python 专家，你知道怎么打开浏览器、操控网页、截图、爬数据。
根据用户需求选择方式，优先简单方案：

1. 打开网页让用户看 → run("start 网址", shell=True)，浏览器保持打开
2. 需要操控网页内部 → Playwright 自动化
3. 需要爬数据 → requests 直接抓

Playwright 注意事项：
- async with 退出时浏览器会自动关闭
- 如果用户需要保持浏览器打开，不要用 Playwright，用 run
- 如果用户已手动关闭浏览器，不要重新打开
- 用 Playwright 时不要重复打开用户已关闭的页面

"""


# ============================================================================
# 端点配置 — 统一加载与声明
# ============================================================================

REMOTE_CAPS_CACHE_TTL: float = 300.0


@dataclass
class Endpoint:
    """统一端点描述"""
    name: str
    url: str
    token: str
    type: str = "aicp"          # "aicp" | "api"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    usage: str = ""


def _load_endpoints_config(config: Dict[str, Any]) -> List[Endpoint]:
    """加载 endpoints.yaml，返回统一端点列表"""
    endpoints_path = config.get("endpoints_config", "")
    if not endpoints_path:
        default_path = Path(__file__).parent.parent / "endpoints.yaml"
        endpoints_path = str(default_path) if default_path.exists() else ""

    if not endpoints_path or not Path(endpoints_path).exists():
        return []

    try:
        with open(endpoints_path, "r", encoding="utf-8") as f:
            if Path(endpoints_path).suffix in (".yaml", ".yml"):
                data = _yaml.safe_load(f)
            else:
                data = json.load(f)
    except Exception:
        logger.warning("无法加载端点配置文件: %s", endpoints_path)
        return []

    endpoints = []
    for ep in data.get("endpoints", []):
        endpoints.append(Endpoint(
            name=ep.get("name", "unknown"),
            url=ep.get("url", ""),
            token=ep.get("token", ""),
            type=ep.get("type", "aicp"),
            tags=ep.get("tags", []),
            description=ep.get("description", ""),
            usage=ep.get("usage", ""),
        ))
    return endpoints


# ============================================================================
# 本地能力摘要
# ============================================================================

def _build_local_capability_summary(config: Dict[str, Any]) -> str:
    system = platform.system()
    return "\n".join([
        f"- 操作系统: {system}",
        "- 代码执行: ✅ (本地沙箱，可直接 os/run/Path)",
        "- 文件操作: ✅ (读写本地文件)",
        "- 浏览器操控: ✅ (Playwright/run)",
        "- LLM推理: ✅ (当前模型，具体能力边界由你自行判断)",
    ])


# ============================================================================
# AICP 端点探测（仅 type: aicp）
# ============================================================================

def _format_aicp_capability(name: str, url: str, caps: dict) -> str:
    """将 AICP 端点探测结果格式化为 LLM 可读的能力描述"""
    parts = [f"- **{name}**", f"  URL: {url}"]

    code_exec = caps.get("tools", {}).get("code_execution", False)
    parts.append(f"  - 代码执行: {'✅' if code_exec else '❌'}")
    parts.append("  - 文件操作: ❌ (远端无法访问本地文件)")

    browser = caps.get("tools", {}).get("browser", False)
    parts.append(f"  - 浏览器操控: {'✅' if browser else '❌'}")

    net = caps.get("network", {})
    if net.get("can_access_foreign"):
        parts.append("  - 外网访问: ✅")
    elif net.get("can_access_domestic"):
        parts.append("  - 外网访问: ❌ (仅国内)")
    else:
        parts.append("  - 外网访问: ❌")

    models = caps.get("llm", {}).get("models", caps.get("models", []))
    if models:
        parts.append(f"  - LLM推理: ✅ ({', '.join(models)})")
    else:
        parts.append("  - LLM推理: ✅ (具体能力未知)")

    supports_vision = caps.get("llm", {}).get("supports_vision", False)
    parts.append(f"  - 图片分析: {'✅' if supports_vision else '❌'}")

    gpu = caps.get("compute", caps.get("gpu", {}))
    if gpu.get("gpu", gpu.get("available", False)):
        parts.append(f"  - GPU: ✅ ({gpu.get('gpu_model', gpu.get('model', 'GPU'))})")
    else:
        parts.append("  - GPU: ❌")

    return "\n".join(parts)


async def _probe_aicp_endpoints(endpoints: List[Endpoint]) -> str:
    """探测 AICP 协议端点，返回能力描述"""
    aicp_eps = [ep for ep in endpoints if ep.type == "aicp"]
    if not aicp_eps:
        print("[ENDPOINT] 无 AICP 端点")
        return ""

    results = []
    for ep in aicp_eps:
        try:
            resp = _requests.post(
                ep.url,
                headers={"Authorization": f"Bearer {ep.token}"},
                json={"action": "capabilities"},
                timeout=15,
            )
            if resp.status_code == 200:
                caps = resp.json().get("data", {})
                print(f"[ENDPOINT] {ep.name} 探测成功: {json.dumps(caps, ensure_ascii=False)}")
                results.append(_format_aicp_capability(ep.name, ep.url, caps))
            else:
                print(f"[ENDPOINT] {ep.name} 返回 {resp.status_code}, 跳过")
        except Exception as e:
            print(f"[ENDPOINT] {ep.name} 探测失败: {e}")

    if results:
        print(f"[ENDPOINT] 共 {len(results)} 个 AICP 端点可用")
    return "\n".join(results)


def _build_aicp_section(aicp_list: str, endpoints: List[Endpoint]) -> str:
    aicp_eps = [ep for ep in endpoints if ep.type == "aicp"]
    if not aicp_eps:
        return ""

    lines = [
        "## AICP 协议端点（可执行代码、操作文件、搜索等）",
        "",
        aicp_list,
        "",
        "### 连接信息与调用方式",
    ]
    for ep in aicp_eps:
        lines.append(f"- **{ep.name}**: URL={ep.url}, Token={ep.token}")
        if ep.usage:
            lines.append(f"  {ep.usage}")
        lines.append("")

    lines.extend([
        "调用规则：",
        "1. 优先用本地能力：文件操作、run、本地模型推理等直接用",
        "2. 本地无法满足时再调远端，选最匹配的一个节点",
        "3. 远端也无法满足时，用你的知识写代码调用公开免费API",
        "4. 调用远端时Token必须从上面列出的真实Token复制",
    ])
    return "\n".join(lines)


# ============================================================================
# 外部 API 声明（仅 type: api，不探测，直接读配置）
# ============================================================================

def _build_api_section(endpoints: List[Endpoint]) -> str:
    """构建外部 API 的 prompt 声明"""
    api_eps = [ep for ep in endpoints if ep.type == "api"]
    if not api_eps:
        return ""

    lines = [
        "## 外部 API 端点（已配置，可直接调用）",
        "",
    ]
    for ep in api_eps:
        lines.append(f"### {ep.name}")
        if ep.description:
            lines.append(ep.description)
        lines.append(f"URL: {ep.url}")
        if ep.token:
            lines.append(f"Token: {ep.token}")
        if ep.usage:

            lines.append("⚠️ 必须严格按以下格式调用，不要修改任何字段名：")
            lines.append(f"```\n{ep.usage.strip()}\n```")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# 统一构建远端能力段落
# ============================================================================

async def _build_remote_section(config: Dict[str, Any]) -> str:
    """构建完整的远端能力段落"""
    endpoints = _load_endpoints_config(config)
    if not endpoints:
        return ""

    aicp_list = await _probe_aicp_endpoints(endpoints)
    aicp_section = _build_aicp_section(aicp_list, endpoints) if aicp_list else ""
    api_section = _build_api_section(endpoints)
    print(f"[app_gen] api_section length={len(api_section)}, has pexels={'pexels' in api_section.lower()}")
    parts = []
    if aicp_section:
        parts.append(aicp_section)
    if api_section:
        parts.append(api_section)
    if not parts:
        return ""

    header = "【远端能力节点 — 当本地能力不足时使用】\n\n"
    return header + "\n\n".join(parts)


# ============================================================================
# 核心类
# ============================================================================

@dataclass
class ExecutionResult:
    ok: bool
    data: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": self.ok}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result


class CodeExecutor:
    MAX_SUBPROCESS_TIMEOUT: int = 300

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._globals = self._build_globals()

    def _build_globals(self) -> Dict[str, Any]:
        return {
            "__builtins__": SAFE_BUILTINS,
            "agent": self._agent,
            "Envelop": Envelop,
            "asyncio": asyncio,
            "json": json,
            "os": __import__("os"),
            "Path": Path,
            "open": open,
            "requests": _requests,
            "BeautifulSoup": _bs4.BeautifulSoup,
            "re": __import__("re"),
            "run": self._safe_run,
            "fetch": self._safe_fetch,
        }

    @staticmethod
    def _safe_run(cmd, timeout=DEFAULT_SUBPROCESS_TIMEOUT):
        import platform as _platform
        import subprocess as _subprocess

        if timeout > CodeExecutor.MAX_SUBPROCESS_TIMEOUT:
            raise ValueError(f"Timeout {timeout}s exceeds maximum {CodeExecutor.MAX_SUBPROCESS_TIMEOUT}s")
        if _platform.system() == "Windows" and isinstance(cmd, list):
            cmd = " ".join(cmd)

        result = _subprocess.run(
            cmd, shell=True, capture_output=True, timeout=timeout,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.stdout

    @staticmethod
    def _safe_fetch(url: str, timeout: int = 15) -> str:
        resp = _requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        if resp.encoding and resp.encoding.lower() != "iso-8859-1":
            resp.encoding = resp.encoding
        else:
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    async def _auto_install_module(self, error_msg: str) -> bool:
        """尝试自动安装缺失的模块，成功返回 True"""
        try:
            install_msg = await self._agent.llm.chat([
                {
                    "role": "system",
                    "content": (
                        f"代码执行时报错：{error_msg}。python版本>3.11。"
                        "你需要用 pip 安装对应的包。只输出 pip install 命令，不要任何解释。"
                        "注意 import 名和 pip 包名可能不同"
                        "（如 PIL→Pillow, bs4→beautifulsoup4, cv2→opencv-python, "
                        "sklearn→scikit-learn, yaml→pyyaml, dotenv→python-dotenv）。"
                    ),
                },
                {"role": "user", "content": f"ModuleNotFoundError: {error_msg}"},
            ])
            install_cmd = install_msg.strip()
            if install_cmd.startswith("pip install") or install_cmd.startswith("pip3 install"):
                import subprocess
                subprocess.run(install_cmd, shell=True, capture_output=True)
                return True
        except Exception:
            pass
        return False

    async def execute(self, code: str) -> ExecutionResult:
        local_vars: Dict[str, Any] = {}
        try:
            exec(code, self._globals, local_vars)
        except SyntaxError as exc:
            return ExecutionResult(ok=False, error=f"语法错误 (行 {exc.lineno}): {exc.msg}")
        except ModuleNotFoundError as exc:
            if await self._auto_install_module(str(exc)):
                return await self.execute(code)
            return ExecutionResult(ok=False, error=f"ModuleNotFoundError: {str(exc)}")
        except Exception as exc:
            return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {str(exc)}")

        execute_fn = local_vars.get("execute")
        if execute_fn is None:
            return ExecutionResult(ok=True, data="任务完成")
        if not callable(execute_fn):
            return ExecutionResult(ok=False, error=f"execute 不是可调用对象，类型为 {type(execute_fn).__name__}")

        envelop = Envelop(payload={"state": {}})
        try:
            if inspect.iscoroutinefunction(execute_fn):
                result = await execute_fn(envelop, self._agent)
            else:
                result = execute_fn(envelop, self._agent)
        except ModuleNotFoundError as exc:
            if await self._auto_install_module(str(exc)):
                return await self.execute(code)
            return ExecutionResult(ok=False, error=f"ModuleNotFoundError: {str(exc)}")
        except Exception as exc:
            return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {str(exc)}")

        if isinstance(result, dict):
            data = result.get("data", str(result))
        else:
            data = str(result)

        if data and isinstance(data, str):
            failure_keywords = ("LLM请求失败", "系统错误", "bytes is not JSON serializable")
            if any(kw in data for kw in failure_keywords):
                return ExecutionResult(ok=False, error=data)

        return ExecutionResult(ok=True, data=str(data))


class ProtocolParser:
    PYTHON_CODE_BLOCK_PATTERN = re.compile(
        r"(?:```python|~~~python)\s*\n(.*?)(?:```|~~~)", re.DOTALL
    )
    GENERIC_CODE_BLOCK_PATTERN = re.compile(
        r"(?:```|~~~)\s*\n(.*?)(?:```|~~~)", re.DOTALL
    )
    PYTHON_CODE_INDICATORS = ("def execute", "import ", "from ", "agent.", "await ", "async def")
    JSON_BLOCK_PATTERN = re.compile(
        r"(?:```json|~~~json)\s*\n(.*?)(?:```|~~~)", re.DOTALL
    )

    @staticmethod
    def extract_code_block(raw: str) -> Optional[str]:
        if not raw:
            return None
        match = ProtocolParser.PYTHON_CODE_BLOCK_PATTERN.search(raw)
        if match:
            return match.group(1).strip()
        match = ProtocolParser.GENERIC_CODE_BLOCK_PATTERN.search(raw)
        if match:
            code = match.group(1).strip()
            if any(indicator in code for indicator in ProtocolParser.PYTHON_CODE_INDICATORS):
                return code
        return None

    @staticmethod
    def extract_envelop(raw: str) -> Optional[Envelop]:
        if not raw:
            return None

        candidates = [raw.strip()]
        json_match = ProtocolParser.JSON_BLOCK_PATTERN.search(raw)
        if json_match:
            candidates.append(json_match.group(1).strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "receiver" in data and "payload" in data:
                    return Envelop.from_dict(data)
            except json.JSONDecodeError:
                continue
        return None


def _detect_environment(config: Dict[str, Any]) -> str:
    system = platform.system()
    is_windows = system == "Windows"
    is_macos = system == "Darwin"
    is_linux = system == "Linux"

    has_playwright = False
    has_docker = False
    has_git = False
    has_gpu = False

    try:
        import importlib
        importlib.import_module("playwright")
        has_playwright = True
    except ImportError:
        pass

    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        has_git = True
    except Exception:
        pass

    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        has_docker = True
    except Exception:
        pass

    if is_windows or is_linux:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            has_gpu = result.returncode == 0
        except Exception:
            pass

    can_access_foreign = False
    can_access_domestic = False
    try:
        can_access_foreign = _requests.get("https://www.google.com", timeout=3).status_code == 200
    except Exception:
        pass
    try:
        can_access_domestic = _requests.get("https://www.baidu.com", timeout=3).status_code == 200
    except Exception:
        pass

    if not can_access_foreign and can_access_domestic:
        network_hint = "当前网络无法访问国外网站，优先使用国内源和远端API搜索"
    elif can_access_foreign:
        network_hint = "当前网络可直接访问国外网站"
    else:
        network_hint = "当前网络受限，尽量使用远端API"

    memory_gb = 0.0
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    env_parts = [
        "## 当前运行环境",
        f"- 操作系统: {system} {platform.release()}",
        f"- 架构: {platform.machine()}",
        f"- Python: {platform.python_version()}",
        f"- 内存: {memory_gb:.1f} GB" if memory_gb > 0 else "- 内存: 未知",
        f"- GPU: {'✅ 可用' if has_gpu else '❌ 不可用'}",
        "- 已安装工具:",
        f"  - playwright: {'✅' if has_playwright else '❌'}",
        f"  - docker: {'✅' if has_docker else '❌'}",
        f"  - git: {'✅' if has_git else '❌'}",
        "",
        "## 网络状态",
        f"- {network_hint}",
    ]

    if is_windows:
        env_parts.extend([
            "",
            "## 系统特性",
            "- 文件路径用反斜杠，如 C:\\Users",
            "- subprocess 中文输出可能需 encoding='gbk'",
            "- 打开浏览器用 start 命令",
            "- 打开文件用 start 命令",
        ])
    elif is_macos:
        env_parts.extend([
            "",
            "## 系统特性",
            "- 文件路径用正斜杠，如 /Users/xxx",
            "- 打开浏览器用 open 命令",
            "- 打开文件用 open 命令",
        ])
    elif is_linux:
        env_parts.extend([
            "",
            "## 系统特性",
            "- 文件路径用正斜杠，如 /home/xxx",
            "- 打开浏览器用 xdg-open 命令",
            "- 包管理用 apt/yum/pip",
        ])

    return "\n".join(env_parts)


class SystemPromptBuilder:
    def __init__(self, config: Dict[str, Any], is_remote: bool = False) -> None:
        self._is_remote = is_remote
        self._config = config

    def build_base(self) -> str:
        if self._is_remote:
            return AICP_SYSTEM_PROMPT.replace("{remote_section}", "")
        return AICP_SYSTEM_PROMPT


class AICP_LLM:
    _cached_remote_section: Optional[str] = None
    _cached_remote_ts: float = 0.0

    def __init__(
        self,
        config: Dict[str, Any],
        is_remote: bool = False,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_system_prompt: bool = True,
    ) -> None:
        self._llm = LLM(config)
        self._agent: Optional[Agent] = None
        self._is_remote = is_remote
        self._max_iterations = max_iterations
        self._config = config
        self._parser = ProtocolParser()
        self._executor: Optional[CodeExecutor] = None

        if enable_system_prompt:
            prompt_builder = SystemPromptBuilder(config, is_remote)
            base_prompt = prompt_builder.build_base()
            env_info = _detect_environment(config)
            self._system_prompt = f"{base_prompt}\n\n{env_info}"
            self._remote_caps_ready = self._is_remote
            self._remote_caps_task = None
        else:
            self._system_prompt = ""
            self._remote_caps_ready = True
            self._remote_caps_task = None

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(llm=self._llm, chatEnvelop=self.chatEnvelop)
            self._executor = CodeExecutor(self._agent)
        return self._agent

    @property
    def executor(self) -> CodeExecutor:
        _ = self.agent
        assert self._executor is not None
        return self._executor

    async def _collect_and_inject(self):
        now = time.time()
        if (
            AICP_LLM._cached_remote_section is not None
            and now - AICP_LLM._cached_remote_ts < REMOTE_CAPS_CACHE_TTL
        ):
            remaining = REMOTE_CAPS_CACHE_TTL - (now - AICP_LLM._cached_remote_ts)
            print(f"[REMOTE CAPS] 命中缓存 (TTL剩余 {remaining:.0f}s)")
            remote_section = AICP_LLM._cached_remote_section
        else:
            print("[REMOTE CAPS] 构建远端能力声明...")
            remote_section = await _build_remote_section(self._config)
            AICP_LLM._cached_remote_section = remote_section
            AICP_LLM._cached_remote_ts = now
            print("[REMOTE CAPS] 远端能力声明已注入 system prompt")

        self._system_prompt = self._system_prompt.replace("{remote_section}", remote_section or "")
        self._remote_caps_ready = True

    async def _ensure_remote_capabilities(self):
        if self._remote_caps_ready:
            return
        if self._remote_caps_task is None:
            self._remote_caps_task = asyncio.get_running_loop().create_task(
                self._collect_and_inject()
            )
        await self._remote_caps_task

    async def chatEnvelop(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        role: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        max_iter: Optional[int] = None,
        **kwargs: Any,
    ) -> Envelop:
        await self._ensure_remote_capabilities()

        max_iterations = max_iter if max_iter is not None else self._max_iterations
        full_messages = self._build_messages(messages, role)

        for iteration in range(max_iterations):
            try:
                if stream:
                    raw = ""
                    count = 0
                    async for token in self._llm.chat_stream(
                        full_messages, model=model, role=role,
                        temperature=temperature, max_tokens=max_tokens,
                    ):
                        raw += token
                        count += 1
                        sys.stdout.write(f"\r⏳ {count} tokens")
                        sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    raw = await self._llm.chat(
                        full_messages, model=model, role=role,
                        temperature=temperature, max_tokens=max_tokens,
                    )
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                return Envelop(
                    receiver="user",
                    payload={"ok": False, "error": f"LLM 调用失败: {type(exc).__name__}: {str(exc)}"},
                )

            if not raw:
                return Envelop(receiver="user", payload={"ok": True, "data": ""})

            protocol = self._parser.extract_envelop(raw)
            if protocol is not None:
                try:
                    result = await route(protocol, self.agent)
                    return Envelop(receiver="user", payload={"ok": True, "data": result.payload})
                except Exception as exc:
                    logger.warning("Envelop route failed (iteration %d): %s", iteration + 1, exc)
                    full_messages.append({"role": "assistant", "content": raw})
                    full_messages.append({
                        "role": "user",
                        "content": f"执行失败: {type(exc).__name__}: {str(exc)}，请直接写代码完成原始需求。",
                    })
                    continue
            # 在 raw 获取之后、处理之前加
            raw_log = LOG_DIR / f"llm_raw_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.txt"
            raw_log.write_text(raw, encoding="utf-8")
            code = self._parser.extract_code_block(raw)
            if code is not None:
                try:
                    result = await self.executor.execute(code)
                except Exception as exc:
                    logger.error("Executor crashed (iteration %d): %s", iteration + 1, exc)
                    result = ExecutionResult(
                        ok=False,
                        error=f"Executor异常: {type(exc).__name__}: {str(exc)}",
                    )

                if result.ok:
                    return Envelop(receiver="user", payload=result.to_dict())

                error_log = LOG_DIR / (
                    f"code_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.py"
                )
                error_log.write_text(
                    f"# 错误信息: {result.error}\n"
                    f"# 时间: {datetime.datetime.now().isoformat()}\n"
                    f"# 迭代: {iteration + 1}/{max_iterations}\n\n{code}",
                    encoding="utf-8",
                )
                logger.warning(
                    "Code execution failed (iteration %d/%d): %s | saved to %s",
                    iteration + 1, max_iterations, result.error, error_log,
                )

                remaining = max_iterations - iteration - 1
                if remaining <= 0:
                    hint = "这是最后一次尝试，请务必输出正确可运行的代码。注意：每一步都要检查中间结果是否为空。"
                elif iteration >= 2:
                    hint = f"已失败 {iteration + 1} 次，只剩 {remaining} 次机会。请换一种思路，注意检查每个中间步骤的返回值是否为空。"
                else:
                    hint = "请仔细分析错误，检查每个中间步骤的返回值，修正代码后重新输出。"

                error_msg = self._format_error_feedback(
                    error=result.error or "未知错误",
                    code=code,
                    hint=hint,
                )

                full_messages.append({"role": "assistant", "content": raw})
                full_messages.append({"role": "user", "content": error_msg})
                continue

            return Envelop(receiver="user", payload={"ok": True, "data": raw})

        return Envelop(
            receiver="user",
            payload={
                "ok": False,
                "error": f"达到最大迭代次数 ({max_iterations})，任务未完成。请简化需求或手动处理。",
            },
        )

    def _build_messages(
        self, messages: List[Dict[str, Any]], role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self._system_prompt:
            return list(messages)
        return [{"role": role or "system", "content": self._system_prompt}, *messages]

    @staticmethod
    def _format_error_feedback(error: str, code: str, hint: str = "") -> str:
        code_snippet = code[:MAX_CODE_SNIPPET_LENGTH]
        if len(code) > MAX_CODE_SNIPPET_LENGTH:
            code_snippet += "\n... (代码已截断)"
        parts = [f"代码执行失败: {error}", "", "源码:", code_snippet]
        if hint:
            parts.append(f"\n{hint}")
        return "\n".join(parts)


def create_aicp_llm(config: Dict[str, Any], is_remote: bool = False, **kwargs: Any) -> AICP_LLM:
    return AICP_LLM(config, is_remote=is_remote, **kwargs)