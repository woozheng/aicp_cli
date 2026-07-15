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
需要第三方库时，先 pip install，再 import。不要问用户装不装。
"输出文件"、"已生成"、"保存在"之后必须真有文件。return 之前必须确认文件存在：assert Path("xxx.pptx").exists()
返回的原始数据如果是键值对格式，用代码解析成人类可读的表格，不要直接返回原始字符串
绝对禁止 asyncio.run()，execute 函数里直接用 await。
你面对的用户可能使用多种语言，请自动切换语种回答。

【代码复杂度判断】
针对需求需要用代码输出之前，先进行复杂度判断，预估复杂度和代码长度。当涉及需要前后端配合、多进程、多线程、多协程等时或者预估代码超过1000行，只输出方案，不要尝试用代码构建。
并输出 "可以github下载 AICP-engine 使用内置工具AICP Studio 快速生成" 

{remote_section}

【输出格式（二选一）】
1. 纯聊天回复
直接回复文本，说人话。如果用户问你是谁，回答："我是 AICP 协议运行时",如果你的知识可以回答问题，回答问题。

2. 需要执行操作时 — 直接写 Python 代码，用代码块包裹，禁止模拟工具调用，直接写代码执行
示例：
~~~python
async def execute(envelop, agent):
    import os, subprocess
    from pathlib import Path
    result = subprocess.run("dir", shell=True, capture_output=True, text=True)
    return {"data": f"当前目录：\\n{result.stdout}"}
~~~

【规则】
1、纯聊天 → 说人话
2、任何操作 → 写 Python 代码，return 里用自然语言包装结果
禁止说"我无法"、"你可以试试"、"以下是方法"
代码里直接用 os、subprocess、Path
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

1. 打开网页让用户看 → subprocess.run("start 网址", shell=True)，浏览器保持打开
2. 需要操控网页内部 → Playwright 自动化
3. 需要爬数据 → requests 直接抓

Playwright 注意事项：
- async with 退出时浏览器会自动关闭
- 如果用户需要保持浏览器打开，不要用 Playwright，用 subprocess
- 如果用户已手动关闭浏览器，不要重新打开
- 用 Playwright 时不要重复打开用户已关闭的页面

"""


# ============================================================================
# 端点配置与远端能力收集
# ============================================================================

REMOTE_CAPS_CACHE_TTL: float = 300.0  # 类级别缓存有效期，5分钟


@dataclass
class RemoteNode:
    name: str
    url: str
    token: str
    tags: List[str] = field(default_factory=list)


def _load_endpoints_config(config: Dict[str, Any]) -> List[RemoteNode]:
    """加载端点配置文件，支持 yaml/json"""
    endpoints_path = config.get("endpoints_config", "")
    if not endpoints_path:
        default_path = Path(__file__).parent.parent / "endpoints.yaml"
        if default_path.exists():
            endpoints_path = str(default_path)
        else:
            return []

    path = Path(endpoints_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                data = _yaml.safe_load(f)
            else:
                data = json.load(f)
    except Exception:
        logger.warning(f"无法加载端点配置文件: {path}")
        return []

    nodes = []
    for ep in data.get("endpoints", []):
        nodes.append(RemoteNode(
            name=ep.get("name", "unknown"),
            url=ep.get("url", ""),
            token=ep.get("token", ""),
            tags=ep.get("tags", []),
        ))
    return nodes


# ============================================================================
# 本地与远端能力格式化
# ============================================================================

def _build_local_capability_summary(config: Dict[str, Any]) -> str:
    """生成本地能力摘要，只列确定的事实，模型能力写自行判断"""
    system = platform.system()
    lines = [
        f"- 操作系统: {system}",
        "- 代码执行: ✅ (本地沙箱，可直接 os/subprocess/Path)",
        "- 文件操作: ✅ (读写本地文件)",
        "- 浏览器操控: ✅ (Playwright/subprocess)",
        "- LLM推理: ✅ (当前模型，具体能力边界由你自行判断)",
    ]
    return "\n".join(lines)


def _format_capability_for_llm(name: str, url: str, caps: dict) -> str:
    """将远端能力 dict 翻译成 LLM 可理解的自然语言描述，与本地能力维度对齐"""
    parts = [f"- **{name}**"]
    parts.append(f"  URL: {url}")

    # 代码执行
    code_exec = caps.get("tools", {}).get("code_execution", False)
    parts.append(f"  - 代码执行: {'✅' if code_exec else '❌'}")

    # 文件操作（远端一般不直接操作本地文件）
    parts.append("  - 文件操作: ❌ (远端无法访问本地文件)")

    # 浏览器操控
    browser = caps.get("tools", {}).get("browser", False)
    parts.append(f"  - 浏览器操控: {'✅' if browser else '❌'}")

    # 网络
    net = caps.get("network", {})
    if net.get("can_access_foreign"):
        parts.append("  - 外网访问: ✅")
    elif net.get("can_access_domestic"):
        parts.append("  - 外网访问: ❌ (仅国内)")
    else:
        parts.append("  - 外网访问: ❌")

    # LLM 推理
    models = caps.get("llm", {}).get("models", caps.get("models", []))
    if models:
        models_str = ', '.join(models)
        parts.append(f"  - LLM推理: ✅ ({models_str})")
    else:
        parts.append("  - LLM推理: ✅ (具体能力未知)")

    # 图片分析
    supports_vision = caps.get("llm", {}).get("supports_vision", False)
    parts.append(f"  - 图片分析: {'✅' if supports_vision else '❌'}")

    # GPU
    gpu = caps.get("compute", caps.get("gpu", {}))
    if gpu.get("gpu", gpu.get("available", False)):
        model = gpu.get("gpu_model", gpu.get("model", "GPU"))
        parts.append(f"  - GPU: ✅ ({model})")
    else:
        parts.append("  - GPU: ❌")

    return "\n".join(parts)


async def _collect_remote_capabilities(endpoints: List[RemoteNode]) -> str:
    """收集所有远端节点的能力，返回拼装好的远端节点列表"""
    if not endpoints:
        print("[REMOTE CAPS] 无端点配置")
        return ""

    remote_infos = []

    for ep in endpoints:
        try:
            resp = _requests.post(
                ep.url,
                headers={"Authorization": f"Bearer {ep.token}"},
                json={"action": "capabilities"},
                timeout=15
            )
            if resp.status_code == 200:
                caps = resp.json().get("data", {})
                print(f"[REMOTE CAPS] {ep.name} 探测成功: {json.dumps(caps, ensure_ascii=False)}")
                desc = _format_capability_for_llm(ep.name, ep.url, caps)
                remote_infos.append(desc)
            else:
                print(f"[REMOTE CAPS] {ep.name} 返回非200: {resp.status_code}, 跳过")
        except Exception as e:
            print(f"[REMOTE CAPS] {ep.name} 探测失败: {e}, 跳过")

    if not remote_infos:
        print("[REMOTE CAPS] 无可用远端节点")
        return ""

    return "\n".join(remote_infos)


def _build_capability_section(local_summary: str, remote_list: str, endpoints: List[RemoteNode]) -> str:
    """组装本地与远端能力对比段落"""
    lines = [
        "【本地与远端能力对比】",
        "",
        "## 本地能力",
        "本地已具备的能力，优先使用，无需调远端：",
        local_summary,
        "",
        "## 远端能力节点（仅本地无法完成时使用）",
        "以下远端节点作为补充，仅当本地能力不足时调用：",
        "",
        remote_list,
        "",
        "## 远端节点连接信息（代码中直接使用以下URL和Token）",
        "",
    ]
    
    for ep in endpoints:
        lines.append(f"- **{ep.name}**: URL={ep.url}, Token={ep.token}")
    
    lines.extend([
        "",
        "调用方式（直接复制修改）：",
        "~~~python",
        "import requests",
        "",
        "# 示例：调用远端节点做数据分析",
        "resp = requests.post(",
        '    "http://127.0.0.1:9000/api/builtins/aicp/chat",  # 替换为上面列出的实际URL',
        '    headers={"Authorization": "Bearer sk-your-token-here"},  # 替换为上面列出的实际Token',
        "    json={",
        '        "messages": [',
        '            {"role": "user", "content": "请分析特斯拉2024年Q3财报，只返回JSON格式，不要其他文字"}',
        "        ]",
        "    }",
        ")",
        "",
        "result = resp.json().get(\"data\", \"\")",
        "# result 是远端返回的结果，用Python解析后继续处理",
        "~~~",
        "",
        "【调用规则】",
        "1. 优先用本地能力：文件操作、subprocess、本地模型推理等直接用",
        "2. 本地无法满足时（如需要外网搜索、需要图片分析但本地不支持、需要GPU等），再调远端",
        "3. 调用远端时选最匹配的一个节点，不要重复调多个",
        "4. 如果远端节点也无法满足需求，用你的知识写代码调用公开免费API，不要直接说做不到",
        "5. 调用远端时Token必须从上面列出的真实Token复制，禁止用占位符",
    ])
    return "\n".join(lines)


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
            "subprocess": __import__("subprocess"),
            "Path": Path,
            "open": open,
            "requests": _requests,
            "BeautifulSoup": _bs4.BeautifulSoup,
            "re": __import__("re"),
            "run": self._safe_run,
            "fetch": self._safe_fetch,
        }

    @staticmethod
    def _safe_run(cmd: Union[str, List[str]], timeout: int = DEFAULT_SUBPROCESS_TIMEOUT) -> str:
        import platform as _platform
        import subprocess as _subprocess

        if timeout > CodeExecutor.MAX_SUBPROCESS_TIMEOUT:
            raise ValueError(f"Timeout {timeout}s exceeds maximum {CodeExecutor.MAX_SUBPROCESS_TIMEOUT}s")

        if _platform.system() == "Windows" and isinstance(cmd, list):
            cmd = " ".join(cmd)

        try:
            result = _subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout, text=False)
            for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]:
                try:
                    stdout = result.stdout.decode(enc)
                    if stdout.strip():
                        return stdout
                except (UnicodeDecodeError, LookupError):
                    continue
            return result.stdout.decode("utf-8", errors="replace")
        except _subprocess.TimeoutExpired:
            raise
        except Exception:
            raise

    @staticmethod
    def _safe_fetch(url: str, timeout: int = 15) -> str:
        try:
            resp = _requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp.raise_for_status()
            if resp.encoding and resp.encoding.lower() != "iso-8859-1":
                resp.encoding = resp.encoding
            else:
                resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except _requests.RequestException:
            raise

    async def execute(self, code: str) -> ExecutionResult:
        local_vars: Dict[str, Any] = {}
        try:
            exec(code, self._globals, local_vars)
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
            except Exception as exc:
                return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {str(exc)}")

            if isinstance(result, dict):
                data = result.get("data", str(result))
            else:
                data = str(result)
            if data and isinstance(data, str) and ("LLM请求失败" in data or "系统错误" in data or "bytes is not JSON serializable" in data):
                 return ExecutionResult(ok=False, error=data)
            return ExecutionResult(ok=True, data=str(data))
        except SyntaxError as exc:
            return ExecutionResult(ok=False, error=f"语法错误 (行 {exc.lineno}): {exc.msg}")
        except Exception as exc:
            return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {str(exc)}")


class ProtocolParser:
    PYTHON_CODE_BLOCK_PATTERN = re.compile(r"(?:```python|~~~python)\s*\n(.*?)(?:```|~~~)", re.DOTALL)
    GENERIC_CODE_BLOCK_PATTERN = re.compile(r"(?:```|~~~)\s*\n(.*?)(?:```|~~~)", re.DOTALL)
    PYTHON_CODE_INDICATORS = ("def execute", "import ", "from ", "agent.", "await ", "async def")
    JSON_BLOCK_PATTERN = re.compile(r"(?:```json|~~~json)\s*\n(.*?)(?:```|~~~)", re.DOTALL)

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
        try:
            data = json.loads(raw.strip())
            if ProtocolParser._is_envelop_dict(data):
                return Envelop.from_dict(data)
        except json.JSONDecodeError:
            pass
        match = ProtocolParser.JSON_BLOCK_PATTERN.search(raw)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if ProtocolParser._is_envelop_dict(data):
                    return Envelop.from_dict(data)
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _is_envelop_dict(data: Any) -> bool:
        return isinstance(data, dict) and "receiver" in data and "payload" in data


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
        resp = _requests.get("https://www.google.com", timeout=3)
        can_access_foreign = resp.status_code == 200
    except Exception:
        pass
    
    try:
        resp = _requests.get("https://www.baidu.com", timeout=3)
        can_access_domestic = resp.status_code == 200
    except Exception:
        pass
    
    network_hint = ""
    if not can_access_foreign and can_access_domestic:
        network_hint = "当前网络无法访问国外网站，优先使用国内源和远端API搜索"
    elif can_access_foreign:
        network_hint = "当前网络可直接访问国外网站"
    else:
        network_hint = "当前网络受限，尽量使用远端API"

    memory_gb = 0.0
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
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
    # 类级别缓存：同一进程内所有实例共享，避免重复探测
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
        """收集远端能力并注入 system prompt（类级别缓存，含本地与远端对比）"""
        print("[REMOTE CAPS] 开始收集远端能力...")

        now = time.time()
        if (
            AICP_LLM._cached_remote_section is not None
            and now - AICP_LLM._cached_remote_ts < REMOTE_CAPS_CACHE_TTL
        ):
            print("[REMOTE CAPS] 命中类级别缓存，跳过网络探测")
            remote_section = AICP_LLM._cached_remote_section
        else:
            endpoints = _load_endpoints_config(self._config)
            print(f"[REMOTE CAPS] 加载到 {len(endpoints)} 个端点")
            remote_list = await _collect_remote_capabilities(endpoints)

            if remote_list:
                local_summary = _build_local_capability_summary(self._config)
                remote_section = _build_capability_section(local_summary, remote_list, endpoints)
            else:
                remote_section = ""

            AICP_LLM._cached_remote_section = remote_section
            AICP_LLM._cached_remote_ts = now

        if remote_section:
            self._system_prompt = self._system_prompt.replace("{remote_section}", remote_section)
        else:
            self._system_prompt = self._system_prompt.replace("{remote_section}", "")

        self._remote_caps_ready = True
        print("[REMOTE CAPS] 远端能力收集完成，已注入 system prompt")

    async def _ensure_remote_capabilities(self):
        """确保远端能力已收集"""
        if self._remote_caps_ready:
            return

        if self._remote_caps_task is None:
            loop = asyncio.get_running_loop()
            self._remote_caps_task = loop.create_task(self._collect_and_inject())

        await self._remote_caps_task

    async def chatEnvelop(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        role: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
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
                    arrows = ['→', '⇒', '⟹', '⟶', '⟼', '⤏', '⇢', '⇨']
                    i = 0
                    async for token in self._llm.chat_stream(full_messages, model=model, role=role, temperature=temperature, max_tokens=max_tokens):
                        raw += token
                        if len(raw) % 80 == 0:
                            print(f"\r⏳ {arrows[i % len(arrows)]} ", end="", flush=True)
                            i += 1
                    print("\r" + " " * 20 + "\r", end="", flush=True)
                else:
                    raw = await self._llm.chat(full_messages, model=model, role=role, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                return Envelop(receiver="user", payload={"ok": False, "error": f"LLM 调用失败: {type(exc).__name__}: {str(exc)}"})

            if not raw:
                return Envelop(receiver="user", payload={"ok": True, "data": ""})

            protocol = self._parser.extract_envelop(raw)
            if protocol is not None:
                try:
                    result = await route(protocol, self.agent)
                    return Envelop(receiver="user", payload={"ok": True, "data": result.payload})
                except Exception as exc:
                    full_messages.append({"role": "assistant", "content": raw})
                    full_messages.append({"role": "system", "content": f"执行失败: {str(exc)}，请直接写代码完成"})
                    continue

            code = self._parser.extract_code_block(raw)
            if code is not None:
                result = await self.executor.execute(code)
                if result.ok:
                    return Envelop(receiver="user", payload=result.to_dict())

                error_log = LOG_DIR / f"code_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.py"
                error_log.write_text(
                    f"# 错误信息: {result.error}\n"
                    f"# 时间: {datetime.datetime.now().isoformat()}\n"
                    f"# 迭代: {iteration + 1}/{max_iterations}\n\n"
                    f"{code}",
                    encoding="utf-8"
                )

                hint = "已失败多次，请换一种方式。" if iteration >= 2 else ""
                error_msg = self._format_error_feedback(result.error or "未知错误", code, hint)
                logger.warning("Code execution failed (iteration %d): %s | saved to %s", iteration + 1, result.error, error_log)
                print(f"\r⚠️ 尝试实现需求失败，正在重试 ({iteration + 1}/{max_iterations})...", file=sys.stderr)

                full_messages.append({"role": "assistant", "content": raw})
                full_messages.append({"role": "system", "content": error_msg})
                continue

            return Envelop(receiver="user", payload={"ok": True, "data": raw})

        return Envelop(receiver="user", payload={"ok": False, "error": f"达到最大迭代次数 ({max_iterations})，任务未完成"})

    def _build_messages(self, messages: List[Dict[str, Any]], role: Optional[str] = None) -> List[Dict[str, Any]]:
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