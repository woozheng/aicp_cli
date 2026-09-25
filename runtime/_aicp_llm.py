"""
AICP LLM Runtime — Protocol-driven LLM execution engine.
完全对齐版：AST预检 + 最小清洗 + 无记忆沙箱 + 提示词对齐 + 自动装包 + 超时 + 死循环检测
"""

from __future__ import annotations

import asyncio
import datetime
import importlib
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
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, FrozenSet

import bs4 as _bs4
import requests as _requests
import yaml as _yaml

from core import Agent, Envelop, route
from runtime._llm import LLM

# ============================================================
# 日志
# ============================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("aicp.runtime")
logger.propagate = False
file_handler = logging.FileHandler(LOG_DIR / "aicp.log", encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

# ============================================================
# 常量
# ============================================================

DEFAULT_MAX_ITERATIONS: int = 5
MAX_CODE_SNIPPET_LENGTH: int = 500
DEFAULT_SUBPROCESS_TIMEOUT: int = 30
DEFAULT_TIMEOUT_SECONDS: int = 60
MAX_WHITELIST_ERRORS: int = 2

# ============================================================
# AST 安全校验
# ============================================================

BLOCKED_BUILTINS: FrozenSet[str] = frozenset({
    "eval", "exec", "compile", "open", "globals", "locals", "__import__"
})

import builtins
SAFE_BUILTINS: Dict[str, Any] = {
    k: v
    for k, v in builtins.__dict__.items()
    if k not in BLOCKED_BUILTINS and not k.startswith("_")
}

# ============================================================
# 安全护栏：危险命令黑名单
# ============================================================

DANGEROUS_COMMANDS = [
    "format ",
    "rm -rf /",
    "rm -rf ~",
    "del /f /s /q c:\\",
    "del /f /s /q d:\\",
    "shutdown",
    "mkfs",
    "dd if=/dev/zero",
    "diskpart",
    "fdisk",
    "format c:",
]

BLOCKED_FETCH_HOSTS = [
    "169.254.169.254",
    "metadata.google.internal",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
]

class SandboxBlockError(RuntimeError):
    """沙箱主动拦截时抛的异常，与 LLM 自己抛的 RuntimeError 区分开"""
    pass


# ============================================================
# 工具函数
# ============================================================

def _build_sandbox_hint() -> str:
    return (
        "当前沙箱规则：\n"
        "- subprocess.run 支持列表形式，或字符串形式\n"
        "- 字符串命令允许管道 `|`，但每段命令必须在白名单内：\n"
        "  grep/head/tail/sort/uniq/wc/cat/cut/tr/sed/awk/find/ls/echo/python/python3/markitdown/pdftoppm/soffice\n"
        "- 字符串命令禁止：`>` `<` `$` 反引号 `;` `&`\n"
        "- 需要重定向/写文件，请在 Python 里用 open() 处理\n"
        "- 禁 os.chdir，用绝对路径或 subprocess.run(..., cwd=...)\n"
        "- 禁 subprocess.Popen，一律用 subprocess.run\n"
        "- 禁 print()，用 return {\"data\": ...}\n"
        "- fetch 仅 http/https，禁内网地址\n"
        "- 需要额外依赖，在代码块前用 ~~~packages ... ~~~ 声明，自动 pip install\n"
    )


def _check_dangerous_command(cmd: str):
    if not isinstance(cmd, str):
        return cmd
    cmd_lower = cmd.lower()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            raise SandboxBlockError(f"危险命令被禁止: {dangerous}")
    return cmd


def _check_fetch_url(url: str):
    if not isinstance(url, str):
        raise SandboxBlockError("fetch url 必须是字符串")
    if not url.startswith(("http://", "https://")):
        raise SandboxBlockError(f"fetch 只支持 http/https: {url}")
    url_lower = url.lower()
    for blocked in BLOCKED_FETCH_HOSTS:
        if blocked in url_lower:
            raise SandboxBlockError(f"fetch 禁止访问: {blocked}")
    return url


# ============================================================
# ★★★ 完全对齐的 System Prompt ★★★
# ============================================================

SYSTEM_PROMPT_APPEND = """### 运行环境重要说明
1. 你的代码必须定义 async def execute(envelop, agent):，所有业务逻辑写在函数内部，禁止顶层执行代码；禁止if __name__=="__main__"，禁止裸except:
2. execute最终必须 return {"data": 结果内容}，只能返回该格式字典
3. subprocess.run 和内置 run() 都会对字符串命令做安全校验：
   - 允许管道 |，但每段命令需在白名单内（grep/head/tail/sort/uniq/wc/cat/cut/tr/sed/awk/find/ls/echo/python/python3/markitdown/pdftoppm/soffice）
   - 禁止 > < $ ` ; &
   - 需要重定向/写文件，请在 Python 里用 open()
4. 执行器只会最小处理HTML转义字符，不会自动修复你的代码；报错使用标记分隔原始源码，堆栈行号是清洗后代码，会存在少量行偏移。
5. 需要额外依赖时，在代码块前用 ~~~packages ... ~~~ 声明，系统会自动 pip install；不需要声明的用已预装库。
6. 禁止用同步 requests 调用本地 API (http://127.0.0.1)，会阻塞事件循环，请用 aiohttp。
7. 禁止使用 print()，所有输出必须通过 return {"data": ...} 返回。
8. 临时默认成果落盘 data/workspace 需判断目录是否存在，没有则自动建立,用户有指定输出目录的优先
9. 禁止 time.sleep()，需要等待用 await asyncio.sleep(N)。禁止 input()。禁止 while True。执行超时 60 秒。
"""

AICP_SYSTEM_PROMPT: str = """## 你是 AICP 协议运行时。你是执行者，不是助手。

═══════════════════════════════════════
【最高优先级 — 代码生成铁律（不可违反）】
═══════════════════════════════════════
⚠️ 返回格式强制：return {"data": "你的结果"}，字段名必须是 "data"
⚠️ 禁止写 print()，禁止任何控制台输出
⚠️ 禁止用 try-except 吞掉错误，错误必须暴露
⚠️ 禁止写 if __name__ == "__main__": 和 def main():
⚠️ 所有 import 必须写在 execute 函数内部第一行
⚠️ async def execute(envelop, agent): 必须有，且只能有一个
⚠️ 禁止使用任何原生 function calling 或 tool_call 格式
⚠️ 禁止输出 <tool_call>、<function_call>、<invoke> 等标签
⚠️ 所有工具调用必须通过 Python 代码块实现

═══════════════════════════════════════
【输出格式 — 三种模式】
═══════════════════════════════════════

## 模式1：纯聊天
直接说人话，友好回复。

## 模式2：写代码执行（不需要额外依赖）
输出 Python 代码块，用 ~~~python ... ~~~ 包裹（三个波浪号 + python）

正确示例：
~~~python
async def execute(envelop, agent):
    import requests
    result = requests.get("https://api.example.com")
    return {"data": result.text}
~~~

## 模式3：写代码执行（需要额外依赖）
先用 ~~~packages ... ~~~ 声明依赖，再用 ~~~python ... ~~~ 写代码

正确示例：
~~~packages
pandas
requests
~~~
~~~python
async def execute(envelop, agent):
    import pandas as pd
    import requests
    df = pd.DataFrame({"a": [1, 2, 3]})
    return {"data": df.to_dict()}
~~~

⚠️ 依赖会自动 pip install，无需手动装。
⚠️ 请尽量用已预装的库（requests、bs4、yaml、aiohttp 等）。

═══════════════════════════════════════
【subprocess.run 铁律 — 最高优先级】
═══════════════════════════════════════
⚠️ 禁止同时使用 shell=True 和列表参数！
✅ 正确：subprocess.run(["python", "script.py"], capture_output=True, text=True)
❌ 错误：subprocess.run(["python", "script.py"], shell=True)
❌ 错误：subprocess.run("python script.py", shell=True)

⚠️ 禁止使用 subprocess.Popen，一律用 subprocess.run
⚠️ Windows 路径用正斜杠 "E:/path" 或原始字符串 r"E:\\path"
⚠️ 优先用 Python 原生库（Path、os、requests），避免执行系统命令

═══════════════════════════════════════
【超时铁律 — 最高优先级】
═══════════════════════════════════════
⚠️ 执行超时默认 60 秒，超时后会被强制中断。
⚠️ 禁止 time.sleep()，需要等待用 await asyncio.sleep(N)。
⚠️ 禁止 input()，沙箱里没有交互输入。
⚠️ 禁止 while True / while 1，必须用有限循环。
⚠️ 长任务请分批处理，单次执行不超过 60 秒。

═══════════════════════════════════════
【网络请求铁律】
═══════════════════════════════════════
⚠️ 禁止用同步 requests 调用本地 API（http://127.0.0.1:9000/api/xxx）
⚠️ 会阻塞事件循环，导致网关死锁！
✅ 用 aiohttp 异步调用：
async with aiohttp.ClientSession() as session:
    async with session.post(url, json={...}) as resp:
        result = await resp.json()

═══════════════════════════════════════
【代码自检规则】
═══════════════════════════════════════
1. 外部调用（API/网络/文件）必须检查返回是否为空，空数据立即 raise ValueError
2. 文件写入后必须 assert 文件存在且大小 > 0
3. 中间关键变量为空立即 raise ValueError
4. 禁止裸 except: 或 except Exception: pass

═══════════════════════════════════════
【重试规则 — 重要】
═══════════════════════════════════════
⚠️ 如果代码执行失败，分析原因后一次性修正，不要每次只改一点点。
⚠️ 如果连续两次失败原因相同，直接回复"无法完成"，不要继续尝试。
⚠️ 禁止 while True / while 1 等死循环，必须用有限循环。
⚠️ 执行超时默认 60 秒，长任务请分批处理。

═══════════════════════════════════════
【LLM 调用规范】
═══════════════════════════════════════
- 需要自然语言理解、文本生成、翻译、摘要、分类、推理时，调用 agent.llm
- 不需要 LLM 的纯计算、文件操作、数据转换，禁止调用 LLM（浪费 token）

文本调用：
  result = await agent.llm.chat([
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
  ])
  result 是 str，不是 dict

JSON 调用：
  result = await agent.llm.chat_json([...])
  result 是 dict

图片分析（多模态）：
  import base64
  with open(img_path, "rb") as f:
      b64 = base64.b64encode(f.read()).decode()
  result = await agent.llm.chat([
      {"role": "user", "content": [
          {"type": "text", "text": "分析这张图片"},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
      ]}
  ])
  content 用数组格式（不是字符串）

⚠️ 不要自己读环境变量 OPENAI_API_KEY / LLM_API_KEY。
   agent.llm 已经配好了 API Key，直接调就行。


═══════════════════════════════════════
【浏览器相关任务】
═══════════════════════════════════════
你是 Python 专家，你知道怎么打开浏览器、操控网页、截图、爬数据。
根据用户需求选择方式，优先简单方案：

1. 打开网页让用户看 → run 启动系统命令（Windows: start，macOS: open，Linux: xdg-open），浏览器保持打开
2. 需要操控网页内部 → Playwright 自动化
3. 需要爬数据 → requests 直接抓

Playwright 注意事项：
- async with 退出时浏览器会自动关闭
- 如果用户需要保持浏览器打开，不要用 Playwright，用 run
- 如果用户已手动关闭浏览器，不要重新打开
- 用 Playwright 时不要重复打开用户已关闭的页面


═══════════════════════════════════════
【远端能力节点】
═══════════════════════════════════════
{remote_section}
""" + "\n\n" + SYSTEM_PROMPT_APPEND


# ============================================================
# 端点配置
# ============================================================

REMOTE_CAPS_CACHE_TTL: float = 300.0


@dataclass
class Endpoint:
    name: str
    url: str
    token: str
    type: str = "aicp"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    usage: str = ""


def _load_endpoints_config(config: Dict[str, Any], exclude_studio: bool = False) -> List[Endpoint]:
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
            if not data:
                return []
    except Exception:
        logger.warning("无法加载端点配置文件: %s", endpoints_path)
        return []

    endpoints = []
    for ep in data.get("endpoints", []):
        name = ep.get("name", "unknown")
        if exclude_studio and "studio" in name.lower():
            continue
        endpoints.append(Endpoint(
            name=name,
            url=ep.get("url", ""),
            token=ep.get("token", ""),
            type=ep.get("type", "aicp"),
            tags=ep.get("tags", []),
            description=ep.get("description", ""),
            usage=ep.get("usage", ""),
        ))
    return endpoints


# ============================================================
# 代码清理与解析
# ============================================================

def minimal_sanitize_code(raw_text: str):
    """最小清洗：只处理 HTML 转义，不改语法"""
    original = raw_text
    s = raw_text

    html_entities = ("&nbsp;", "&gt;", "&lt;", "&amp;", "&quot;", "&#39;", "&apos;")
    if any(kw in s for kw in html_entities):
        s = s.replace("&nbsp;", " ")
        s = s.replace("&gt;", ">")
        s = s.replace("&lt;", "<")
        s = s.replace("&amp;", "&")
        s = s.replace("&quot;", '"')
        s = s.replace("&#39;", "'")
        s = s.replace("&apos;", "'")

    lines = [line.rstrip() for line in s.splitlines()]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return original, s





def ast_detect_blocking(tree: ast.AST) -> Optional[str]:
    """AST 静态检测死循环和阻塞调用"""
    for node in ast.walk(tree):
        # while True / while 1
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant):
                if node.test.value in (True, 1):
                    return f"while {node.test.value}"

        # 阻塞调用
        if isinstance(node, ast.Call):
            func = node.func
            # time.sleep(...) / sleep(...)
            if isinstance(func, ast.Attribute) and func.attr == "sleep":
                return "time.sleep"
            if isinstance(func, ast.Name) and func.id == "sleep":
                return "sleep"
            # input()
            if isinstance(func, ast.Name) and func.id == "input":
                return "input"

    return None


def ast_pre_check(source: str) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"语法解析错误：第{e.lineno}行 {e.msg}"

    has_execute_func = False

    # 只检查模块顶层
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return "违反提示词铁律：禁止顶层执行代码，全部业务逻辑必须写在 execute 函数内部"
        if isinstance(node, ast.If):
            test_node = node.test
            if isinstance(test_node, ast.Compare):
                left = test_node.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return "违反提示词铁律：禁止使用 if __name__ == \"__main__\""

    # 全局遍历：execute 参数
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
            has_execute_func = True
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names != ["envelop", "agent"]:
                return f"违反提示词铁律：async def execute(envelop, agent) 参数必须严格为(envelop, agent)，实际参数列表：{arg_names}"

    if not has_execute_func:
        return "违反提示词铁律：代码中必须定义 async def execute(envelop, agent) 入口函数"

    # 死循环 / 阻塞检测
    blocking = ast_detect_blocking(tree)
    if blocking:
        return (
            f"违反提示词铁律：检测到阻塞调用（{blocking}）。\n"
            f"请改用有限循环，且不要用 time.sleep / input。\n"
            f"如果需要等待，用 await asyncio.sleep(N)，N 为有限秒数。"
        )

    return None


def extract_python_code_block(text: str) -> str:
    """从 markdown 提取 python 代码块，支持 ~~~python 和 ```python

    没有代码块时返回空字符串。
    """
    pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)

    pattern2 = re.compile(r"~~~python\s*\n(.*?)~~~", re.DOTALL)
    match2 = pattern2.search(text)
    if match2:
        return match2.group(1)

    return ""


# ============================================================
# 包管理：自动安装
# ============================================================

_installed_packages: set = set()
_install_lock: Optional[asyncio.Lock] = None


def extract_packages(text: str) -> List[str]:
    """从 LLM 输出中提取 ~~~packages ... ~~~ 块"""
    m = re.search(r"~~~packages\s*\n(.*?)~~~", text, re.DOTALL)
    if not m:
        return []

    packages = []
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 去掉版本号：pandas==2.0.0 → pandas
        # 去掉 extras：pandas[all] → pandas
        pkg = re.split(r"[=<>!\[]", line)[0].strip()
        if pkg:
            packages.append(pkg)
    return packages


async def ensure_packages(packages: List[str]) -> None:
    """自动安装缺失的包"""
    global _install_lock

    if not packages:
        return

    # 检查哪些缺失
    missing = []
    for pkg in packages:
        if pkg in _installed_packages:
            continue
        # pip 包名和 import 名可能不同，尝试 import
        import_name = pkg.replace("-", "_")
        try:
            importlib.import_module(import_name)
            _installed_packages.add(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    # 并发安全
    if _install_lock is None:
        _install_lock = asyncio.Lock()

    async with _install_lock:
        # 双重检查
        still_missing = []
        for pkg in missing:
            if pkg in _installed_packages:
                continue
            import_name = pkg.replace("-", "_")
            try:
                importlib.import_module(import_name)
                _installed_packages.add(pkg)
            except ImportError:
                still_missing.append(pkg)

        if not still_missing:
            return

        logger.info(f"[ensurePackages] 开始安装: {still_missing}")

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", *still_missing,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"pip install 失败（exit={proc.returncode}）：{err[:500]}")

        for pkg in still_missing:
            _installed_packages.add(pkg)

        # 清 import 缓存（让新装的包能被 import）
        importlib.invalidate_caches()

        logger.info(f"[ensurePackages] 安装完成: {still_missing}")


# ============================================================
# 安全 OS 包装
# ============================================================

class SafeOS:
    def __init__(self):
        self._os = __import__("os")

    def __getattr__(self, name):
        if name == "chdir":
            raise SandboxBlockError(
                "os.chdir is disabled. Use absolute paths or run(cmd, cwd='...') "
                "to execute commands in a specific directory."
            )
        return getattr(self._os, name)

    def __dir__(self):
        return dir(self._os)


# ============================================================
# 执行结果
# ============================================================

@dataclass
class ExecutionResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    raw_llm_code: str = ""
    sanitized_code: str = ""
    error_category: str = ""   # LLM_CODE / SANDBOX_BLOCK / RUNTIME_INTERNAL / SYSTEM

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": self.ok}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        if self.error_category:
            result["error_category"] = self.error_category
        return result

    @property
    def fixable_by_llm(self) -> bool:
        """LLM 能不能自己修"""
        return self.error_category in ("LLM_CODE", "SANDBOX_BLOCK")


# ============================================================
# 核心执行器 — 无记忆沙箱
# ============================================================
# 允许出现在管道里的命令（只读、无副作用）
PIPE_SAFE_COMMANDS = {
    # 文本处理
    "grep", "head", "tail", "sort", "uniq", "wc", "cat", "cut",
    "tr", "sed", "awk", "echo",

    # 文件
    "find", "ls", "dir", "tree",

    # 进程 / 系统
    "ps", "tasklist", "wmic", "systeminfo",
    "whoami", "hostname", "date", "uptime",

    # 网络
    "ping", "netstat", "ipconfig", "ifconfig",

    # 磁盘
    "df", "du",

    # Python / 工具
    "python", "python3", "pip", "markitdown", "pdftoppm", "soffice",
}


def _split_pipeline(cmd: str) -> list:
    return [seg.strip().split()[0] for seg in cmd.split("|") if seg.strip()]


def _check_pipeline(cmd: str):
    """检查管道里每一段是否都是安全命令"""
    for head in _split_pipeline(cmd):
        # 去掉路径前缀，只看命令名
        name = head.split("/")[-1].split("\\")[-1]
        if name not in PIPE_SAFE_COMMANDS:
            raise SandboxBlockError(
                f"管道中的命令 `{name}` 不在安全白名单内。\n"
                f"允许的管道命令：{', '.join(sorted(PIPE_SAFE_COMMANDS))}"
            )

# ============================================================
# 超时 trace（运行时检测）
# ============================================================

def _build_timeout_tracer(timeout_seconds: float):
    """构建超时检测的 trace 钩子

    在每行 / 每次调用时检查是否超时。
    能中断纯 Python 死循环，但无法中断 C 层阻塞（如 time.sleep）。
    """
    start_time = time.time()

    def trace_func(frame, event, arg):
        # 只在关键事件上检查
        if event in ("line", "call"):
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"执行超时（{timeout_seconds}秒）")
        return trace_func

    return trace_func

class NoMemoryCodeExecutor:
    def __init__(self, agent: Optional[Agent] = None):
        self._agent = agent

    def _build_safe_globals(self) -> Dict[str, Any]:
        g = dict(SAFE_BUILTINS)

        # 原始 requests 引用（用于包装）
        _original_requests_get = _requests.get
        _original_requests_post = _requests.post

        # ============================================================
        # ★ 安全 subprocess：不偷偷改 shell 参数
        # ============================================================
        def _safe_subprocess_run(cmd, **kwargs):
            if isinstance(cmd, str):
                _check_dangerous_command(cmd)
                if "|" in cmd:
                    _check_pipeline(cmd)
                for c in (">", "<", "`", "$", ";", "&"):
                    if c in cmd:
                        raise SandboxBlockError(
                            f"字符串命令中不允许 `{c}`。\n"
                            f"替代写法：\n"
                            f"  - 重定向到文件：在 Python 里 open() 写\n"
                            f"  - 读取文件：在 Python 里 open() 读\n"
                            f"  - 管道：用 | 且命令在安全白名单内（如 grep/head/wc）\n"
                            f"  - 变量：在 Python 里拼接字符串\n"
                            f"  - 多命令：拆成多次 subprocess.run 调用"
                        )

            kwargs.setdefault("capture_output", True)
            kwargs.setdefault("text", True)
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("errors", "replace")
            res = subprocess.run(cmd, **kwargs)
            res.stdout = res.stdout or ""
            res.stderr = res.stderr or ""
            return res

        class SubprocessProxy:
            run = staticmethod(_safe_subprocess_run)
            PIPE = subprocess.PIPE
            STDOUT = subprocess.STDOUT
            DEVNULL = subprocess.DEVNULL
            TimeoutExpired = subprocess.TimeoutExpired
            CalledProcessError = subprocess.CalledProcessError

            @staticmethod
            def Popen(*args, **kwargs):
                raise SandboxBlockError(
                    "❌ 禁止使用 subprocess.Popen。\n"
                    "原因：Popen 会创建后台进程，可能与主进程产生资源冲突。\n"
                    "正确做法：用 subprocess.run(...)，它是同步阻塞的，更安全。"
                )

        # ============================================================
        # ★ 安全 requests：拦截本地 API
        # ============================================================
        def _safe_requests_get(url, *args, **kwargs):
            if url.startswith(("http://127.0.0.1", "http://localhost")):
                raise SandboxBlockError("❌ 禁止用同步 requests 调用本地 API (http://127.0.0.1)，请用 aiohttp")
            return _original_requests_get(url, *args, **kwargs)

        def _safe_requests_post(url, *args, **kwargs):
            if url.startswith(("http://127.0.0.1", "http://localhost")):
                raise SandboxBlockError("❌ 禁止用同步 requests 调用本地 API (http://127.0.0.1)，请用 aiohttp")
            return _original_requests_post(url, *args, **kwargs)

        class SafeRequests:
            get = staticmethod(_safe_requests_get)
            post = staticmethod(_safe_requests_post)

        # ============================================================
        # ★ 禁用 print
        # ============================================================
        def _noop_print(*args, **kwargs):
            raise SandboxBlockError(
                "❌ 禁止使用 print()。\n"
                "原因：沙箱里的 print 不会产生输出，会丢失结果。\n"
                "正确做法：return {\"data\": 你的结果}"
            )

        # ============================================================
        # ★ 安全 run（shell 模式，用于快速命令）
        # ============================================================
        def _safe_run(cmd, timeout=DEFAULT_SUBPROCESS_TIMEOUT, cwd=None):
            import platform as _platform

            if timeout > 300:
                raise ValueError(f"Timeout {timeout}s exceeds maximum 300s")
            if _platform.system() == "Windows" and isinstance(cmd, list):
                cmd = " ".join(cmd)

            if isinstance(cmd, str):
                _check_dangerous_command(cmd)
                if "|" in cmd:
                    _check_pipeline(cmd)
                for c in (">", "<", "`", "$", ";", "&"):
                    if c in cmd:
                        raise SandboxBlockError(f"run() 字符串命令中不允许 `{c}`")

            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=timeout,
                text=True, encoding="utf-8", errors="replace",
                cwd=cwd,
            )
            return result.stdout or ""

        # ============================================================
        # ★ 安全 fetch
        # ============================================================
        def _safe_fetch(url: str, timeout: int = 15) -> str:
            _check_fetch_url(url)
            resp = _original_requests_get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp.raise_for_status()
            if resp.encoding and resp.encoding.lower() != "iso-8859-1":
                resp.encoding = resp.encoding
            else:
                resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text

        g.update({
            "agent": self._agent,
            "Envelop": Envelop,
            "asyncio": asyncio,
            "json": json,
            "os": SafeOS(),
            "Path": Path,
            "open": open,
            "requests": SafeRequests,
            "BeautifulSoup": _bs4.BeautifulSoup,
            "re": __import__("re"),
            "run": _safe_run,
            "fetch": _safe_fetch,
            "subprocess": SubprocessProxy,
            "print": _noop_print,
        })

        return g

    def _build_error_result(
        self,
        category: str,
        summary: str,
        reason: str,
        detail: str = "",
        raw_code: str = "",
        sanitized_code: str = "",
    ) -> ExecutionResult:
        fixable = category in ("LLM_CODE", "SANDBOX_BLOCK")

        lines = [
            f"【错误分类】{category}",
            f"【一句话】{summary}",
            f"【原因】{reason}",
            f"【可修复】{'yes' if fixable else 'no'}",
        ]

        if fixable:
            lines.append("【处理】这是你的代码的问题，请根据下面的信息修改后重新输出完整代码。")
        else:
            lines.append("【处理】这是 runtime 内部问题，你的代码没有问题，不要尝试修改代码。")

        if category == "SANDBOX_BLOCK":
            lines.append("")
            lines.append(_build_sandbox_hint())

        if detail:
            lines.append(f"【详情】\n{detail}")

        if raw_code:
            lines.append(f"---RAW_CODE_BEGIN---\n{raw_code}\n---RAW_CODE_END---")

        return ExecutionResult(
            ok=False,
            error="\n".join(lines),
            raw_llm_code=raw_code,
            sanitized_code=sanitized_code,
            error_category=category,
        )

    async def run(self, llm_output_text: str, envelop: Any, agent: Any) -> ExecutionResult:
        # === 前置检查：输入必须是字符串 ===
        if llm_output_text is None:
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="run() 收到 None 作为输入",
                reason="调用方没有传入有效的代码文本。",
                detail=f"期望 str，实际 {type(llm_output_text).__name__}",
            )
        if not isinstance(llm_output_text, str):
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="run() 输入类型错误",
                reason="调用方传入的不是字符串。",
                detail=f"期望 str，实际 {type(llm_output_text).__name__}",
            )

        # === 代码提取（调用方已提取，直接使用） ===
        raw_extracted = llm_output_text

        # === 代码清洗 ===
        try:
            raw_code, sanitized_code = minimal_sanitize_code(raw_extracted)
        except Exception as e:
            tb_str = traceback.format_exc()
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="minimal_sanitize_code 内部异常",
                reason=f"{type(e).__name__}: {e}",
                detail=f"堆栈：\n{tb_str}",
                raw_code=raw_extracted,
            )

        if sanitized_code is None:
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="minimal_sanitize_code 返回 None",
                reason="代码清洗环节出错，未返回有效字符串。",
                raw_code=raw_code,
            )

        # === AST 预检 ===
        try:
            ast_err = ast_pre_check(sanitized_code)
        except Exception as e:
            tb_str = traceback.format_exc()
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="ast_pre_check 内部异常",
                reason=f"{type(e).__name__}: {e}",
                detail=f"堆栈：\n{tb_str}",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        if ast_err is not None:
            return self._build_error_result(
                category="LLM_CODE",
                summary="AST 静态校验未通过",
                reason=ast_err,
                detail="请修正代码结构，重新输出。",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        # === 构建沙箱 ===
        try:
            safe_globals = self._build_safe_globals()
        except Exception as e:
            tb_str = traceback.format_exc()
            return self._build_error_result(
                category="RUNTIME_INTERNAL",
                summary="构建执行沙箱失败",
                reason="runtime 内部的 safe_globals 构建出错。",
                detail=f"{type(e).__name__}: {e}\n{tb_str}",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        local_ns: Dict[str, Any] = {}

        # === exec 编译执行 ===
        try:
            exec(sanitized_code, safe_globals, local_ns)
        except SyntaxError as e:
            return self._build_error_result(
                category="LLM_CODE",
                summary="代码存在语法错误",
                reason=f"第 {e.lineno} 行: {e.msg}",
                detail="请检查括号闭合、缩进、字符串引号。",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )
        except Exception as e:
            return self._build_error_result(
                category="LLM_CODE",
                summary="代码加载/编译失败",
                reason=f"{type(e).__name__}: {e}",
                detail="请检查语法、缩进、import 是否写在函数内。",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        # === 检查 execute 入口 ===
        execute_fn = local_ns.get("execute")
        if execute_fn is None:
            return self._build_error_result(
                category="LLM_CODE",
                summary="找不到 execute 入口函数",
                reason="代码中必须定义 async def execute(envelop, agent)。",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )
        if not asyncio.iscoroutinefunction(execute_fn):
            return self._build_error_result(
                category="LLM_CODE",
                summary="execute 不是 async 函数",
                reason="必须定义为 async def execute(envelop, agent)。",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        # ============================================================
        # === 真正执行（三层防护）===
        # 第 1 层：AST 静态检测（已在前面做了）
        # 第 2 层：sys.settrace 运行时超时（拦纯 Python 死循环）
        # 第 3 层：asyncio.wait_for 超时（拦 await 阻塞）
        # ============================================================

        tracer = _build_timeout_tracer(DEFAULT_TIMEOUT_SECONDS)
        old_trace = sys.gettrace()
        sys.settrace(tracer)

        try:
            ret = await asyncio.wait_for(
                execute_fn(envelop, agent),
                timeout=DEFAULT_TIMEOUT_SECONDS + 5,  # 略大于 trace 超时
            )

        except asyncio.TimeoutError:
            # 第 3 层：asyncio 超时（拦 await 阻塞）
            return self._build_error_result(
                category="LLM_CODE",
                summary="执行超时",
                reason=f"代码执行超过 {DEFAULT_TIMEOUT_SECONDS} 秒（asyncio 层拦截）",
                detail=(
                    "可能原因：\n"
                    "  - 代码里有 await 长时间阻塞\n"
                    "  - 代码里有同步阻塞（如 time.sleep），导致事件循环卡住\n"
                    "请检查代码，避免长时间阻塞。"
                ),
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        except TimeoutError as e:
            # 第 2 层：sys.settrace 超时（拦纯 Python 死循环）
            return self._build_error_result(
                category="LLM_CODE",
                summary="执行超时（trace 触发）",
                reason=str(e),
                detail=(
                    "代码执行超过限制时间，可能是死循环或大量计算。\n"
                    "请改用有限循环，并控制单次执行时长。"
                ),
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        except SandboxBlockError as e:
            # 沙箱拦截
            tb_str = traceback.format_exc()
            return self._build_error_result(
                category="SANDBOX_BLOCK",
                summary="沙箱拦截：使用了不被允许的操作",
                reason=str(e),
                detail=(
                    "这不是 Python 语法错误，是沙箱策略拦截。\n"
                    "请换一种实现方式，不要使用被禁的功能。\n"
                    f"堆栈：\n{tb_str}"
                ),
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        except Exception as e:
            # 其他异常
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return self._build_error_result(
                category="LLM_CODE",
                summary="代码运行时抛出异常",
                reason=f"{type(e).__name__}: {e}",
                detail=f"堆栈：\n{tb_str}",
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        finally:
            # 无论如何，恢复原来的 trace
            sys.settrace(old_trace)

        # === 返回值校验 ===
        if not isinstance(ret, dict):
            return self._build_error_result(
                category="LLM_CODE",
                summary="返回值不是字典",
                reason=f'execute 必须 return {{"data": ...}}，实际返回 {type(ret).__name__}',
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )
        if "data" not in ret:
            return self._build_error_result(
                category="LLM_CODE",
                summary="返回值缺少 data 字段",
                reason='execute 必须 return {"data": ...}',
                raw_code=raw_code,
                sanitized_code=sanitized_code,
            )

        return ExecutionResult(
            ok=True,
            data=ret["data"],
            raw_llm_code=raw_code,
            sanitized_code=sanitized_code,
        )


# ============================================================
# 环境探测
# ============================================================

def _detect_environment(config: Dict[str, Any]) -> str:
    """快速版：只填系统信息，不探测外部命令和网络"""
    system = platform.system()

    if system == "Windows":
        sys_name = "Windows"
        path_sep = "\\"
    elif system == "Darwin":
        sys_name = "macOS"
        path_sep = "/"
    elif system == "Linux":
        sys_name = "Linux"
        path_sep = "/"
    else:
        sys_name = system
        path_sep = "/"

    return (
        f"## 当前运行环境\n"
        f"- 操作系统: {sys_name} {platform.release()}\n"
        f"- 架构: {platform.machine()}\n"
        f"- Python: {platform.python_version()}\n"
        f"- 路径分隔符: {path_sep}\n"
        f"\n"
        f"写代码时按上述环境处理路径、命令、编码等差异。"
    )


# ============================================================
# SystemPromptBuilder
# ============================================================

class SystemPromptBuilder:
    def __init__(self, config: Dict[str, Any], is_remote: bool = False) -> None:
        self._is_remote = is_remote
        self._config = config

    def build_base(self) -> str:
        if self._is_remote:
            return AICP_SYSTEM_PROMPT.replace("{remote_section}", "")
        return AICP_SYSTEM_PROMPT


# ============================================================
# AICP_LLM — 主入口
# ============================================================

class AICP_LLM:
    _cached_remote_section: Optional[str] = None
    _cached_remote_ts: float = 0.0

    def __init__(
        self,
        config: Dict[str, Any],
        is_remote: bool = False,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_system_prompt: bool = True,
        llm: Optional[LLM] = None,
        skip_remote_caps: bool = False,
    ) -> None:
        self._llm = llm if llm is not None else LLM(config)
        self._agent: Optional[Agent] = None
        self._is_remote = is_remote
        self._max_iterations = max_iterations
        self._config = config
        self._parser = ProtocolParser()
        self._executor: Optional[NoMemoryCodeExecutor] = None

        if enable_system_prompt:
            prompt_builder = SystemPromptBuilder(config, is_remote)
            base_prompt = prompt_builder.build_base()
            env_info = _detect_environment(config)
            self._system_prompt = f"{base_prompt}\n\n{env_info}"
            self._remote_caps_ready = self._is_remote or skip_remote_caps
            self._remote_caps_task = None
        else:
            self._system_prompt = ""
            self._remote_caps_ready = True
            self._remote_caps_task = None

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(llm=self._llm, chatEnvelop=self.chatEnvelop)
            self._executor = NoMemoryCodeExecutor(self._agent)
        return self._agent

    @property
    def executor(self) -> NoMemoryCodeExecutor:
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
            exclude_studio = not self._is_remote
            remote_section = await _build_remote_section(self._config, exclude_studio=exclude_studio)
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

        # ★ 白名单错误计数（防止反复重试）
        whitelist_error_count = 0

        for iteration in range(max_iterations):
            # === 调 LLM ===
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
                    sys.stdout.write(f"\r✅ {count} tokens\n")
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

            # === tool_call 标签检测 ===
            tool_call_pattern = re.compile(r'<\w*:?\s*tool_call\s*>', re.IGNORECASE)
            if tool_call_pattern.search(raw) or 'tool_calls' in raw:
                full_messages.append({"role": "assistant", "content": raw})
                full_messages.append({
                    "role": "user",
                    "content": "不要输出 tool_call 标签！请直接写 Python 代码块（~~~python ... ~~~）或纯文本回复。"
                })
                continue

            # === Envelop 协议解析 ===
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

            # === 保存原始输出 ===
            raw_log = LOG_DIR / f"llm_raw_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.txt"
            raw_log.write_text(raw, encoding="utf-8")

            # === 提取代码块 ===
            code = extract_python_code_block(raw)
            if not code:
                # 纯文本回复
                return Envelop(receiver="user", payload={"ok": True, "data": raw})

            # === 保存代码 ===
            code_log = LOG_DIR / f"code_cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.py"
            code_log.write_text(code, encoding="utf-8")
            logger.info(f"✅ 代码已保存: {code_log}")

            # === 提取并安装依赖 ===
            packages = extract_packages(raw)
            if packages:
                try:
                    await ensure_packages(packages)
                except Exception as e:
                    logger.error("ensure_packages failed: %s", e)
                    full_messages.append({"role": "assistant", "content": raw})
                    full_messages.append({
                        "role": "user",
                        "content": f"依赖安装失败：{e}\n请换其他方案。",
                    })
                    continue

            # === 执行代码（带超时）===
            envelop = Envelop(payload={"state": {}})

            try:
                result = await asyncio.wait_for(
                    self.executor.run(code, envelop, self.agent),
                    timeout=DEFAULT_TIMEOUT_SECONDS + 10,
                )
            except asyncio.TimeoutError:
                result = ExecutionResult(
                    ok=False,
                    error=(
                        f"【错误分类】LLM_CODE\n"
                        f"【一句话】执行超时（{DEFAULT_TIMEOUT_SECONDS} 秒）\n"
                        f"【原因】代码可能包含死循环或长时间阻塞操作\n"
                        f"【可修复】yes\n"
                        f"【处理】请检查代码，避免死循环和长时间阻塞。"
                    ),
                    raw_llm_code=code,
                    sanitized_code=code,
                    error_category="LLM_CODE",
                )
            except Exception as exc:
                # 极端情况：run() 自身崩溃
                tb_str = traceback.format_exc()
                logger.error("Executor crashed (iteration %d):\n%s", iteration + 1, tb_str)
                result = ExecutionResult(
                    ok=False,
                    error=(
                        f"【错误分类】RUNTIME_INTERNAL\n"
                        f"【一句话】executor.run 自身崩溃\n"
                        f"【原因】{type(exc).__name__}: {exc}\n"
                        f"【可修复】no\n"
                        f"【处理】runtime 内部问题，不要尝试修改代码。\n"
                        f"【详情】\n{tb_str}"
                    ),
                    raw_llm_code=code,
                    sanitized_code=code,
                    error_category="RUNTIME_INTERNAL",
                )

            # === 执行成功 ===
            if result.ok:
                return Envelop(receiver="user", payload=result.to_dict())

            # === 写错误日志 ===
            error_log_dir = LOG_DIR / "llm_errors"
            error_log_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = error_log_dir / f"error_{timestamp}.txt"
            error_file.write_text(
                f"=== LLM 执行错误 ===\n"
                f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Session: {kwargs.get('session_id', 'unknown')}\n"
                f"Iteration: {iteration + 1}/{max_iterations}\n"
                f"错误分类: {result.error_category}\n"
                f"可修复: {'yes' if result.fixable_by_llm else 'no'}\n\n"
                f"--- 错误信息 ---\n{result.error}\n\n"
                f"--- LLM 原始输出 ---\n{raw}\n",
                encoding="utf-8",
            )
            logger.error(f"❌ 执行失败 [{result.error_category}]，日志: {error_file}")

            # === 判断：runtime 内部错误 → 直接终止 ===
            if not result.fixable_by_llm:
                logger.error("Runtime internal error, aborting retry loop.")
                return Envelop(
                    receiver="user",
                    payload={
                        "ok": False,
                        "error": result.error,
                        "error_category": result.error_category,
                        "aborted": True,
                    },
                )

            # ★=== 白名单/沙箱错误计数（防止反复重试）===★
            if result.error and (
                "白名单" in result.error
                or "沙箱" in result.error
                or "SANDBOX_BLOCK" in result.error
                or "执行超时" in result.error
                or "阻塞" in result.error
            ):
                whitelist_error_count += 1
                logger.warning(
                    f"沙箱/超时错误计数 {whitelist_error_count}/{MAX_WHITELIST_ERRORS}（iteration {iteration + 1}）"
                )
                if whitelist_error_count >= MAX_WHITELIST_ERRORS:
                    return Envelop(
                        receiver="user",
                        payload={
                            "ok": False,
                            "error": (
                                f"连续 {whitelist_error_count} 次沙箱/超时错误，终止重试。\n"
                                f"请换方案，或检查代码是否符合沙箱规则。\n\n"
                                f"最后一次错误：\n{result.error}"
                            ),
                            "error_category": result.error_category,
                            "aborted": True,
                        },
                    )

            # === LLM 能修 → 喂回继续迭代 ===
            full_messages.append({"role": "assistant", "content": raw})
            full_messages.append({"role": "user", "content": result.error})
            continue

        # === 达到最大迭代 ===
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
        return [{"role": "system", "content": self._system_prompt}, *messages]

    @staticmethod
    def _format_error_feedback(error: str, code: str, hint: str = "") -> str:
        code_snippet = code[:MAX_CODE_SNIPPET_LENGTH]
        if len(code) > MAX_CODE_SNIPPET_LENGTH:
            code_snippet += "\n... (代码已截断)"
        parts = [f"代码执行失败: {error}", "", "源码:", code_snippet]
        if hint:
            parts.append(f"\n{hint}")
        return "\n".join(parts)


# ============================================================
# ProtocolParser（兼容保留）
# ============================================================

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
        return extract_python_code_block(raw)

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


# ============================================================
# 远端能力构建（保持原样）
# ============================================================

async def _build_remote_section(config: Dict[str, Any], exclude_studio: bool = False) -> str:
    endpoints = _load_endpoints_config(config, exclude_studio=exclude_studio)
    if not endpoints:
        return ""
    aicp_list = await _probe_aicp_endpoints(endpoints)
    aicp_section = _build_aicp_section(aicp_list, endpoints) if aicp_list else ""
    api_section = _build_api_section(endpoints)
    parts = []
    if aicp_section:
        parts.append(aicp_section)
    if api_section:
        parts.append(api_section)
    if not parts:
        return ""
    header = "【远端能力节点 — 当本地能力不足时使用】\n\n"
    return header + "\n\n".join(parts)


async def _probe_aicp_endpoints(endpoints: List[Endpoint]) -> str:
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


def _format_aicp_capability(name: str, url: str, caps: dict) -> str:
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


def _build_api_section(endpoints: List[Endpoint]) -> str:
    api_eps = [ep for ep in endpoints if ep.type == "api"]
    if not api_eps:
        return ""
    lines = ["## 外部 API 端点（已配置，可直接调用）", ""]
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


def create_aicp_llm(config: Dict[str, Any], is_remote: bool = False, **kwargs: Any) -> AICP_LLM:
    return AICP_LLM(config, is_remote=is_remote, **kwargs)