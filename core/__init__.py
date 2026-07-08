
"""AICP Core — Envelop + Agent + plugins dict + route"""
import asyncio
import uuid
from typing import Dict, Optional, Callable
from datetime import datetime


class Envelop:
    __slots__ = (
        "sender", "receiver", "intent", "payload",
        "trace_id", "message_id", "channel_id", "ttl", "meta",
        "created_at", "path_history"
    )
    
    def __init__(self, sender="", receiver="", intent="", payload=None,
                 channel_id="", ttl=10, meta=None):
        self.sender = sender
        self.receiver = receiver
        self.intent = intent
        self.payload = payload if payload is not None else {}
        self.trace_id = f"tr_{uuid.uuid4().hex[:8]}"
        self.message_id = f"msg_{uuid.uuid4().hex[:6]}"
        self.channel_id = channel_id
        self.ttl = ttl
        self.meta = meta if meta is not None else {}
        self.created_at = datetime.now().isoformat()
        self.path_history = []

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "intent": self.intent,
            "payload": self.payload,
            "trace_id": self.trace_id,
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "ttl": self.ttl,
            "meta": self.meta,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Envelop":
        env = cls(
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            intent=data.get("intent", ""),
            payload=data.get("payload", {}),
            channel_id=data.get("channel_id", ""),
            ttl=data.get("ttl", 10),
            meta=data.get("meta", {}),
        )
        env.trace_id = data.get("trace_id", env.trace_id)
        env.message_id = data.get("message_id", env.message_id)
        env.created_at = data.get("created_at", env.created_at)
        return env


class Agent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


# 全局插件地址簿
plugins: Dict[str, Callable] = {}


async def route(envelop: Envelop, agent: Agent = None, timeout: float = 300.0) -> Optional[Envelop]:
    """引擎路由 — 根据 envelop.receiver 找到插件并执行"""
    if not envelop.receiver:
        envelop.payload = {"error": "Missing receiver"}
        return envelop
    
    if envelop.ttl <= 0:
        envelop.payload = {"error": "TTL expired"}
        return envelop
    
    envelop.ttl -= 1
    
    plugin = plugins.get(envelop.receiver)
    if not plugin:
        envelop.payload = {"error": f"Plugin not found: {envelop.receiver}"}
        return envelop
    
    if agent is None:
        agent = Agent()
    
    try:
        result = await asyncio.wait_for(plugin(envelop, agent), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        envelop.payload = {"error": f"Plugin timeout after {timeout}s"}
        return envelop
    except Exception as e:
        envelop.payload = {"error": f"Plugin execution error: {str(e)[:200]}"}
        return envelop


# 兼容旧代码
engine_route = route
