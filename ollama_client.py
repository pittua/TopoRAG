"""
OllamaClient — ローカル Ollama (http://localhost:11434) を叩く最小クライアント。
llm_client の各クラスと同じ .chat(system, user) -> str インターフェースを持つ。
弱モデル比較（ir_repr_eval / ir_repr_eval_scale）で使う。依存は標準ライブラリのみ。
"""
from __future__ import annotations

import json
import urllib.request


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434", timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]

    def __repr__(self):
        return f"OllamaClient(model={self.model!r})"
