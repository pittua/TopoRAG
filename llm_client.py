"""
LLMClient     — Anthropic / Gemini SDK をAPIキーで呼ぶラッパー
CLILLMClient  — claude / gemini CLI を subprocess で呼ぶラッパー（APIキー不要）
MockLLMClient — テスト用モック

使い方:
    # SDK（APIキー必要）
    client = LLMClient(provider="anthropic")
    client = LLMClient(provider="gemini")

    # CLI（ログイン済みなら APIキー不要）
    client = CLILLMClient(provider="claude")
    client = CLILLMClient(provider="gemini")

    result = client.chat(system="...", user="...")
"""

import os
import shutil
import subprocess


class LLMClient:
    SUPPORTED = ("anthropic", "gemini")

    def __init__(self, provider: str = "anthropic", model: str | None = None):
        if provider not in self.SUPPORTED:
            raise ValueError(f"provider は {self.SUPPORTED} のいずれかを指定してください")

        self.provider = provider

        # ── Anthropic ──────────────────────────────────────
        if provider == "anthropic":
            try:
                import anthropic as _anthropic
            except ImportError:
                raise ImportError("pip install anthropic  を実行してください")

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "\n[Anthropic] APIキーが見つかりません。\n"
                    "ターミナルで以下を実行してから再度お試しください:\n"
                    "  export ANTHROPIC_API_KEY='sk-ant-XXXXXX'\n"
                    "APIキーの取得: https://console.anthropic.com/"
                )
            self._client = _anthropic.Anthropic(api_key=api_key)
            self.model   = model or "claude-sonnet-4-6"

        # ── Gemini ─────────────────────────────────────────
        elif provider == "gemini":
            try:
                from google import genai as _genai
            except ImportError:
                raise ImportError("pip install google-genai  を実行してください")

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "\n[Gemini] APIキーが見つかりません。\n"
                    "ターミナルで以下を実行してから再度お試しください:\n"
                    "  export GEMINI_API_KEY='AIzaXXXXXX'\n"
                    "APIキーの取得: https://aistudio.google.com/app/apikey"
                )
            self._genai  = _genai
            self._client = _genai.Client(api_key=api_key)
            self.model   = model or "gemini-2.0-flash"

    # ── 共通呼び出し口 ─────────────────────────────────────

    def chat(self, system: str, user: str) -> str:
        """
        system と user を受け取り、LLMの返答テキストを返す。
        どちらのプロバイダでも同じ呼び方でOK。
        """
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        elif self.provider == "gemini":
            return self._call_gemini(system, user)

    # ── Anthropic 実装 ─────────────────────────────────────

    def _call_anthropic(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    # ── Gemini 実装 ────────────────────────────────────────

    def _call_gemini(self, system: str, user: str) -> str:
        # Gemini は system_instruction を config で渡す
        from google.genai import types as genai_types
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1000,
            ),
        )
        return response.text

    # ── デバッグ用 ─────────────────────────────────────────

    def __repr__(self):
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"


# ── CLI経由（APIキー不要）─────────────────────────────────

class CLILLMClient:
    """
    claude / gemini CLI を subprocess で呼ぶ。
    CLIにログイン済みであればAPIキー不要で動作する。

    provider="claude" → claude --print --system-prompt "..." "..."
    provider="gemini" → gemini --prompt "system\n\nuser"  (stderr破棄)
    """

    SUPPORTED = ("claude", "gemini")

    def __init__(self, provider: str = "claude", timeout: int = 120):
        if provider not in self.SUPPORTED:
            raise ValueError(f"provider は {self.SUPPORTED} のいずれかを指定してください")
        self.provider = provider
        self.model    = f"{provider}-cli"
        self.timeout  = timeout
        # Windows では claude.cmd / gemini.cmd を shutil.which で解決する
        self._exe = self._resolve(provider)

    @staticmethod
    def _resolve(name: str) -> str:
        cmd = shutil.which(name)
        if cmd is None:
            raise FileNotFoundError(
                f"'{name}' CLI が PATH に見つかりません。\n"
                f"  Claude: npm install -g @anthropic-ai/claude-code\n"
                f"  Gemini: npm install -g @google/gemini-cli"
            )
        return cmd

    def chat(self, system: str, user: str) -> str:
        if self.provider == "claude":
            return self._call_claude(system, user)
        else:
            return self._call_gemini(system, user)

    @staticmethod
    def _decode(b: bytes) -> str:
        """UTF-8 → cp932 の順でフォールバックデコード"""
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("cp932", errors="replace")

    def _run(self, args: list[str], stdin_data: bytes | None = None) -> str:
        result = subprocess.run(
            args,
            input=stdin_data,
            capture_output=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            err = self._decode(result.stderr) if result.stderr else "(no stderr)"
            raise RuntimeError(f"{self.provider} CLI error:\n{err.strip()}")
        return self._decode(result.stdout).strip()

    def _call_claude(self, system: str, user: str) -> str:
        # 長いプロンプトやマルチバイト文字をCMD引数で渡すと壊れるため stdin 経由にする
        return self._run(
            [self._exe, "--print", "--system-prompt", system],
            stdin_data=user.encode("utf-8"),
        )

    def _call_gemini(self, system: str, user: str) -> str:
        # gemini に --system-prompt オプションがないため先頭に結合して stdin で渡す
        combined = f"{system}\n\n{user}"
        return self._run(
            [self._exe, "--prompt", "-"],
            stdin_data=combined.encode("utf-8"),
        )

    def __repr__(self):
        return f"CLILLMClient(provider={self.provider!r})"


# ── APIキーなしで動作確認するモック ───────────────────────

class MockLLMClient:
    """APIキーなしでテスト用に使うモック（回路判定の形式を模倣）"""

    def __init__(self, provider="mock"):
        self.provider = provider
        self.model    = "mock"

    def chat(self, system: str, user: str) -> str:
        # プロンプトから回路名を抽出して擬似判定
        if "ハイパス" in user and "1位" in user and "RCハイパス" in user:
            name = "RCハイパスフィルタ"
        elif "ローパス" in user and "1位" in user and "RCローパス" in user:
            name = "RCローパスフィルタ"
        else:
            name = "不明な回路"
        return (
            f"【判定】{name}\n"
            f"【根拠】RAG検索で類似度1.00の{name}が1位に検索されました。"
            f"先頭直列部品・GND並列部品・ループ数がすべて一致しています。\n"
            f"【類似度の解釈】スコア1.00は完全一致を意味し、"
            f"2位との差（約0.32）も十分大きく、判定の信頼度は高いです。\n"
            f"（※ これはモック応答です。実際のLLMではありません）"
        )

    def __repr__(self):
        return "MockLLMClient()"
