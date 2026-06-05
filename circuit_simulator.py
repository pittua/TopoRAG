"""
circuit_simulator.py — ngspice / PySpice による回路シミュレーション解析

設計: docs/SIMULATION_DESIGN.md

役割:
  ネットリスト（部品値あり）を PySpice 経由で ngspice に渡し、
  AC 解析結果をフィルタ特性などの「整理済み特徴量」として返す。
  結果は RAG ベクトルには入れず、後段で LLM プロンプトに渡す。

設計上のポイント:
  - 判定しきい値はマジックナンバーをコードに散らさず定数として集約する。
  - DC/高周波利得は固定周波数ではなく解析レンジ端から内側の点で評価する
    （端点アーティファクトと fc の極端な回路での誤判定を避ける）。
  - ダイオード回路は大信号挙動が本質のため AC 解析せず、過渡解析（未実装）送り。
  - ngspice / PySpice が無い環境では例外を投げず skipped_* にフォールバックする。
  - 周波数応答 → 特徴量 の解釈ロジック (_classify_response) は ngspice 非依存の
    純関数として分離し、単体テスト可能にする。
"""

from __future__ import annotations

import os
import re
import glob
import math
import shutil

import numpy as np


# ─────────────────────────────────────────────────────────
# 判定しきい値（定数として集約。採用部品値での実測で調整する）
# ─────────────────────────────────────────────────────────

RATIO_PASS_STOP = 2.0    # 通過 / 阻止を分ける利得比
RATIO_RESONANCE = 1.5    # 共振とみなすピーク比
GAIN_FLOOR      = 0.05   # 有意な利得とみなす下限（線形値）
EDGE_MARGIN_DEC = 0.5    # 端点アーティファクト除外（解析レンジ端からの余裕・デケード）

# AC 解析レンジ（採用部品値の fc: RC=1.6kHz, LC=50kHz を内包する）
AC_F_START = 1.0         # Hz
AC_F_STOP  = 10e6        # Hz
AC_PTS_DEC = 100         # points / decade

_SQRT2 = math.sqrt(2.0)

# 能動素子（3端子以上）。線形 AC 解析の対象外（DCバイアスが必要）
_ACTIVE_TYPES = {"NPN", "PNP", "NMOS", "PMOS", "OPAMP"}


# ─────────────────────────────────────────────────────────
# SI 接頭辞パーサ
# ─────────────────────────────────────────────────────────

_SI_PREFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9,
    "u": 1e-6,  "μ": 1e-6,  "m": 1e-3,
    "k": 1e3,   "K": 1e3,   "M": 1e6, "G": 1e9,
}

# 数値 + 任意の SI 接頭辞 + 任意の単位文字（F / H / Ω / ohm 等）
_VALUE_RE = re.compile(
    r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*([fpnuμmkKMG]?)\s*([a-zA-ZΩ]*)\s*$"
)


def parse_value(v) -> float | None:
    """
    ネットリストの文字列値を float に変換する。

    >>> parse_value("1k")     # 1000.0
    >>> parse_value("100n")   # 1e-7
    >>> parse_value("10uF")   # 1e-5
    >>> parse_value("4.7m")   # 0.0047
    >>> parse_value(None)     # None

    対応接頭辞: f / p / n / u / μ / m / k / K / M / G
    （m=ミリ, M=メガ。docs/SIMULATION_DESIGN.md に準拠）
    """
    if v is None:
        return None
    if isinstance(v, bool):           # True/False を 1/0 と誤変換させない
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if not s:
        return None

    # ngspice 流の "名前=値" 形式（KiCad の Value 例: "c=100u", "r=5"）に対応
    if "=" in s:
        s = s.rsplit("=", 1)[-1].strip()

    m = _VALUE_RE.match(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None

    num = float(m.group(1))
    prefix = m.group(2)
    return num * _SI_PREFIX.get(prefix, 1.0)


# ─────────────────────────────────────────────────────────
# ngspice パス解決（環境非依存・グレースフルフォールバック）
# ─────────────────────────────────────────────────────────

# ngspice の実体は環境により実行ファイルか共有ライブラリのいずれか。
#   - 単体インストール: ngspice / ngspice.exe（実行ファイル）
#   - KiCad 同梱: ngspice.dll / libngspice.so / libngspice.dylib（共有ライブラリ）
# PySpice は共有ライブラリ（NgSpiceShared）でも動作するため、両方を検出対象にする。
_NGSPICE_FILENAMES = (
    "ngspice.exe", "ngspice",
    "ngspice.dll", "libngspice.dll",
    "libngspice.so", "libngspice.dylib",
)


def resolve_ngspice_path() -> str | None:
    """
    ngspice（実行ファイル or 共有ライブラリ）の場所を以下の優先順で探索する。
      1. 環境変数 NGSPICE_PATH（最優先。ファイルパス or それを含むディレクトリ）
      2. PATH 上の ngspice 実行ファイル
      3. KiCad 同梱を含む既知パス候補（バージョン非依存にグロブ探索）
    見つからない場合は None を返す（呼び出し側で skipped_no_ngspice にする）。
    開発環境固有の絶対パスはハードコードしない。
    """
    env = os.environ.get("NGSPICE_PATH")
    if env:
        if os.path.isfile(env):
            return env
        if os.path.isdir(env):
            for fn in _NGSPICE_FILENAMES:
                cand = os.path.join(env, fn)
                if os.path.isfile(cand):
                    return cand

    found = shutil.which("ngspice") or shutil.which("ngspice.exe")
    if found:
        return found

    candidates_base = [
        r"C:\Program Files\KiCad",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\KiCad"),
        "/usr/bin",
        "/usr/local/bin",
        "/usr/lib",
        "/Applications/KiCad/KiCad.app/Contents/Frameworks",
    ]
    patterns = []
    for base in candidates_base:
        for fn in _NGSPICE_FILENAMES:
            patterns.append(os.path.join(base, "*", "bin", fn))  # KiCad/<ver>/bin/
            patterns.append(os.path.join(base, "bin", fn))
            patterns.append(os.path.join(base, fn))
    for pat in patterns:
        for hit in sorted(glob.glob(pat)):
            return hit
    return None


# ─────────────────────────────────────────────────────────
# 周波数応答 → 特徴量（ngspice 非依存の純関数。単体テスト可能）
# ─────────────────────────────────────────────────────────

def _to_db(x: float) -> float | None:
    if x is None or x <= 0.0:
        return None
    return 20.0 * math.log10(x)


def _classify_response(freqs, gains) -> dict:
    """
    AC 解析結果（周波数配列・利得配列）からフィルタ特性を判定する。

    引数:
      freqs : 昇順の周波数配列（Hz）
      gains : 各周波数の |H| 線形値
    返り値:
      simulation_type を除く特徴量 dict（confidence / warnings 含む）
    """
    freqs = np.asarray(freqs, dtype=float)
    gains = np.asarray(np.abs(gains), dtype=float)

    warnings: list[str] = []

    # 端点アーティファクト除外: レンジ端から EDGE_MARGIN_DEC デケード内側で評価
    f_lo = freqs[0]  * (10.0 ** EDGE_MARGIN_DEC)
    f_hi = freqs[-1] / (10.0 ** EDGE_MARGIN_DEC)
    lo_idx = min(int(np.searchsorted(freqs, f_lo)), len(freqs) - 1)
    hi_idx = max(min(int(np.searchsorted(freqs, f_hi)), len(freqs) - 1), 0)

    dc = float(gains[lo_idx])   # 低域利得（DC 利得の代理）
    hf = float(gains[hi_idx])   # 高域利得
    pk = float(gains.max())     # ピーク利得
    mn = float(gains.min())     # 最小利得
    pk_idx = int(gains.argmax())
    peak_freq = float(freqs[pk_idx])

    peak_interior = lo_idx < pk_idx < hi_idx

    is_lowpass  = dc > hf * RATIO_PASS_STOP and dc > GAIN_FLOOR
    is_highpass = hf > dc * RATIO_PASS_STOP and hf > GAIN_FLOOR
    is_bandpass = pk > max(dc, hf) * RATIO_PASS_STOP and peak_interior
    is_bandstop = (mn < min(dc, hf) / RATIO_PASS_STOP
                   and dc > GAIN_FLOOR and hf > GAIN_FLOOR)
    has_resonance = pk > max(dc, hf) * RATIO_RESONANCE

    # カットオフ周波数（-3 dB 点）
    cutoff = None
    if is_lowpass:
        below = np.where(gains < dc / _SQRT2)[0]
        if below.size:
            cutoff = float(freqs[int(below[0])])
    elif is_highpass:
        above = np.where(gains > hf / _SQRT2)[0]
        if above.size:
            cutoff = float(freqs[int(above[0])])
    elif is_bandpass or is_bandstop:
        cutoff = peak_freq

    # 信頼度の決定: 主たる利得比がしきい値からどれだけ離れているか
    classified = is_lowpass or is_highpass or is_bandpass or is_bandstop
    if not classified:
        confidence = "low"
        warnings.append("明確なフィルタ特性を判定できなかった")
    else:
        hi_g, lo_g = max(dc, hf), max(min(dc, hf), 1e-12)
        ratio = hi_g / lo_g
        if ratio >= RATIO_PASS_STOP * 2.0:
            confidence = "high"
        elif ratio >= RATIO_PASS_STOP:
            confidence = "medium"
            warnings.append("利得比がしきい値境界に近い")
        else:
            confidence = "low"
            warnings.append("利得比がしきい値を十分に超えていない")

    return {
        "dc_gain_db":     round(_to_db(dc), 2) if _to_db(dc) is not None else None,
        "hf_gain_db":     round(_to_db(hf), 2) if _to_db(hf) is not None else None,
        "peak_gain_db":   round(_to_db(pk), 2) if _to_db(pk) is not None else None,
        "peak_freq_hz":   round(peak_freq, 2),
        "is_lowpass":     bool(is_lowpass),
        "is_highpass":    bool(is_highpass),
        "is_bandpass":    bool(is_bandpass),
        "is_bandstop":    bool(is_bandstop),
        "has_resonance":  bool(has_resonance),
        "cutoff_freq_hz": round(cutoff, 2) if cutoff is not None else None,
        "confidence":     confidence,
        "warnings":       warnings,
    }


# ─────────────────────────────────────────────────────────
# スキップ結果のテンプレート
# ─────────────────────────────────────────────────────────

_NONE_FIELDS = (
    "dc_gain_db", "hf_gain_db", "peak_gain_db", "peak_freq_hz",
    "is_lowpass", "is_highpass", "is_bandpass", "is_bandstop",
    "has_resonance", "cutoff_freq_hz",
)


def _skip_result(simulation_type: str, reason: str) -> dict:
    result = {"simulation_type": simulation_type}
    for k in _NONE_FIELDS:
        result[k] = None
    result["confidence"] = None
    result["warnings"] = [reason]
    return result


# ─────────────────────────────────────────────────────────
# CircuitSimulator
# ─────────────────────────────────────────────────────────

class CircuitSimulator:
    """
    TopoRAG 回路 dict を受け取り、シミュレーション特徴量を返す。
    """

    def __init__(self, circuit: dict):
        self.circuit = circuit

    def extract_simulation_features(self) -> dict:
        """
        回路構成に応じて解析種別を決定し、シミュレーション特徴量を返す。
        スキップ／失敗時も例外は投げず skipped_* を返す。
        """
        comps = self.circuit.get("components", [])
        types = {c["type"] for c in comps}

        # 能動素子（BJT/MOSFET/OpAmp）→ DCバイアス＋小信号解析が必要（未実装）
        if types & _ACTIVE_TYPES:
            return _skip_result("skipped_active",
                                "能動素子を含むためスキップ（将来: DCバイアス＋小信号AC解析）")

        # SW を含む → 過渡解析が必要（未実装）のためスキップ
        if "SW" in types:
            return _skip_result("skipped_switch",
                                "スイッチング回路のためスキップ（将来: 過渡解析）")

        # 部品値の解決と欠落チェック（R/L/C は値必須）
        values: dict[str, float | None] = {}
        for c in comps:
            val = parse_value(c.get("value"))
            values[c["id"]] = val
            if val is None and c["type"] in ("R", "C", "L"):
                return _skip_result("skipped_missing_values",
                                    f"部品 {c['id']} の値が欠落しているためスキップ")

        # ダイオードを含む → 大信号挙動が本質。AC 解析は用いず過渡解析（未実装）送り
        if types & {"D", "DZ"}:
            return _skip_result("skipped_nonlinear",
                                "ダイオードを含むため過渡解析が必要（未実装）")

        # ngspice の所在確認
        if resolve_ngspice_path() is None:
            return _skip_result("skipped_no_ngspice",
                                "ngspice が見つからない（NGSPICE_PATH 未設定 / 未インストール）")

        # AC 解析実行
        try:
            freqs, gains = self._run_ac(values)
        except ImportError:
            return _skip_result("skipped_no_ngspice",
                                "PySpice が未インストール（pip install PySpice）")
        except Exception as e:  # noqa: BLE001  解析失敗は全体を止めない
            return _skip_result("skipped_error", f"シミュレーション失敗: {e}")

        feat = _classify_response(freqs, gains)
        feat["simulation_type"] = "ac_passive"
        return feat

    def _run_ac(self, values: dict[str, float | None]):
        """
        PySpice で AC 解析を実行し、(freqs, gains) を返す。
        PySpice 未インストール時は ImportError を送出（呼び出し側で捕捉）。
        """
        # KiCad 同梱の ngspice 共有ライブラリ（ngspice.dll 等）を PySpice の
        # NgSpiceShared が読み込めるよう環境を整える。
        ng = resolve_ngspice_path()
        is_shared = ng and os.path.splitext(ng)[1].lower() in (".dll", ".so", ".dylib")
        if is_shared:
            ng_dir = os.path.dirname(ng)
            # PySpice に共有ライブラリの場所を明示
            os.environ.setdefault("NGSPICE_LIBRARY_PATH", ng)
            # SPICE_LIB_DIR 未設定だと PySpice が NGSPICE_PATH(None) を参照して落ちるため、
            # ngspice のコードモデル/スクリプト相当ディレクトリを与えておく。
            if "SPICE_LIB_DIR" not in os.environ:
                root = os.path.dirname(ng_dir)            # <root>/bin → <root>
                lib_ng = os.path.join(root, "lib", "ngspice")
                os.environ["SPICE_LIB_DIR"] = lib_ng if os.path.isdir(lib_ng) else ng_dir
            # 依存 DLL を解決できるよう所在ディレクトリを検索パスへ追加
            os.environ["PATH"] = ng_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(ng_dir)
                except OSError:
                    pass

        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_V

        if is_shared:
            # 共有ライブラリの所在を環境変数から確定させる（NGSPICE_LIBRARY_PATH を反映）
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            NgSpiceShared.setup_platform()

        c = self.circuit
        gnd = c["ports"]["gnd"]
        inp = c["ports"]["input"]
        out = c["ports"]["output"]

        spice = Circuit("toporag_ac")

        def node(name):
            # GND ポートは PySpice の基準ノード(0)に対応づける
            return spice.gnd if name == gnd else name

        # 入力に AC 1V 源を接続（DC 0 / AC 1）
        spice.SinusoidalVoltageSource("src", node(inp), spice.gnd,
                                      amplitude=1 @ u_V)

        for comp in c["components"]:
            t, cid = comp["type"], comp["id"]
            terms = list(comp["terminals"].values())
            n1, n2 = node(terms[0]), node(terms[1])
            val = values[cid]
            if t == "R":
                spice.R(cid, n1, n2, val)
            elif t == "C":
                spice.C(cid, n1, n2, val)
            elif t == "L":
                spice.L(cid, n1, n2, val)

        sim = spice.simulator()
        analysis = sim.ac(start_frequency=AC_F_START,
                          stop_frequency=AC_F_STOP,
                          number_of_points=AC_PTS_DEC,
                          variation="dec")

        freqs = np.array(analysis.frequency)
        vout = np.array(analysis[str(out)])
        return freqs, np.abs(vout)


# ─────────────────────────────────────────────────────────
# 動作確認（ngspice 非依存の判定ロジックをテスト）
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("--- parse_value ---")
    for s in ["1k", "100n", "10uF", "4.7m", "1000", "2.2kΩ", None, "", "abc"]:
        print(f"  parse_value({s!r:>8}) = {parse_value(s)}")

    print("\n--- resolve_ngspice_path ---")
    print(f"  {resolve_ngspice_path()}")

    print("\n--- _classify_response（合成 RC ローパス応答）---")
    # 1次 RC ローパス: fc=1592Hz の理想応答を合成
    fc = 1592.0
    f = np.logspace(0, 7, 700)            # 1Hz〜10MHz
    h = 1.0 / np.sqrt(1.0 + (f / fc) ** 2)
    feat = _classify_response(f, h)
    feat["simulation_type"] = "ac_passive"
    for k, v in feat.items():
        print(f"  {k:16}: {v}")
