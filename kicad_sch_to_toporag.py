"""
KiCad .kicad_sch → TopoRAG query_netlists.json コンバータ

動作フロー:
  1. kicad-cli で .kicad_sch から kicadsexpr ネットリストを生成
  2. S式をパースして nets / components / libparts を抽出
  3. libparts のスキーマピン番号 → 役割名 (K/A, G/D/S 等) を解決
  4. nets セクションから (部品ref, ピン番号) → ネット名 マッピングを構築
  5. 部品種別・terminals・ports を解決して TopoRAG 形式に変換

対応部品:
  R / C / L / SW / D / DZ / BJT(NPN/PNP) / MOSFET(NMOS/PMOS) / OpAmp

スキップ:
  電圧源・電流源・論理ゲート 等
"""

import sys
import os
import re
import glob
import json
import tempfile
import subprocess
import shutil
from pathlib import Path

from circuit_simulator import parse_value


# ─── kicad-cli の場所 ───────────────────────────────────────

# KiCad 標準インストール先（バージョン非依存にグロブ探索する。環境固有パスは埋め込まない）
_KICAD_CLI_BASES = [
    r"C:\Program Files\KiCad",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\KiCad"),
    "/usr/bin",
    "/usr/local/bin",
    "/Applications/KiCad/KiCad.app/Contents/MacOS",
]


def _find_kicad_cli() -> str:
    """
    kicad-cli の場所を以下の順で解決する。
      1. 環境変数 KICAD_CLI_PATH
      2. PATH 上の kicad-cli
      3. KiCad 標準インストール先のグロブ探索
    """
    env = os.environ.get("KICAD_CLI_PATH")
    if env and os.path.exists(env):
        return env

    cmd = shutil.which("kicad-cli") or shutil.which("kicad-cli.exe")
    if cmd:
        return cmd

    for base in _KICAD_CLI_BASES:
        for pat in (os.path.join(base, "*", "bin", "kicad-cli.exe"),
                    os.path.join(base, "kicad-cli.exe"),
                    os.path.join(base, "kicad-cli")):
            for hit in sorted(glob.glob(pat)):
                return hit

    raise FileNotFoundError(
        "kicad-cli が見つかりません。KiCad がインストールされているか、"
        "環境変数 KICAD_CLI_PATH を確認してください。"
    )


# ─── 部品種別マッピング ────────────────────────────────────

# Sim.Device プロパティ値 → TopoRAG 種別
SIM_DEVICE_MAP: dict[str, str | None] = {
    "R": "R", "C": "C", "L": "L",
    "SW": "SW",                     # スイッチ（スイッチング変換器の開閉素子）
    "D": "D", "DZ": "DZ",
    "NPN": "NPN", "PNP": "PNP",     # BJT
    "NMOS": "NMOS", "PMOS": "PMOS", # MOSFET
    "VDMOS": "NMOS",                # パワーMOSFET（極性不明時は N 既定。pchan は未判定）
    "OPAMP": "OPAMP",
    "V": None, "I": None,          # 電源・電流源スキップ
    # OpAmp は Sim.Device が "SUBCKT" のことが多い → part 名キーワードで解決する
}

# part 名キーワード → TopoRAG 種別（Sim.Device がない / SUBCKT の場合）
# 極性が名前に現れる NPN/PNP/NMOS/PMOS を汎用 BJT/MOSFET より先に判定する。
PART_KEYWORD_MAP = [
    ("ZENER",    "DZ"),
    ("SCHOTTKY", "D"),
    ("DIODE",    "D"),
    ("SWITCH",   "SW"),
    ("INDUCTOR", "L"),
    ("CAPACITOR","C"),
    ("RESISTOR", "R"),
    ("OPAMP",    "OPAMP"),
    ("NPN",      "NPN"),
    ("PNP",      "PNP"),
    ("NMOS",     "NMOS"),
    ("PMOS",     "PMOS"),
    ("MOSFET",   None),   # 極性不明 → Sim.Device に委ねる
    ("JFET",     None),
    ("BJT",      None),   # 極性不明
]

# ref 頭文字 → TopoRAG 種別（最終フォールバック）
# Q/M は極性（NPN/PNP・NMOS/PMOS）を頭文字から決められないため None（Sim.Device 依存）。
REF_PREFIX_MAP = {
    "SW": "SW",                          # スイッチ（KiCad の標準リファレンス SW*）
    "R": "R", "C": "C", "L": "L",
    "D": "D", "Z": "DZ",
    "V": None, "I": None,
    "Q": None, "M": None,
    "U": None, "IC": None,
    "X": None, "J": None, "P": None, "T": None,
}

# ─── ネット名パターン ───────────────────────────────────────

GND_RE = re.compile(r"^(GND|/GND|\bGND\b|VSS|AGND|DGND|0)$", re.IGNORECASE)
# 信号入力（電源レールは含めない。VCC/VDD/V+ は POWER_RAIL_RE 側で除外する）
IN_RE  = re.compile(r"(^|/)(V?IN|VIN|VSIG|VSIGNAL|SIG|SIGNAL|INPUT|SRC|SOURCE|AC_IN)$", re.IGNORECASE)
OUT_RE = re.compile(r"(^|/)(V?OUT|VOUT|OUTPUT|RECT_OUT|DC_POS|LOAD)$", re.IGNORECASE)
# 電源レール（DCバイアス供給）。能動回路の入出力ポート候補から除外する。
POWER_RAIL_RE = re.compile(r"(^|/)(VDC|VCC|VDD|VEE|VSS|VBAT|VSUP|VSUPPLY|V\+|V-)$", re.IGNORECASE)


# ─── S式パーサ ─────────────────────────────────────────────

def _parse_sexp(text: str) -> list:
    tokens = re.findall(r'"[^"]*"|\(|\)|[^\s()]+', text)
    pos = [0]

    def atom() -> str:
        t = tokens[pos[0]]; pos[0] += 1
        return t[1:-1] if t.startswith('"') else t

    def expr() -> list:
        pos[0] += 1          # consume '('
        items: list = []
        while tokens[pos[0]] != ")":
            items.append(expr() if tokens[pos[0]] == "(" else atom())
        pos[0] += 1          # consume ')'
        return items

    return expr()


def _children(node: list, tag: str) -> list[list]:
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == tag]


def _child(node: list, tag: str) -> list | None:
    r = _children(node, tag)
    return r[0] if r else None


def _text(node: list, tag: str, default: str = "") -> str:
    c = _child(node, tag)
    if c and len(c) >= 2 and isinstance(c[1], str):
        return c[1]
    return default


# ─── ネットリスト解析 ──────────────────────────────────────

def _parse_netlist(net_path: str) -> dict:
    with open(net_path, encoding="utf-8") as f:
        root = _parse_sexp(f.read())

    # libparts: {(lib, part): {schematic_pin_num: pin_name}}
    libpart_pins: dict[tuple, dict[str, str]] = {}
    lp_root = _child(root, "libparts")
    if lp_root:
        for lp in _children(lp_root, "libpart"):
            lib  = _text(lp, "lib")
            part = _text(lp, "part")
            pnode = _child(lp, "pins")
            pmap: dict[str, str] = {}
            if pnode:
                for p in _children(pnode, "pin"):
                    pmap[_text(p, "num")] = _text(p, "name")
            libpart_pins[(lib, part)] = pmap

    # nets: (ref, pin_num) → net_name
    pin_to_net: dict[tuple[str, str], str] = {}
    nets_root = _child(root, "nets")
    if nets_root:
        for net in _children(nets_root, "net"):
            name = _text(net, "name")
            if not name:
                continue
            for node in _children(net, "node"):
                ref = _text(node, "ref")
                pin = _text(node, "pin")
                pin_to_net[(ref, pin)] = name

    # components
    components: list[dict] = []
    comps_root = _child(root, "components")
    if comps_root:
        for comp in _children(comps_root, "comp"):
            ref  = _text(comp, "ref")
            lsrc = _child(comp, "libsource")
            lib  = _text(lsrc, "lib")  if lsrc else ""
            part = _text(lsrc, "part") if lsrc else ""

            props: dict[str, str] = {}
            for p in _children(comp, "property"):
                pname  = _text(p, "name")
                pvalue = _text(p, "value")
                if pname:
                    props[pname] = pvalue

            sim_device = props.get("Sim.Device", "").upper().strip()
            sim_pins   = props.get("Sim.Pins",   "").strip()

            # ピン役割マップ: libparts を優先（スキーマピン番号が一致するため）
            pin_role: dict[str, str] = {}
            lp_pmap = libpart_pins.get((lib, part), {})
            for pnum, pname in lp_pmap.items():
                if pname:
                    pin_role[pnum] = pname.upper()
            # libparts に名前がない場合は Sim.Pins で補完
            if not any(pin_role.values()):
                for m in re.finditer(r"(\d+)=(\S+)", sim_pins):
                    pin_role.setdefault(m.group(1), m.group(2).upper())

            # 部品値は comp 直下の (value "...") に入る（例: "100n" / "1k" / "1N4148"）。
            # property "Value" は古い形式向けのフォールバック。
            raw_value = _text(comp, "value") or props.get("Value", "")

            components.append({
                "ref": ref, "lib": lib, "part": part,
                "sim_device": sim_device,
                "pin_role": pin_role,
                "value":     parse_value(raw_value),   # SI接頭辞を解釈した float / None
                "value_raw": raw_value,                 # デバッグ用の元文字列
            })

    return {"components": components, "pin_to_net": pin_to_net}


# ─── 部品種別の解決 ────────────────────────────────────────

def _resolve_type(comp: dict) -> str | None:
    sd = comp["sim_device"]
    if sd in SIM_DEVICE_MAP:
        return SIM_DEVICE_MAP[sd]

    part_upper = comp["part"].upper()
    for kw, val in PART_KEYWORD_MAP:
        if kw in part_upper:
            return val

    ref_upper = comp["ref"].upper()
    for prefix, val in REF_PREFIX_MAP.items():
        if ref_upper.startswith(prefix):
            return val

    return None


# ─── terminals の構築 ────────────────────────────────────────

def _terminals_by_pinnum(pin_nets: dict[str, str],
                         role_order: list[str]) -> dict | None:
    """ピン番号昇順に role_order を割り当てるフォールバック（役割名が無い素子用）。"""
    pnums = sorted(pin_nets, key=lambda x: (len(x), x))
    if len(pnums) < len(role_order):
        return None
    return {role: pin_nets[pnums[i]] for i, role in enumerate(role_order)}


def _build_terminals(ttype: str, pin_role: dict[str, str],
                     pin_nets: dict[str, str]) -> dict | None:
    """ピン役割とネット名から TopoRAG terminals を構築。"""

    if ttype in ("D", "DZ"):
        anode = cathode = None
        for pnum, net in pin_nets.items():
            role = pin_role.get(pnum, "")
            if role in ("A", "ANODE"):
                anode = net
            elif role in ("K", "CATHODE"):
                cathode = net
        if anode is None or cathode is None:
            # フォールバック: ピン番号昇順で pin1=K, pin2=A
            pnums = sorted(pin_nets)
            if len(pnums) >= 2:
                cathode, anode = pin_nets[pnums[0]], pin_nets[pnums[1]]
            else:
                return None
        return {"anode": anode, "cathode": cathode}

    if ttype in ("NPN", "PNP"):
        # BJT: 役割 C/B/E → collector/base/emitter
        role_map = {"C": "collector", "COLLECTOR": "collector",
                    "B": "base", "BASE": "base",
                    "E": "emitter", "EMITTER": "emitter"}
        terms: dict[str, str] = {}
        for pnum, net in pin_nets.items():
            r = role_map.get(pin_role.get(pnum, ""))
            if r:
                terms[r] = net
        if len(terms) == 3:
            return terms
        # フォールバック: KiCad 既定ピン番号 1=C 2=B 3=E
        return _terminals_by_pinnum(pin_nets, ["collector", "base", "emitter"])

    if ttype in ("NMOS", "PMOS"):
        # MOSFET: 役割 D/G/S → drain/gate/source（Bulk はスキップ）
        role_map = {"D": "drain", "DRAIN": "drain",
                    "G": "gate", "GATE": "gate",
                    "S": "source", "SOURCE": "source"}
        terms = {}
        for pnum, net in pin_nets.items():
            r = role_map.get(pin_role.get(pnum, ""))
            if r and r not in terms:
                terms[r] = net
        if len(terms) == 3:
            return terms
        # フォールバック: KiCad 既定ピン番号 1=D 2=G 3=S
        return _terminals_by_pinnum(pin_nets, ["drain", "gate", "source"])

    if ttype == "OPAMP":
        # 役割 + → in_p, - → in_n。電源ピン(V+/V-等)はスキップ。
        # 出力は +/-/電源以外のピン（KiCad では役割名なしのことが多い）。
        POWER_ROLES = {"V+", "V-", "VCC", "VEE", "VDD", "VSS",
                       "VS+", "VS-", "VCC+", "VCC-", "VDD+", "VSS-"}
        in_p = in_n = None
        leftover: list[tuple[str, str]] = []
        for pnum, net in pin_nets.items():
            role = pin_role.get(pnum, "")
            if role == "+":
                in_p = net
            elif role == "-":
                in_n = net
            elif role in POWER_ROLES:
                continue
            else:
                leftover.append((pnum, net))
        out = None
        if leftover:
            leftover.sort()
            out = leftover[0][1]
        if in_p and in_n and out:
            return {"in_p": in_p, "in_n": in_n, "out": out}
        return None

    # R / C / L / SW（2 端子素子）
    p = n = None
    for pnum, net in pin_nets.items():
        role = pin_role.get(pnum, "")
        if role in ("+", "1", "~"):
            p = net
        elif role in ("-", "2"):
            n = net
    if p is None or n is None:
        pnums = sorted(pin_nets)
        if len(pnums) >= 2:
            p, n = pin_nets[pnums[0]], pin_nets[pnums[1]]
        else:
            return None
    return {"p": p, "n": n}


# ─── ポート推定 ────────────────────────────────────────────

def _gnd_priority(net: str) -> tuple:
    """代表 GND を決定的に選ぶための優先度。GND/0 を VSS/AGND 等より優先する。"""
    s = net.lstrip("/").upper()
    if s in ("GND", "0"):
        return (0, net)
    if s in ("AGND", "DGND"):
        return (1, net)
    return (2, net)          # VSS など（負電源とも解釈されうるエイリアス）


def _infer_ports(all_nets: set[str],
                 pin_to_net: dict,
                 orig_comps: list[dict]) -> dict | None:

    # GND: GND/0/VSS/AGND/DGND 等の「接地系ネット」をすべて把握し、
    # 代表を決定的に選ぶ（集合の反復順に依存して VSS を GND より先に選ぶと、
    # 残った GND が出力ポートに誤って漏れる問題を防ぐ）。
    gnd_nets = {n for n in all_nets if GND_RE.match(n)}
    if not gnd_nets:
        return None
    gnd = sorted(gnd_nets, key=_gnd_priority)[0]

    # 電源レール（VCC/VDD/V+/V-/Vdc 等）。接地系と合わせて入出力候補から除外する。
    rails = {n for n in all_nets if POWER_RAIL_RE.search(n)}
    excluded = gnd_nets | rails        # 入出力ポートになり得ないネット集合

    # 電圧源の + 端子ネット（信号源候補）。接地系・電源レールは除く。
    vsrc_pos: list[str] = []
    for comp in orig_comps:
        if comp["sim_device"] == "V":
            ref = comp["ref"]
            role_to_pin = {v: k for k, v in comp["pin_role"].items()}
            pos_pin = role_to_pin.get("+") or role_to_pin.get("1") or "1"
            net = pin_to_net.get((ref, pos_pin))
            if net and net not in excluded:
                vsrc_pos.append(net)

    # 入力: 信号名にマッチ かつ 接地系/電源レールでないネットを優先
    inp = next((n for n in sorted(all_nets)
                if IN_RE.search(n) and n not in excluded), None)
    if not inp:
        inp = vsrc_pos[0] if vsrc_pos else None

    # 出力: 出力名にマッチ かつ 接地系/入力/電源レール以外
    out = next((n for n in sorted(all_nets)
                if OUT_RE.search(n) and n not in excluded and n != inp), None)

    # 出力が見つからない → 最多ノード接続の非 接地系/入力/電源レールのネット
    if not out:
        degree: dict[str, int] = {}
        for net in pin_to_net.values():
            if net not in excluded and net != inp:
                degree[net] = degree.get(net, 0) + 1
        if degree:
            # 次数が同点の場合に備え、名前順で決定的に選ぶ
            out = max(sorted(degree), key=lambda n: degree[n])

    if inp and out and gnd and inp != out:
        return {"input": inp, "output": out, "gnd": gnd}
    return None


# ─── 単一 .kicad_sch の変換 ──────────────────────────────

def convert_kicad_sch(sch_path: str,
                      kicad_cli: str | None = None) -> dict | None:
    """
    .kicad_sch を TopoRAG 回路 dict に変換する。
    変換できない場合は None を返す。
    """
    sch_path = os.path.abspath(sch_path)
    cli = kicad_cli or _find_kicad_cli()

    with tempfile.NamedTemporaryFile(suffix=".net", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        r = subprocess.run(
            [cli, "sch", "export", "netlist",
             "--format", "kicadsexpr", "-o", tmp_path, sch_path],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", errors="replace").strip())

        parsed = _parse_netlist(tmp_path)
    finally:
        os.unlink(tmp_path)

    orig_comps = parsed["components"]
    pin_to_net = parsed["pin_to_net"]

    toporag_comps: list[dict] = []
    skipped: list[str] = []

    for comp in orig_comps:
        ttype = _resolve_type(comp)
        if ttype is None:
            skipped.append(f"{comp['ref']}({comp['sim_device'] or comp['part']})")
            continue

        ref = comp["ref"]
        pin_nets = {pnum: net for (r, pnum), net in pin_to_net.items() if r == ref}
        terminals = _build_terminals(ttype, comp["pin_role"], pin_nets)
        if terminals is None:
            skipped.append(f"{comp['ref']}(端子解決失敗)")
            continue

        toporag_comps.append({
            "id":        ref,
            "type":      ttype,
            "value":     comp.get("value"),   # KiCad Value プロパティ由来（None 可）
            "terminals": terminals,
        })

    if not toporag_comps:
        return None

    all_nets = set(pin_to_net.values())
    ports = _infer_ports(all_nets, pin_to_net, orig_comps)
    if ports is None:
        return None

    # 推定した入出力ポートが、残した部品の接続ノード上に存在するか検証する。
    # （OpAmp 等をスキップした結果、ポートが能動素子側にしか繋がっていない場合は
    #   パッシブ部品グラフと不整合になるため、変換不能として扱う）
    comp_nodes: set[str] = set()
    for c in toporag_comps:
        comp_nodes.update(c["terminals"].values())
    if ports["input"] not in comp_nodes or ports["output"] not in comp_nodes:
        return None

    stem = Path(sch_path).stem
    return {
        "id":          stem.lower().replace(" ", "_"),
        "name":        stem,
        "description": "",
        "components":  toporag_comps,
        "ports":       ports,
        "_skipped":    skipped,
    }


# ─── CLI エントリポイント ─────────────────────────────────

def _print_result(circuit: dict, fname: str):
    skipped = circuit.pop("_skipped", [])
    p = circuit["ports"]
    types = [c["type"] for c in circuit["components"]]
    skip_msg = f"\n           スキップ: {', '.join(skipped)}" if skipped else ""
    print(f"[OK]       {fname}")
    print(f"           部品={types}")
    print(f"           IN={p['input']}  OUT={p['output']}  GND={p['gnd']}{skip_msg}")


def convert_directory(input_dir: str, output_file: str):
    sch_files = sorted(Path(input_dir).glob("*.kicad_sch"))
    results: list[dict] = []
    cli = _find_kicad_cli()

    print(f"\n変換対象: {len(sch_files)} ファイル (.kicad_sch)")
    print("=" * 65)

    for sch_path in sch_files:
        fname = sch_path.stem
        try:
            circuit = convert_kicad_sch(str(sch_path), cli)
            if circuit is None:
                print(f"[スキップ] {fname} — 変換可能な部品なし / ポート未特定")
            else:
                _print_result(circuit, fname)
                results.append(circuit)
        except Exception as e:
            print(f"[エラー]   {fname}: {e}")

    print("=" * 65)
    print(f"変換成功: {len(results)} / {len(sch_files)} 回路")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"circuits": results}, f, ensure_ascii=False, indent=2)
    print(f"→ {output_file} に出力完了\n")


def convert_single(sch_path: str, output_file: str):
    fname = Path(sch_path).stem
    print(f"\n変換: {fname}")
    print("=" * 65)
    try:
        circuit = convert_kicad_sch(sch_path)
        if circuit is None:
            print("[スキップ] 変換可能な部品なし / ポート未特定")
            return
        _print_result(circuit, fname)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"circuits": [circuit]}, f, ensure_ascii=False, indent=2)
        print(f"→ {output_file} に出力完了")
    except Exception as e:
        print(f"[エラー] {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方:")
        print("  # ディレクトリ内の全 .kicad_sch を変換")
        print("  python kicad_sch_to_toporag.py <dir> [output.json]")
        print("")
        print("  # 単一ファイルを変換")
        print("  python kicad_sch_to_toporag.py <file.kicad_sch> [output.json]")
        sys.exit(1)

    target      = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "kicad_query.json"

    if os.path.isdir(target):
        convert_directory(target, output_file)
    else:
        convert_single(target, output_file)
