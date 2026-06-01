"""
KiCad .kicad_sch → TopoRAG query_netlists.json コンバータ

動作フロー:
  1. kicad-cli で .kicad_sch から kicadsexpr ネットリストを生成
  2. S式をパースして nets / components / libparts を抽出
  3. libparts のスキーマピン番号 → 役割名 (K/A, G/D/S 等) を解決
  4. nets セクションから (部品ref, ピン番号) → ネット名 マッピングを構築
  5. 部品種別・terminals・ports を解決して TopoRAG 形式に変換

対応部品:
  R / C / L / D / DZ / MOSFET(SW として近似)

スキップ:
  電圧源・電流源・BJT・OpAmp・論理ゲート 等
"""

import sys
import os
import re
import json
import tempfile
import subprocess
import shutil
from pathlib import Path


# ─── kicad-cli の場所 ───────────────────────────────────────

_KICAD_CLI_FALLBACK = (
    r"C:\Users\fukui\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe"
)


def _find_kicad_cli() -> str:
    cmd = shutil.which("kicad-cli") or shutil.which("kicad-cli.exe")
    if cmd:
        return cmd
    if os.path.exists(_KICAD_CLI_FALLBACK):
        return _KICAD_CLI_FALLBACK
    raise FileNotFoundError(
        "kicad-cli が見つかりません。KiCad がインストールされているか確認してください。"
    )


# ─── 部品種別マッピング ────────────────────────────────────

# Sim.Device プロパティ値 → TopoRAG 種別
SIM_DEVICE_MAP: dict[str, str | None] = {
    "R": "R", "C": "C", "L": "L",
    "D": "D", "DZ": "DZ",
    "NMOS": "SW", "PMOS": "SW", "VDMOS": "SW",
    "V": None, "I": None,          # 電源・電流源スキップ
    "NPN": None, "PNP": None,      # BJT スキップ
    "OPAMP": None,                  # OpAmp スキップ
}

# part 名キーワード → TopoRAG 種別（Sim.Device がない場合）
PART_KEYWORD_MAP = [
    ("ZENER",    "DZ"),
    ("SCHOTTKY", "D"),
    ("DIODE",    "D"),
    ("INDUCTOR", "L"),
    ("CAPACITOR","C"),
    ("RESISTOR", "R"),
    ("MOSFET",   "SW"),
    ("JFET",     None),
    ("BJT",      None),
    ("OPAMP",    None),
]

# ref 頭文字 → TopoRAG 種別（最終フォールバック）
REF_PREFIX_MAP = {
    "R": "R", "C": "C", "L": "L",
    "D": "D", "Z": "DZ",
    "Q": "SW", "M": "SW",
    "V": None, "I": None,
    "U": None, "IC": None,
    "X": None, "J": None, "P": None, "T": None,
}

# ─── ネット名パターン ───────────────────────────────────────

GND_RE = re.compile(r"^(GND|/GND|\bGND\b|VSS|AGND|DGND|0)$", re.IGNORECASE)
IN_RE  = re.compile(r"(^|/)(V?IN|VIN|VCC|VDD|V\+|INPUT|SRC|SOURCE|SIGNAL|AC_IN)$", re.IGNORECASE)
OUT_RE = re.compile(r"(^|/)(V?OUT|VOUT|OUTPUT|RECT_OUT|DC_POS|LOAD)$", re.IGNORECASE)


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

            components.append({
                "ref": ref, "lib": lib, "part": part,
                "sim_device": sim_device,
                "pin_role": pin_role,
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

    if ttype == "SW":
        # MOSFET: Drain → p, Source → n (Gate/Bulk はスキップ)
        GATE_ROLES  = {"G", "GATE", "B", "BULK"}
        DRAIN_ROLES = {"D", "DRAIN", "+"}
        SRC_ROLES   = {"S", "SOURCE", "-"}
        p = n = None
        for pnum, net in pin_nets.items():
            role = pin_role.get(pnum, "")
            if role in DRAIN_ROLES:
                p = net
            elif role in SRC_ROLES:
                n = net
        if p and n:
            return {"p": p, "n": n}
        # 役割不明 → Gate/Bulk 以外のピンから 2 本取る
        power_pins = [(pnum, net) for pnum, net in pin_nets.items()
                      if pin_role.get(pnum, "") not in GATE_ROLES]
        if len(power_pins) >= 2:
            power_pins.sort()
            return {"p": power_pins[0][1], "n": power_pins[1][1]}
        return None

    # R / C / L
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

def _infer_ports(all_nets: set[str],
                 pin_to_net: dict,
                 orig_comps: list[dict]) -> dict | None:

    # GND
    gnd = next((n for n in all_nets if GND_RE.match(n)), None)
    if not gnd:
        return None

    # 電圧源 + 端子ネットを入力候補に追加
    vsrc_pos: list[str] = []
    for comp in orig_comps:
        if comp["sim_device"] == "V":
            ref = comp["ref"]
            # + ピンを探す（role=+ or pin 1）
            role_to_pin = {v: k for k, v in comp["pin_role"].items()}
            pos_pin = role_to_pin.get("+") or role_to_pin.get("1") or "1"
            net = pin_to_net.get((ref, pos_pin))
            if net and net != gnd:
                vsrc_pos.append(net)

    inp = next((n for n in all_nets if IN_RE.search(n)), None)
    if not inp:
        inp = vsrc_pos[0] if vsrc_pos else None

    out = next((n for n in all_nets
                if OUT_RE.search(n) and n not in (gnd, inp)), None)

    # 出力が見つからない → 最多ノード接続の非 GND/入力ネット
    if not out:
        degree: dict[str, int] = {}
        for net in pin_to_net.values():
            if net not in (gnd, inp):
                degree[net] = degree.get(net, 0) + 1
        if degree:
            out = max(degree, key=lambda n: degree[n])

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

        toporag_comps.append({"id": ref, "type": ttype, "terminals": terminals})

    if not toporag_comps:
        return None

    all_nets = set(pin_to_net.values())
    ports = _infer_ports(all_nets, pin_to_net, orig_comps)
    if ports is None:
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
