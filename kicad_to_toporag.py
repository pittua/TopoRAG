"""
KiCad JSON ネットリスト → TopoRAG query_netlists.json コンバータ

対応部品:
  resistor   → R
  capacitor  → C
  inductor   → L
  diode      → D
  zener_diode → DZ
  switch     → SW

スキップ部品:
  電圧源/電流源, op_amp, 論理ゲート, MOSFET/BJT など
"""

import json
import os
import sys
import glob
from pathlib import Path

TYPE_MAP = {
    "resistor":    "R",
    "capacitor":   "C",
    "inductor":    "L",
    "diode":       "D",
    "zener_diode": "DZ",
    "switch":      "SW",
}

SKIP_TYPES = {
    "voltage_source", "ac_voltage_source", "dc_voltage_source",
    "current_source", "ground", "wire",
    "op_amp", "opamp", "operational_amplifier",
    "xor_gate", "and_gate", "or_gate", "not_gate", "nand_gate", "nor_gate",
    "nmos", "pmos", "njfet", "pjfet", "npn", "pnp",
    "bjt", "mosfet", "jfet",
}

# ポート推定用ノード名セット
GND_NAMES = {"GND", "0", "VSS", "DC_NEG", "AGND", "DGND", "GND_NET"}
IN_NAMES  = {"IN", "VIN", "INPUT", "AC1", "IN+", "VINP"}
OUT_NAMES = {"OUT", "VOUT", "OUTPUT", "DC_POS"}


def convert_component(comp: dict) -> dict | None:
    raw_type = comp.get("type", "").lower().replace(" ", "_").replace("-", "_")

    if raw_type in SKIP_TYPES:
        return None

    mapped = TYPE_MAP.get(raw_type)
    if mapped is None:
        return None

    connections = comp.get("connections") or {}
    nodes_list  = comp.get("nodes") or []

    if mapped == "D":
        if "anode" in connections and "cathode" in connections:
            terminals = {"anode": connections["anode"], "cathode": connections["cathode"]}
        elif nodes_list and len(nodes_list) >= 2:
            terminals = {"anode": nodes_list[0], "cathode": nodes_list[1]}
        else:
            return None
    else:
        if "t1" in connections and "t2" in connections:
            terminals = {"p": connections["t1"], "n": connections["t2"]}
        elif nodes_list and len(nodes_list) >= 2:
            terminals = {"p": nodes_list[0], "n": nodes_list[1]}
        elif connections:
            vals = list(connections.values())
            if len(vals) >= 2:
                terminals = {"p": vals[0], "n": vals[1]}
            else:
                return None
        else:
            return None

    return {"id": comp["id"], "type": mapped, "terminals": terminals}


def infer_ports(components: list[dict]) -> dict | None:
    all_nodes: set[str] = set()
    for c in components:
        for v in c["terminals"].values():
            all_nodes.add(str(v))

    gnd = next((n for n in all_nodes if n.upper() in GND_NAMES), None)
    inp = next((n for n in all_nodes if n.upper() in IN_NAMES),  None)
    out = next((n for n in all_nodes if n.upper() in OUT_NAMES), None)

    if gnd and inp and out:
        return {"input": inp, "output": out, "gnd": gnd}
    return None


def convert_file(path: str) -> dict | None:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    fname = Path(path).stem
    name  = raw.get("circuit_name", fname)
    desc  = raw.get("description", "")

    converted = []
    skipped   = []

    for c in raw.get("components", []):
        result = convert_component(c)
        if result:
            converted.append(result)
        else:
            skipped.append(f"{c['id']}({c.get('type','?')})")

    if not converted:
        return None

    ports = infer_ports(converted)
    if ports is None:
        return None

    return {
        "id":          fname.lower(),
        "name":        name,
        "description": desc,
        "components":  converted,
        "ports":       ports,
        "_skipped":    skipped,
    }


def main(input_dir: str, output_file: str):
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    results = []

    print(f"\n変換対象: {len(json_files)} ファイル")
    print("=" * 60)

    for path in json_files:
        fname = Path(path).stem
        circuit = convert_file(path)
        if circuit is None:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            types = [c.get("type", "?") for c in raw.get("components", [])]
            print(f"[スキップ] {fname}")
            print(f"           部品種別: {types}")
        else:
            skipped = circuit.pop("_skipped")
            results.append(circuit)
            n_comp = len(circuit["components"])
            p = circuit["ports"]
            comp_types = [c["type"] for c in circuit["components"]]
            skip_msg = f"\n           スキップ: {', '.join(skipped)}" if skipped else ""
            print(f"[OK]       {fname}")
            print(f"           部品={comp_types}  IN={p['input']}  OUT={p['output']}  GND={p['gnd']}{skip_msg}")

    print("=" * 60)
    print(f"変換成功: {len(results)} / {len(json_files)} 回路")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"circuits": results}, f, ensure_ascii=False, indent=2)
    print(f"→ {output_file} に出力完了\n")
    return len(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python kicad_to_toporag.py <input_dir> [output_file]")
        sys.exit(1)
    input_dir   = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "kicad_query.json"
    main(input_dir, output_file)
