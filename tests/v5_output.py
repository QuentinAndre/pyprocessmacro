"""
Parser for the PROCESS for R (version 5) output files in tests/Results/v5.

parse(text) returns a dict:
    outcomes:  {outcome name: {"coefficients": DataFrame indexed by term (Cons, variable names, products
               written a*b), "summary": one-row DataFrame}}, main models only (not the total-effect model)
    direct:    DataFrame with the moderator columns (if any) then effect, se, stat (t or Z), p, LLCI, ULCI
    indirect:  {"unmoderated": DataFrame indexed by label (TOTAL, mediator names or path labels, contrast
               definitions as "a minus b"), or "conditional": {path label: DataFrame with moderator columns
               then Effect, BootSE, BootLLCI, BootULCI}}
    indices:   {path label: {"MM": DataFrame indexed by moderator, "PMM": DataFrame indexed by moderator,
               "MMM": one-row DataFrame, "CMM": {focal moderator: DataFrame with the other moderator's
               column then Index...}}}
    effsize:   DataFrame indexed by label for the completely standardized effects, or None
    notes:     list of lines from the notes section
"""
import re

import numpy as np
import pandas as pd


def _block(lines, start):
    """Rows of one column block: (header tokens, labels, rows, next index)."""
    header = lines[start].split()
    labels, rows = [], []
    j = start + 1
    while j < len(lines) and lines[j].strip():
        parts = lines[j].split()
        if len(parts) == len(header) + 1:
            labels.append(parts[0])
            parts = parts[1:]
        elif len(parts) != len(header):
            break
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            break
        j += 1
    return header, labels, rows, j


def _table(lines, start):
    """
    Parse the whitespace-separated table whose header is lines[start]. PROCESS wraps tables wider than
    seven columns into a second column block after a blank line; such a block is glued back on.
    Returns (DataFrame, next index).
    """
    header, labels, rows, j = _block(lines, start)
    k = j
    while k < len(lines) and not lines[k].strip():
        k += 1
    if rows and k < len(lines):
        tokens = lines[k].split()
        looks_like_columns = tokens and all(re.fullmatch(r"[A-Za-z'_()\-]+", t) for t in tokens) \
            and not lines[k].rstrip().endswith(":")
        if looks_like_columns:
            more_header, _, more_rows, k2 = _block(lines, k)
            if len(more_rows) == len(rows):
                header = header + more_header
                rows = [r + m for r, m in zip(rows, more_rows)]
                j = k2
    frame = pd.DataFrame(rows, columns=[_clean(h) for h in header])
    if labels and len(labels) == len(rows):
        frame.index = labels
    return frame, j


def _clean(column):
    column = re.sub(r"\(HC\d\)", "", column)
    return {"effect": "effect", "coeff": "coeff", "se": "se", "t": "stat", "Z": "stat"}.get(column, column)


def _path(line):
    return " -> ".join(t.strip() for t in line.split("->"))


def parse(text):
    lines = [l.rstrip() for l in text.replace("\r\n", "\n").split("\n")]
    out = {"outcomes": {}, "direct": None, "indirect": {}, "indices": {}, "effsize": None, "notes": []}
    products = {}
    i = 0
    in_total_effect_model = False
    current_path = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if "TOTAL EFFECT MODEL" in stripped:
            in_total_effect_model = True
        elif stripped.startswith("Outcome Variable:") and not in_total_effect_model:
            name = stripped.split(":", 1)[1].strip()
            block = {"coefficients": None, "summary": None, "conditional": None}
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("Outcome Variable:") \
                    and not lines[j].startswith("****") and not lines[j].strip().startswith("TOTAL EFFECT MODEL"):
                s = lines[j].strip()
                if s == "Model Summary:":
                    block["summary"], j = _table(lines, j + 1)
                    continue
                if s == "Model:":
                    block["coefficients"], j = _table(lines, j + 1)
                    continue
                if s == "Conditional effects of the focal predictor at values of the moderator(s):":
                    block["conditional"], j = _table(lines, j + 1)
                    continue
                if s == "Product terms key:":
                    j += 1
                    while j < len(lines) and lines[j].strip():
                        key, rhs = lines[j].split(":", 1)
                        products[key.strip()] = "*".join(re.split(r"\s{2,}x\s{2,}", rhs.strip()))
                        j += 1
                    continue
                j += 1
            block["coefficients"].index = [
                "Cons" if t == "constant" else products.get(t, t) for t in block["coefficients"].index
            ]
            out["outcomes"][name] = block
            i = j
            continue
        elif stripped in ("Direct effect of X on Y:", "Conditional direct effect(s) of X on Y:"):
            out["direct"], i = _table(lines, i + 1)
            out["direct"] = out["direct"].rename(columns={"Effect": "effect"})  # logit tables capitalize it
            continue
        elif stripped == "Indirect effect(s) of X on Y:":
            table, i = _table(lines, i + 1)
            out["indirect"]["unmoderated"] = table
            continue
        elif stripped.startswith("Indirect effect key"):
            key = {}
            j = i + 1
            while j < len(lines) and lines[j].strip():
                name, path = lines[j].split(None, 1)
                key[name] = _path(path)
                j += 1
            out["path_key"] = key
            table = out["indirect"]["unmoderated"]
            table.index = [key.get(l, l) for l in table.index]
            if out["effsize"] is not None:
                out["effsize"].index = [key.get(l, l) for l in out["effsize"].index]
            i = j
            continue
        elif stripped.startswith("Specific indirect effect contrast definition"):
            j = i + 1
            defs = {}
            while j < len(lines) and lines[j].strip():
                name, rest = lines[j].split(None, 1)
                key = out.get("path_key", {})
                parts = re.sub(r"\s+", " ", rest.strip()).split(" minus ")
                defs[name] = " minus ".join(key.get(t.strip(), t.strip()) for t in parts)
                j += 1
            for table in (out["indirect"].get("unmoderated"), out["effsize"]):
                if table is not None:
                    table.index = [defs.get(l, l) for l in table.index]
            i = j
            continue
        elif stripped == "INDIRECT EFFECT:":
            j = i + 1
            while not lines[j].strip():
                j += 1
            current_path = _path(lines[j])
            j += 1
            while not lines[j].strip():
                j += 1
            table, j = _table(lines, j)
            out["indirect"].setdefault("conditional", {})[current_path] = table
            out["indices"].setdefault(current_path, {})
            i = j
            continue
        elif stripped == "Index of moderated mediation:":
            table, i = _table(lines, i + 1)
            out["indices"][current_path]["MM"] = table
            continue
        elif stripped == "Indices of partial moderated mediation:":
            table, i = _table(lines, i + 1)
            out["indices"][current_path]["PMM"] = table
            continue
        elif stripped == "Index of moderated moderated mediation:":
            table, i = _table(lines, i + 1)
            out["indices"][current_path]["MMM"] = table
            continue
        elif stripped.startswith("Indices of conditional moderated mediation by "):
            focal = stripped.split("by ")[1].rstrip(":").strip()
            table, i = _table(lines, i + 1)
            out["indices"][current_path].setdefault("CMM", {})[focal] = table
            continue
        elif stripped.startswith("Completely standardized"):
            out["effsize"], i = _table(lines, i + 1)
            continue
        elif stripped.startswith("ANALYSIS NOTES"):
            out["notes"] = [l.strip() for l in lines[i + 1:] if l.strip()]
            break
        i += 1
    key = out.get("path_key")
    if key:  # contrast definitions may precede the path key: relabel them with the paths
        for table in (out["indirect"].get("unmoderated"), out["effsize"]):
            if table is not None:
                table.index = [" minus ".join(key.get(t, t) for t in label.split(" minus ")) for label in table.index]
    if out["direct"] is None:  # moderation-only models print the conditional effects inside the outcome block
        for block in out["outcomes"].values():
            if block["conditional"] is not None:
                out["direct"] = block["conditional"]
                break
    return out


def moderator_values(parsed, moderators):
    """Spotlight values PROCESS used, per moderator name, gathered from every conditional table."""
    values = {m: [] for m in moderators}
    tables = [parsed["direct"]] + list(parsed["indirect"].get("conditional", {}).values())
    for table in tables:
        if table is None:
            continue
        for m in moderators:
            if m in table.columns:
                values[m].extend(table[m].to_numpy(dtype=float))
    out = {}
    for m, found in values.items():
        distinct = []
        for v in sorted(found):  # the same value can print with a different last decimal in two tables
            if not distinct or abs(v - distinct[-1]) > 1e-5:
                distinct.append(float(v))
        if distinct:
            out[m] = distinct
    return out
