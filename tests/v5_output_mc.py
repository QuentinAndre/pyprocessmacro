"""
Parser for the multicategorical comparison files in tests/Results/v5/mc (#17), the output of PROCESS for R
version 5 with a multicategorical X (mcx) and/or W (mcw). Builds on tests/v5_output.py.
"""
import pandas as pd

from tests.v5_output import _block, _clean, _path, parse


def _plain_table(lines, start):
    """A table without the wrapped-block detection (these files never wrap); rows may carry a code label."""
    header, labels, rows, j = _block(lines, start)
    frame = pd.DataFrame(rows, columns=[_clean(h) for h in header])
    if labels and len(labels) == len(rows):
        frame.insert(0, "X", labels)
    return frame, j


def _skip_blank(lines, i):
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def parse_mc(text):
    """
    Returns a dict:
        codings:     {"X": DataFrame, "W": DataFrame} category-to-code tables
        outcomes:    as v5_output.parse()
        direct:      DataFrame of the (relative, conditional) direct effects, with an "X" column for the code when
                     X is multicategorical and one column per moderator when moderated; None if absent
        omnibus:     one-row DataFrame of the omnibus test of the direct effect, or None
        conditional: for the moderation-only models: with a multicategorical X a list of blocks
                     {"at": {moderator: value}, "effects": DataFrame with X column, "equality": DataFrame}; with
                     a continuous X a single DataFrame under "table"
        indirect:    {path: DataFrame} of the (relative) indirect effects: an "X" column for the code, moderator
                     columns when conditional
        indices:     {path: {label: DataFrame}} indices of moderated mediation: label is the X code when X is
                     multicategorical ("" otherwise); the table is indexed by the moderator name or its codes
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out = {"codings": {}, "outcomes": parse(text)["outcomes"], "direct": None, "omnibus": None,
           "conditional": None, "indirect": {}, "indices": {}}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("Coding of categorical ") and s.endswith("variable for analysis:"):
            letter = s.split("Coding of categorical ")[1][0]
            out["codings"][letter], i = _plain_table(lines, i + 1)
            continue
        if s in ("Relative direct effects of X on Y:", "Relative conditional direct effects of X on Y:",
                 "Direct effect of X on Y:", "Conditional direct effect(s) of X on Y:"):
            out["direct"], i = _plain_table(lines, i + 1)
            out["direct"] = out["direct"].rename(columns={"Effect": "effect"})
            continue
        if s.startswith("Omnibus") and "direct effect of X on Y" in s:
            out["omnibus"], i = _plain_table(lines, i + 1)
            continue
        if s == "Conditional effects of the focal predictor at values of the moderator(s):":
            j = _skip_blank(lines, i + 1)
            if lines[j].strip().startswith("Moderator value(s):"):
                blocks = []
                while j < len(lines) and lines[j].strip().startswith("Moderator value(s):"):
                    j = _skip_blank(lines, j + 1)
                    at = {}
                    while lines[j].strip():
                        name, value = lines[j].split()
                        at[name] = float(value)
                        j += 1
                    j = _skip_blank(lines, j)
                    effects, j = _plain_table(lines, j)
                    effects = effects.rename(columns={"Effect": "effect"})
                    j = _skip_blank(lines, j)
                    equality = None
                    if lines[j].strip().startswith("Test of equality of conditional means"):
                        equality, j = _plain_table(lines, j + 1)
                        j = _skip_blank(lines, j)
                    if lines[j].strip().startswith("Estimated conditional means"):
                        _, j = _plain_table(lines, j + 1)
                        j = _skip_blank(lines, j)
                    if lines[j].strip().startswith("----------"):
                        j = _skip_blank(lines, j + 1)
                    blocks.append({"at": at, "effects": effects, "equality": equality})
                out["conditional"] = blocks
                i = j
            else:
                table, i = _plain_table(lines, j)
                out["conditional"] = {"table": table.rename(columns={"Effect": "effect"})}
            continue
        if s in ("Relative indirect effects of X on Y:", "Indirect effect(s) of X on Y:"):
            j = _skip_blank(lines, i + 1)
            while j < len(lines) and "->" in lines[j]:
                path = _path(lines[j])
                j = _skip_blank(lines, j + 1)
                table, j = _plain_table(lines, j)
                out["indirect"][path] = table
                j = _skip_blank(lines, j)
            i = j
            continue
        if s in ("Relative conditional indirect effects of X on Y:", "Conditional indirect effects of X on Y:"):
            j = _skip_blank(lines, i + 1)
            assert lines[j].strip() == "INDIRECT EFFECT:", lines[j]
            j = _skip_blank(lines, j + 1)
            path = _path(lines[j])
            j = _skip_blank(lines, j + 1)
            tables, current = [], ""
            while j < len(lines) and not lines[j].startswith("****"):
                line = lines[j].strip()
                if line.startswith("Index of moderated mediation"):
                    j += 1
                    if lines[j].strip().startswith("("):  # "(differences beween conditional indirect effects):"
                        j += 1
                    j = _skip_blank(lines, j)
                    header, labels, rows, j = _block(lines, j)
                    table = pd.DataFrame(rows, columns=[_clean(h) for h in header], index=labels or None)
                    out["indices"].setdefault(path, {})[current] = table
                else:
                    table, j = _plain_table(lines, j)
                    if "X" in table.columns:
                        current = table["X"].iloc[0]
                    tables.append(table)
                j = _skip_blank(lines, j)
            out["indirect"][path] = pd.concat(tables, ignore_index=True)
            i = j
            continue
        i += 1
    return out
