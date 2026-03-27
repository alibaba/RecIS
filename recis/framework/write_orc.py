"""
Single-sample wrapper + schema-first ORC writer (adds default label).

Input payload (single sample):
{
  "table_schema": { "col": {"type": "array<double>"}, ... },
  "fg_record": { "col": [...], "col2": [[...]], ... }   # values for ONE sample
}

Output ORC: 1 row table. Adds:
- fg_record["label"] = [[0.0]] if missing
- table_schema["label"] = {"type": "array<array<double>>"} if missing
"""

import pyarrow as pa
import pyarrow.orc as orc


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def hive_type_to_pa(type_str: str) -> pa.DataType:
    s = type_str.strip().lower()

    def parse(t: str) -> pa.DataType:
        t = t.strip().lower()

        if t.startswith("array<") and t.endswith(">"):
            inner = t[len("array<") : -1].strip()
            return pa.list_(parse(inner))

        if t.startswith("struct<") and t.endswith(">"):
            inner = t[len("struct<") : -1].strip()
            fields = []
            for item in _split_top_level(inner, sep=","):
                if ":" not in item:
                    raise ValueError(f"Invalid struct field: {item} in {type_str}")
                name, ftype = item.split(":", 1)
                fields.append(pa.field(name.strip(), parse(ftype.strip())))
            return pa.struct(fields)

        if t in ("bigint", "int", "integer"):
            return pa.int64()
        if t == "double":
            return pa.float64()
        if t == "float":
            return pa.float32()
        if t in ("string", "varchar", "char"):
            return pa.string()
        if t in ("boolean", "bool"):
            return pa.bool_()

        raise ValueError(f"Unsupported type: {t}")

    return parse(s)


def build_arrow_schema(table_schema: dict, cols: list[str]) -> pa.Schema:
    return pa.schema(
        [pa.field(c, hive_type_to_pa(table_schema[c]["type"])) for c in cols]
    )


def ensure_default_label(payload: dict):
    payload.setdefault("fg_record", {})
    payload.setdefault("table_schema", {})

    payload["fg_record"].setdefault("label", [[0.0]])
    payload["table_schema"].setdefault("label", {"type": "array<double>"})


def wrap_single_sample_to_1row_columns(fg_record_single_sample: dict) -> dict:
    """
    fg_record[col] is ONE sample's value, but might already be row-wrapped for label ([[0.0]]).
    Rule:
      - if it's already a 1-row column (list with len==1), keep as-is
      - else wrap to [value]
    """
    wrapped = {}
    for col, v in fg_record_single_sample.items():
        if isinstance(v, list) and len(v) == 1:
            wrapped[col] = v
        else:
            wrapped[col] = [v]
    return wrapped


def write_single_sample_orc(
    payload: dict, out_orc_path: str, compression: str = "zstd"
):
    ensure_default_label(payload)

    table_schema = payload["table_schema"]
    fg_record = payload["fg_record"]

    # write intersection columns
    cols = [c for c in fg_record.keys() if c in table_schema]
    if not cols:
        raise ValueError(
            "No columns to write: intersection(fg_record, table_schema) is empty"
        )

    schema = build_arrow_schema(table_schema, cols)

    # wrap to 1-row columns
    col_data = wrap_single_sample_to_1row_columns(fg_record)
    col_data = {c: col_data[c] for c in cols}

    table = pa.Table.from_pydict(col_data, schema=schema)
    orc.write_table(table, out_orc_path, compression=compression)
