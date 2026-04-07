from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _materialize_config(cfg: Any) -> dict:
    if is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = dict(cfg)
    return _to_serializable(d)


def run_parameter_freeze_check(
    cfg: Any,
    snapshot_path: str | Path,
    auto_write_if_missing: bool = True,
) -> tuple[pd.DataFrame, str, bool]:
    snap = Path(snapshot_path)
    snap.parent.mkdir(parents=True, exist_ok=True)

    current = _materialize_config(cfg)
    # Governance controls themselves are not part of the locked strategy definition.
    current.pop("enforce_frozen_config", None)
    current.pop("auto_write_frozen_config_if_missing", None)

    if not snap.exists():
        if auto_write_if_missing:
            snap.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
            out = pd.DataFrame(
                [{"param": "_snapshot", "status": "CREATED", "current": str(snap), "frozen": ""}]
            )
            text = (
                "Parameter Freeze Summary\n"
                f"Snapshot created at: {snap}\n"
                "Lock status: baseline created (next runs will compare against this)."
            )
            return out, text, True
        out = pd.DataFrame(
            [{"param": "_snapshot", "status": "MISSING", "current": str(snap), "frozen": ""}]
        )
        text = (
            "Parameter Freeze Summary\n"
            f"Snapshot missing: {snap}\n"
            "Lock status: no baseline snapshot available."
        )
        return out, text, False

    frozen = json.loads(snap.read_text(encoding="utf-8"))
    keys = sorted(set(current.keys()) | set(frozen.keys()))
    rows = []
    matched = True
    for k in keys:
        cur = current.get(k, "<MISSING>")
        fro = frozen.get(k, "<MISSING>")
        same = cur == fro
        if not same:
            matched = False
        rows.append(
            {
                "param": k,
                "status": "MATCH" if same else "DIFF",
                "current": json.dumps(cur, sort_keys=True) if isinstance(cur, (dict, list)) else str(cur),
                "frozen": json.dumps(fro, sort_keys=True) if isinstance(fro, (dict, list)) else str(fro),
            }
        )

    out = pd.DataFrame(rows)
    n_diff = int((out["status"] == "DIFF").sum())
    text = (
        "Parameter Freeze Summary\n"
        f"Snapshot: {snap}\n"
        f"Differences vs frozen config: {n_diff}\n"
        + ("Lock status: MATCHED." if matched else "Lock status: CHANGED (potential re-tuning risk).")
    )
    return out, text, matched
