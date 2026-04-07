from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_quality_control_alias(source_json: str | Path, alias_json: str | Path) -> Path:
    source_json = Path(source_json)
    alias_json = Path(alias_json)
    alias_json.parent.mkdir(parents=True, exist_ok=True)
    alias_json.write_text(source_json.read_text())
    return alias_json


def load_single_slide_qc_row(qc_results_json: str | Path) -> dict:
    payload = json.loads(Path(qc_results_json).read_text())
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError(f"expected a single-slide QC JSON payload in {qc_results_json}")
    return payload[0]


def write_batch_results_csv(slide_payloads: list[dict], output_csv: str | Path) -> Path:
    rows = []
    for payload in slide_payloads:
        row = {
            **load_single_slide_qc_row(payload["qc_results_json"]),
            "input_wsi": payload["input_wsi"],
            "output_dir": payload["output_dir"],
            "predictions_csv": payload.get("predictions_csv", ""),
            "qc_results_json": payload["qc_results_json"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return output_csv
