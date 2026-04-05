#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TRACKING_ID_RE = re.compile(r"^[GS]\d+[A-Za-z]?$")
COLAB_DRIVE_MARKER = "EBME398_ArtifactDetection/Exp_2/"
LABEL_NORMALIZATION = {
    "clean": "clean",
    "tissue_damge": "tissue_damage",
    "tissue_damage": "tissue_damage",
    "tissuedamage": "tissue_damage",
    "blurry+fold": "blurry+fold",
    "blur+fold": "blurry+fold",
    "fold+blur": "blurry+fold",
    "fold+blurry": "blurry+fold",
}


def nonempty(value: Any) -> bool:
    return value is not None and value != ""


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if nonempty(value):
            return value
    return None


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower().replace(" ", "").replace("-", "_")
    return LABEL_NORMALIZATION.get(text, str(label))


def cell_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def round_cell(value: Any, digits: int = 6) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, digits)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def resolve_colab_path(raw_path: str | None, source_root: Path, metrics_path: Path) -> Path | None:
    if not raw_path:
        return None
    raw = Path(raw_path)
    candidates = []
    if raw.exists():
        candidates.append(raw)
    candidates.append(metrics_path.parent / raw.name)
    candidates.append(metrics_path.parent.parent / raw.name)
    text = str(raw_path)
    if COLAB_DRIVE_MARKER in text:
        suffix = text.split(COLAB_DRIVE_MARKER, 1)[1]
        candidates.append(source_root / suffix)
    if "Exp_2/" in text:
        suffix = text.split("Exp_2/", 1)[1]
        candidates.append(source_root / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_workbook(workbook_path: Path) -> dict[str, dict[str, Any]]:
    workbook = pd.ExcelFile(workbook_path)
    entries: dict[str, dict[str, Any]] = {}
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name)
        df.columns = [str(col).strip() for col in df.columns]
        for _, row in df.iterrows():
            tracking_id = cell_text(row.get("ID", ""))
            if not TRACKING_ID_RE.match(tracking_id):
                continue
            entries[tracking_id] = {
                "tracking_sheet": sheet_name,
                "tracking_id": tracking_id,
                "tracking_model_type": cell_text(row.get("Model Type", "")),
                "tracked_data_used": cell_text(row.get("Data Used", "")),
                "tracked_method_summary": cell_text(row.get("Method Summary", "")),
                "tracked_metrics_to_log": cell_text(row.get("Metrics to Log", "")),
                "tracked_success_criterion": cell_text(row.get("Success Criterion", "")),
            }
    return entries


def load_split_summary(split_path: Path) -> dict[str, Any]:
    split = read_json(split_path)
    train = split.get("train", [])
    val = split.get("val", [])
    test = split.get("test", [])
    test_prefixes = sorted({str(item).split("-", 1)[0] for item in test})
    return {
        "rule": split.get("rule"),
        "train_wsi_count": len(train),
        "val_wsi_count": len(val),
        "test_wsi_count": len(test),
        "test_prefixes": test_prefixes,
        "test_only_sr040": test_prefixes == ["SR040"],
        "split_path": str(split_path),
    }


def extract_confusion_binary(test: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    direct = [test.get("tn"), test.get("fp"), test.get("fn"), test.get("tp")]
    if all(value is not None for value in direct):
        return tuple(int(value) for value in direct)
    cm = test.get("cm")
    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
        tn, fp = cm[0]
        fn, tp = cm[1]
        return int(tn), int(fp), int(fn), int(tp)
    return None, None, None, None


def classification_report_positive_f1(test: dict[str, Any]) -> float | None:
    report = test.get("classification_report")
    if not isinstance(report, dict):
        return None
    positive = report.get("1")
    if isinstance(positive, dict):
        return positive.get("f1-score")
    return None


def derive_binary_metrics(test: dict[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = extract_confusion_binary(test)
    accuracy = first_nonempty(test.get("accuracy"), test.get("acc"))
    if accuracy is None:
        accuracy = safe_div((tn or 0) + (tp or 0), (tn or 0) + (fp or 0) + (fn or 0) + (tp or 0))
    precision = first_nonempty(test.get("precision"), test.get("precision_1"), test.get("precision_pos"))
    if precision is None:
        precision = safe_div(tp, (tp or 0) + (fp or 0))
    sensitivity = first_nonempty(test.get("recall"), test.get("sensitivity"), test.get("recall_sensitivity"))
    if sensitivity is None:
        sensitivity = safe_div(tp, (tp or 0) + (fn or 0))
    specificity = first_nonempty(test.get("specificity"))
    if specificity is None:
        specificity = safe_div(tn, (tn or 0) + (fp or 0))
    f1 = first_nonempty(test.get("f1"), test.get("f1_pos"), classification_report_positive_f1(test))
    if f1 is None and precision is not None and sensitivity is not None and (precision + sensitivity):
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision_pos": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }


def infer_run_family(rel_path: str, metrics: dict[str, Any]) -> dict[str, Any]:
    parts = rel_path.split("/")
    task_dir = parts[0]
    group_dir = parts[1]
    filename = Path(rel_path).name
    balance = "balance" in rel_path.lower()
    if task_dir == "Binary":
        if group_dir == "G1":
            return {"task": "binary", "experiment_group": "G1", "variant": str(metrics.get("model", "handcrafted_qc")), "balanced_variant": False}
        if group_dir == "G1_balance":
            variant = "KAN" if "_kan_" in filename else "SVM" if "_svm_" in filename else "XGBoost" if "_xgb_" in filename else filename.replace("_metrics.json", "")
            return {"task": "binary", "experiment_group": "G1", "variant": variant, "balanced_variant": True}
        if group_dir == "G2":
            return {"task": "binary", "experiment_group": "G2", "variant": str(metrics.get("model_arch", "resnet")), "balanced_variant": False}
        if group_dir == "G2_balance":
            return {"task": "binary", "experiment_group": "G2", "variant": str(metrics.get("model_arch", "resnet")), "balanced_variant": True}
        if group_dir == "G3":
            subdir = parts[2]
            variant = "KAN" if "KAN" in subdir else "MLP"
            return {"task": "binary", "experiment_group": "G3", "variant": variant, "balanced_variant": balance}
        if group_dir == "G4":
            return {"task": "binary", "experiment_group": "G4", "variant": str(metrics.get("exp_name", filename.replace("_metrics.json", ""))), "balanced_variant": bool(metrics.get("data", {}).get("train_balance"))}
    if task_dir == "Multi_class":
        if group_dir == "S1":
            return {"task": "multiclass", "experiment_group": "S1", "variant": str(metrics.get("model", "handcrafted_multihead")), "balanced_variant": False}
        if group_dir == "S1_balance":
            return {"task": "multiclass", "experiment_group": "S1", "variant": str(metrics.get("model", "handcrafted_multihead")), "balanced_variant": True}
        if group_dir == "S2_balance":
            return {"task": "multiclass", "experiment_group": "S2", "variant": str(metrics.get("model_arch", "cnn_multihead")), "balanced_variant": True}
        if group_dir == "S3":
            return {"task": "multiclass", "experiment_group": "S3", "variant": "Frozen UNI + MLP", "balanced_variant": False}
        if group_dir == "S3_balance":
            return {"task": "multiclass", "experiment_group": "S3", "variant": str(metrics.get("model_arch", "Frozen UNI + balanced head")), "balanced_variant": True}
        if group_dir == "S4_new":
            variant = "tile_fused_multiclass_multihead" if "multihead" in filename else "tile_fused_multiclass"
            return {"task": "multiclass", "experiment_group": "S4", "variant": variant, "balanced_variant": bool(metrics.get("data", {}).get("train_balance"))}
    raise ValueError(f"Unsupported metrics path: {rel_path}")


def tracking_id_for_run(experiment_group: str, variant: str) -> str | None:
    if experiment_group == "G1":
        return "G1"
    if experiment_group == "G2":
        return "G2"
    if experiment_group == "G3":
        return "G3b" if variant.upper() == "KAN" else "G3a"
    if experiment_group == "S1":
        return "S1"
    if experiment_group == "S2":
        return "S2"
    if experiment_group == "S3":
        return "S3"
    if experiment_group == "S4":
        return "S4"
    return None


def best_split_source(metrics: dict[str, Any], metrics_path: Path) -> str:
    for key in ("splits_source", "splits_path"):
        if nonempty(metrics.get(key)):
            return str(metrics[key])
    sibling = metrics_path.parent / "splits_used.json"
    if sibling.exists():
        payload = read_json(sibling)
        return str(payload.get("splits_source", sibling))
    return ""


def find_prediction_artifacts(metrics: dict[str, Any], metrics_path: Path, source_root: Path) -> list[str]:
    artifacts: list[str] = []
    csv_candidate = metrics_path.parent / "preds_test.csv"
    if csv_candidate.exists():
        artifacts.append(relative_to(csv_candidate, source_root.parent))
    for key in ("probs_path", "ytrue_path", "preds_path", "logits_path"):
        resolved = resolve_colab_path(metrics.get("test", {}).get(key), source_root, metrics_path)
        if resolved:
            artifacts.append(relative_to(resolved, source_root.parent))
    return sorted(set(artifacts))


def top_level_hparams(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "seed",
        "model",
        "model_arch",
        "patch_size0",
        "target_ps",
        "threshold",
        "train_balancing",
        "use_smote",
        "imbalance",
        "kan_hparams",
        "svm_hparams",
        "xgb_hparams",
    )
    payload = {key: metrics.get(key) for key in keys if key in metrics}
    for key in ("data",):
        if key in metrics:
            payload[key] = metrics[key]
    return payload


def load_label_mapping_for_run(metrics_path: Path, metrics: dict[str, Any], project_root: Path) -> tuple[list[str], str]:
    label_map_path = metrics_path.parent / "label_mapping.json"
    if label_map_path.exists():
        payload = read_json(label_map_path)
        id2label = payload.get("ID2LABEL", {})
        ordered = [normalize_label(id2label[str(idx)]) for idx in sorted(int(key) for key in id2label)]
        return [label for label in ordered if label], relative_to(label_map_path, project_root)
    top_level = metrics.get("id2label")
    if isinstance(top_level, dict):
        ordered = [normalize_label(top_level[str(idx)]) for idx in sorted(int(key) for key in top_level)]
        return [label for label in ordered if label], "metrics.id2label"
    top_label2id = metrics.get("label2id")
    if isinstance(top_label2id, dict):
        ordered = sorted(((int(idx), normalize_label(label)) for label, idx in top_label2id.items()), key=lambda item: item[0])
        return [label for _, label in ordered if label], "metrics.label2id"
    class_names = metrics.get("test", {}).get("class_names")
    if isinstance(class_names, list) and class_names:
        return [normalize_label(name) for name in class_names], "metrics.test.class_names"
    ovr_auc = metrics.get("test", {}).get("ovr_auc")
    if isinstance(ovr_auc, dict) and ovr_auc:
        return [normalize_label(name) for name in ovr_auc.keys()], "metrics.test.ovr_auc"
    if "S4_new" in metrics_path.as_posix():
        s1_label_map = project_root / "source" / "working_dir" / "10x_512px_0px_overlap" / "experiments" / "Multi_class" / "S1" / "label_mapping.json"
        if s1_label_map.exists():
            payload = read_json(s1_label_map)
            id2label = payload.get("ID2LABEL", {})
            ordered = [normalize_label(id2label[str(idx)]) for idx in sorted(int(key) for key in id2label)]
            return [label for label in ordered if label], relative_to(s1_label_map, project_root)
    return [], ""


def extract_supports_from_report(report: dict[str, Any], label_order: list[str]) -> dict[str, int]:
    supports: dict[str, int] = {}
    for index, label in enumerate(label_order):
        entry = report.get(str(index))
        if isinstance(entry, dict) and "support" in entry:
            supports[label] = int(entry["support"])
    return supports


def derive_multiclass_metrics(test: dict[str, Any], label_order: list[str]) -> dict[str, Any]:
    accuracy = first_nonempty(test.get("accuracy"), test.get("acc"))
    macro_f1 = test.get("macro_f1")
    weighted_f1 = test.get("weighted_f1")
    if weighted_f1 is None and isinstance(test.get("classification_report"), dict):
        weighted_f1 = test["classification_report"].get("weighted avg", {}).get("f1-score")
    cm = test.get("cm")
    balanced_accuracy = test.get("balanced_accuracy")
    if balanced_accuracy is None and isinstance(cm, list) and label_order:
        recalls = []
        for idx, row in enumerate(cm):
            denom = sum(row)
            if denom:
                recalls.append(row[idx] / denom)
        if recalls:
            balanced_accuracy = sum(recalls) / len(recalls)
    macro_roc_auc = first_nonempty(test.get("macro_roc_auc_ovr"), test.get("auc_macro_ovr"))
    weighted_roc_auc = first_nonempty(test.get("weighted_roc_auc_ovr"), test.get("auc_weighted_ovr"))
    ovr_auc = first_nonempty(test.get("per_class_auc_ovr"), test.get("ovr_auc"))
    ovr_ap = test.get("ovr_ap")
    macro_ap = first_nonempty(test.get("macro_ap_ovr"), test.get("ap_macro_ovr"))
    if macro_ap is None and isinstance(ovr_ap, dict) and ovr_ap:
        macro_ap = sum(float(value) for value in ovr_ap.values()) / len(ovr_ap)
    report = test.get("classification_report") if isinstance(test.get("classification_report"), dict) else {}
    counts = {}
    raw_counts = first_nonempty(test.get("counts"), test.get("class_counts"))
    if isinstance(raw_counts, dict):
        if all(str(key).isdigit() for key in raw_counts):
            for idx, label in enumerate(label_order):
                if str(idx) in raw_counts:
                    counts[label] = int(raw_counts[str(idx)])
        else:
            for key, value in raw_counts.items():
                counts[normalize_label(key)] = int(value)
    if not counts and report and label_order:
        counts = extract_supports_from_report(report, label_order)
    if not counts and isinstance(cm, list) and label_order:
        counts = {label_order[idx]: int(sum(row)) for idx, row in enumerate(cm)}
    per_class_auc = {}
    if isinstance(ovr_auc, dict):
        if all(str(key).isdigit() for key in ovr_auc):
            for idx, label in enumerate(label_order):
                value = ovr_auc.get(str(idx))
                if value is not None:
                    per_class_auc[label] = value
        else:
            for key, value in ovr_auc.items():
                per_class_auc[normalize_label(key)] = value
    per_class_ap = {}
    if isinstance(ovr_ap, dict):
        if all(str(key).isdigit() for key in ovr_ap):
            for idx, label in enumerate(label_order):
                value = ovr_ap.get(str(idx))
                if value is not None:
                    per_class_ap[label] = value
        else:
            for key, value in ovr_ap.items():
                per_class_ap[normalize_label(key)] = value
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "macro_roc_auc_ovr": macro_roc_auc,
        "weighted_roc_auc_ovr": weighted_roc_auc,
        "macro_ap_ovr": macro_ap,
        "counts": counts,
        "per_class_auc": per_class_auc,
        "per_class_ap": per_class_ap,
    }


def missing_reproducibility_details(metrics: dict[str, Any], split_source: str, label_source: str, is_binary: bool) -> str:
    missing: list[str] = []
    if not nonempty(metrics.get("seed")):
        missing.append("seed_not_saved")
    if not split_source:
        missing.append("split_source_not_saved")
    if is_binary and not nonempty(first_nonempty(metrics.get("threshold"), metrics.get("test", {}).get("thr"))):
        missing.append("threshold_not_saved")
    if not nonempty(metrics.get("best_ckpt")):
        missing.append("best_checkpoint_not_saved")
    if not label_source:
        missing.append("explicit_label_mapping_not_saved")
    if "train_balancing" not in metrics and "use_smote" not in metrics and "data" not in metrics:
        missing.append("train_balancing_details_not_saved")
    return "; ".join(missing)


def build_run_logbook_row(
    binary_or_multiclass: str,
    tracking: dict[str, Any] | None,
    split_summary: dict[str, Any],
    source_metrics_path: str,
    experiment_group: str,
    variant: str,
    hyperparameters: dict[str, Any],
    test_metrics: dict[str, Any],
    main_result: str,
    missing_details: str,
    note: str,
) -> dict[str, Any]:
    goal = ""
    hypothesis = ""
    dataset_and_split = (
        f"10x_512px_0px_overlap; rule={split_summary['rule']}; "
        f"train_wsi={split_summary['train_wsi_count']}; val_wsi={split_summary['val_wsi_count']}; "
        f"test_wsi={split_summary['test_wsi_count']}; test_prefixes={','.join(split_summary['test_prefixes'])}"
    )
    if tracking:
        goal = tracking.get("tracking_model_type", "")
        hypothesis = tracking.get("tracked_success_criterion", "")
    preprocessing = tracking.get("tracked_data_used", "") if tracking else ""
    model_or_method = tracking.get("tracked_method_summary", "") if tracking else ""
    if variant:
        model_or_method = "; ".join(part for part in (model_or_method, f"variant={variant}") if part)
    suggested_next_action = ""
    if missing_details:
        suggested_next_action = "Recover the missing reproducibility details before using this run in cross-model claims."
    elif note:
        suggested_next_action = "Review the note before using this run as a headline comparison."
    return {
        "experiment_title": f"{experiment_group} {variant}".strip(),
        "date": "",
        "goal": goal,
        "hypothesis": hypothesis,
        "dataset_and_split": dataset_and_split,
        "preprocessing": preprocessing,
        "model_or_method": model_or_method,
        "hyperparameters": json_cell(hyperparameters),
        "test_metrics": json_cell(test_metrics),
        "main_result": main_result,
        "interpretation": "",
        "problems_encountered": note,
        "what_changed_from_previous_run": "balanced training variant" if "balance" in source_metrics_path.lower() else "",
        "missing_reproducibility_details": missing_details,
        "suggested_next_action": suggested_next_action,
    }


def build_readme_rows(
    project_root: Path,
    split_summary: dict[str, Any],
    binary_count: int,
    multiclass_count: int,
    generated_at: str,
) -> list[dict[str, Any]]:
    return [
        {"Key": "Purpose", "Value": "Test-set-only summary of EBME398 H&E quality assessment experiments."},
        {"Key": "Generated At", "Value": generated_at},
        {"Key": "Project Root", "Value": str(project_root)},
        {"Key": "Experiments Root", "Value": str(project_root / "source" / "working_dir" / "10x_512px_0px_overlap" / "experiments")},
        {"Key": "Tracking Workbook", "Value": str(project_root / "source" / "Experiment Tracking.xlsx")},
        {"Key": "Split JSON", "Value": split_summary["split_path"]},
        {"Key": "Split Rule", "Value": str(split_summary["rule"])},
        {"Key": "WSI Counts", "Value": f"train={split_summary['train_wsi_count']}, val={split_summary['val_wsi_count']}, test={split_summary['test_wsi_count']}"},
        {"Key": "Test Cohort Note", "Value": "All test WSIs are SR040-derived." if split_summary["test_only_sr040"] else ",".join(split_summary["test_prefixes"])},
        {"Key": "Measurement Source Policy", "Value": "Summary tabs use only saved JSON test metrics. Workbook is taxonomy/provenance only."},
        {"Key": "Label Normalization", "Value": "Report displays tissue_damage even when source files use tissue_damge."},
        {"Key": "Binary Result Rows", "Value": binary_count},
        {"Key": "Multiclass Result Rows", "Value": multiclass_count},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized test-set report tables from saved experiment metrics.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing source/, scripts/, and pyproject.toml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write report TSV/JSON artifacts into. Defaults to analysis/report_exports/<timestamp>.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_root = project_root / "source"
    experiments_root = source_root / "working_dir" / "10x_512px_0px_overlap" / "experiments"
    workbook_path = source_root / "Experiment Tracking.xlsx"
    split_path = source_root / "working_dir" / "10x_512px_0px_overlap" / "splits" / "sr040_seed42_split.json"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "analysis" / "report_exports" / timestamp
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_entries = parse_workbook(workbook_path)
    split_summary = load_split_summary(split_path)

    binary_rows: list[dict[str, Any]] = []
    multiclass_rows: list[dict[str, Any]] = []
    run_logbook_rows: list[dict[str, Any]] = []
    matched_runs: defaultdict[str, list[str]] = defaultdict(list)
    run_notes: defaultdict[str, list[str]] = defaultdict(list)

    metrics_paths = sorted(experiments_root.rglob("*metrics.json"))
    for metrics_path in metrics_paths:
        metrics = read_json(metrics_path)
        rel_metrics_path = relative_to(metrics_path, project_root)
        family = infer_run_family(relative_to(metrics_path, experiments_root), metrics)
        tracking_id = tracking_id_for_run(family["experiment_group"], family["variant"])
        tracking = tracking_entries.get(tracking_id) if tracking_id else None
        split_source = best_split_source(metrics, metrics_path)
        prediction_artifacts = find_prediction_artifacts(metrics, metrics_path, source_root)
        note_parts: list[str] = []
        parent_metrics = sorted(metrics_path.parent.glob("*metrics.json"))
        if len(parent_metrics) > 1:
            note_parts.append("directory contains multiple metrics files")
        if family["experiment_group"] == "G4":
            note_parts.append("fused feature experiment; no planned G4 row exists in the workbook")
        if family["experiment_group"] == "S4":
            note_parts.append("S4 results come from S4_new fused-feature runs")

        test = metrics.get("test", {})
        hyperparameters = top_level_hparams(metrics)

        if family["task"] == "binary":
            binary = derive_binary_metrics(test)
            row = {
                "tracking_id": tracking_id or "",
                "tracking_sheet": tracking["tracking_sheet"] if tracking else "",
                "experiment_group": family["experiment_group"],
                "variant": family["variant"],
                "balanced_variant": "yes" if family["balanced_variant"] else "no",
                "source_metrics_path": rel_metrics_path,
                "source_predictions_available": "yes" if prediction_artifacts else "no",
                "prediction_artifacts": "; ".join(prediction_artifacts),
                "split_source": split_source,
                "test_n": round_cell(test.get("n")),
                "test_base_rate": round_cell(test.get("base_rate")),
                "auc": round_cell(test.get("auc")),
                "ap": round_cell(test.get("ap")),
                "accuracy": round_cell(binary["accuracy"]),
                "f1": round_cell(binary["f1"]),
                "precision_pos": round_cell(binary["precision_pos"]),
                "recall_sensitivity": round_cell(binary["recall_sensitivity"]),
                "specificity": round_cell(binary["specificity"]),
                "threshold": round_cell(first_nonempty(test.get("thr"), metrics.get("threshold"))),
                "tn": round_cell(binary["tn"]),
                "fp": round_cell(binary["fp"]),
                "fn": round_cell(binary["fn"]),
                "tp": round_cell(binary["tp"]),
                "best_ckpt": metrics.get("best_ckpt", ""),
                "notes": "; ".join(note_parts),
            }
            binary_rows.append(row)
            missing_details = missing_reproducibility_details(metrics, split_source, "binary", is_binary=True)
            main_result = f"AUC={round_cell(test.get('auc'))}; AP={round_cell(test.get('ap'))}; F1={round_cell(binary['f1'])}; n={round_cell(test.get('n'))}"
            run_logbook_rows.append(
                build_run_logbook_row(
                    "binary",
                    tracking,
                    split_summary,
                    rel_metrics_path,
                    family["experiment_group"],
                    family["variant"],
                    hyperparameters,
                    row,
                    main_result,
                    missing_details,
                    row["notes"],
                )
            )
        else:
            label_order, label_source = load_label_mapping_for_run(metrics_path, metrics, project_root)
            if not label_order and isinstance(test.get("classes"), list):
                label_order = [str(item) for item in test["classes"]]
            multiclass = derive_multiclass_metrics(test, label_order)
            if label_source and "S1/label_mapping.json" in label_source and family["experiment_group"] == "S4":
                note_parts.append("label order inherited from S1 label_mapping.json")
            row = {
                "tracking_id": tracking_id or "",
                "tracking_sheet": tracking["tracking_sheet"] if tracking else "",
                "experiment_group": family["experiment_group"],
                "variant": family["variant"],
                "balanced_variant": "yes" if family["balanced_variant"] else "no",
                "source_metrics_path": rel_metrics_path,
                "source_predictions_available": "yes" if prediction_artifacts else "no",
                "prediction_artifacts": "; ".join(prediction_artifacts),
                "split_source": split_source,
                "test_n": round_cell(test.get("n")),
                "accuracy": round_cell(multiclass["accuracy"]),
                "macro_f1": round_cell(multiclass["macro_f1"]),
                "weighted_f1": round_cell(multiclass["weighted_f1"]),
                "balanced_accuracy": round_cell(multiclass["balanced_accuracy"]),
                "macro_roc_auc_ovr": round_cell(multiclass["macro_roc_auc_ovr"]),
                "weighted_roc_auc_ovr": round_cell(multiclass["weighted_roc_auc_ovr"]),
                "macro_ap_ovr": round_cell(multiclass["macro_ap_ovr"]),
                "count_clean": round_cell(multiclass["counts"].get("clean")),
                "count_tissue_damage": round_cell(multiclass["counts"].get("tissue_damage")),
                "count_blurry_fold": round_cell(multiclass["counts"].get("blurry+fold")),
                "auc_clean": round_cell(multiclass["per_class_auc"].get("clean")),
                "auc_tissue_damage": round_cell(multiclass["per_class_auc"].get("tissue_damage")),
                "auc_blurry_fold": round_cell(multiclass["per_class_auc"].get("blurry+fold")),
                "ap_clean": round_cell(multiclass["per_class_ap"].get("clean")),
                "ap_tissue_damage": round_cell(multiclass["per_class_ap"].get("tissue_damage")),
                "ap_blurry_fold": round_cell(multiclass["per_class_ap"].get("blurry+fold")),
                "cm_json": json_cell(test.get("cm")),
                "best_ckpt": metrics.get("best_ckpt", ""),
                "notes": "; ".join(note_parts),
            }
            multiclass_rows.append(row)
            missing_details = missing_reproducibility_details(metrics, split_source, label_source, is_binary=False)
            main_result = (
                f"macro_F1={round_cell(multiclass['macro_f1'])}; "
                f"macro_OVR_AUC={round_cell(multiclass['macro_roc_auc_ovr'])}; "
                f"accuracy={round_cell(multiclass['accuracy'])}; n={round_cell(test.get('n'))}"
            )
            run_logbook_rows.append(
                build_run_logbook_row(
                    "multiclass",
                    tracking,
                    split_summary,
                    rel_metrics_path,
                    family["experiment_group"],
                    family["variant"],
                    hyperparameters,
                    row,
                    main_result,
                    missing_details,
                    row["notes"],
                )
            )
        if tracking_id:
            matched_runs[tracking_id].append(rel_metrics_path)
        else:
            unmatched_key = f"UNTRACKED::{family['experiment_group']}"
            matched_runs[unmatched_key].append(rel_metrics_path)
        if note_parts:
            run_notes[tracking_id or f"UNTRACKED::{family['experiment_group']}"].extend(note_parts)

    binary_df = pd.DataFrame(binary_rows).sort_values(by=["auc", "ap"], ascending=[False, False], na_position="last")
    multiclass_df = pd.DataFrame(multiclass_rows).sort_values(by=["macro_f1", "macro_roc_auc_ovr"], ascending=[False, False], na_position="last")
    logbook_df = pd.DataFrame(run_logbook_rows)

    crosswalk_rows: list[dict[str, Any]] = []
    for tracking_id, tracking in tracking_entries.items():
        matched = sorted(matched_runs.get(tracking_id, []))
        discrepancy_note = "; ".join(sorted(set(run_notes.get(tracking_id, []))))
        match_status = "direct_match" if matched else "missing_saved_run"
        if matched and any("balance" in item.lower() or "S4_new" in item or "G4" in item for item in matched):
            match_status = "matched_with_variants"
        if tracking_id == "S4" and matched:
            discrepancy_note = "; ".join(part for part in [discrepancy_note, "matched to S4_new fused-feature result files"] if part)
        crosswalk_rows.append(
            {
                "tracking_sheet": tracking["tracking_sheet"],
                "tracking_id": tracking["tracking_id"],
                "tracking_model_type": tracking["tracking_model_type"],
                "tracked_data_used": tracking["tracked_data_used"],
                "tracked_method_summary": tracking["tracked_method_summary"],
                "matched_run_paths": "; ".join(matched),
                "match_status": match_status,
                "discrepancy_note": discrepancy_note,
            }
        )
    for key, matched in sorted(matched_runs.items()):
        if not key.startswith("UNTRACKED::"):
            continue
        crosswalk_rows.append(
            {
                "tracking_sheet": "",
                "tracking_id": key.replace("UNTRACKED::", ""),
                "tracking_model_type": "",
                "tracked_data_used": "",
                "tracked_method_summary": "",
                "matched_run_paths": "; ".join(sorted(matched)),
                "match_status": "untracked_filesystem_run",
                "discrepancy_note": "; ".join(sorted(set(run_notes.get(key, [])))),
            }
        )

    readme_df = pd.DataFrame(build_readme_rows(project_root, split_summary, len(binary_df), len(multiclass_df), generated_at))
    crosswalk_df = pd.DataFrame(crosswalk_rows)

    readme_df.to_csv(output_dir / "README.tsv", sep="\t", index=False)
    binary_df.to_csv(output_dir / "Binary_Test_Summary.tsv", sep="\t", index=False)
    multiclass_df.to_csv(output_dir / "Multiclass_Test_Summary.tsv", sep="\t", index=False)
    logbook_df.to_csv(output_dir / "Run_Logbook.tsv", sep="\t", index=False)
    crosswalk_df.to_csv(output_dir / "Tracking_Crosswalk.tsv", sep="\t", index=False)

    summary = {
        "generated_at_utc": generated_at,
        "project_root": str(project_root),
        "experiments_root": str(experiments_root),
        "output_dir": str(output_dir),
        "binary_metrics_files": len(binary_df),
        "multiclass_metrics_files": len(multiclass_df),
        "all_metrics_files": len(metrics_paths),
        "split_summary": split_summary,
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
