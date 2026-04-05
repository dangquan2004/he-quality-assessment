from __future__ import annotations

import re
from enum import Enum


class Task(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"


BINARY_LABELS = {0: "clean", 1: "unclean"}
MULTICLASS_LABELS = {0: "clean", 1: "tissue_damage", 2: "blurry+fold"}


def normalize_label(value: object) -> str:
    if value is None:
        raise ValueError("label is None")
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip().lower()
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned in {"tissue_damge", "tissuedamage", "tissue_damage"}:
        return "tissue_damage"
    if cleaned in {
        "blurry+fold",
        "blur+fold",
        "fold+blur",
        "fold+blurry",
        "foldblur",
        "blurfold",
        "fold&blur",
        "blur&fold",
    }:
        return "blurry+fold"
    if cleaned == "clean":
        return "clean"
    return cleaned


def to_binary_label(value: object) -> int:
    return 0 if normalize_label(value) == "clean" else 1


def to_multiclass_label(value: object) -> int:
    normalized = normalize_label(value)
    mapping = {"clean": 0, "tissue_damage": 1, "blurry+fold": 2}
    if normalized not in mapping:
        raise ValueError(f"unsupported multiclass label: {value!r} -> {normalized!r}")
    return mapping[normalized]


def to_task_label(value: object, task: Task | str) -> int:
    task = Task(task)
    if task is Task.BINARY:
        return to_binary_label(value)
    return to_multiclass_label(value)


def task_labels(task: Task | str) -> dict[int, str]:
    task = Task(task)
    if task is Task.BINARY:
        return BINARY_LABELS
    return MULTICLASS_LABELS
