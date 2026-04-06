#!/usr/bin/env python3
from pathlib import Path
import sys

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from ebme398_artifact_detection.cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
