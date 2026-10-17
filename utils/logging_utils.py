"""Lightweight structured logging for research experiments."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_dir: Path, level=logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
