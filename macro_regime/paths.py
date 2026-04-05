from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    return package_root() / "data"


def reports_dir() -> Path:
    return package_root() / "reports"


def notebooks_dir() -> Path:
    return package_root() / "notebooks"


def ensure_layout() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    reports_dir().mkdir(parents=True, exist_ok=True)
    notebooks_dir().mkdir(parents=True, exist_ok=True)


def input_file(name: str) -> Path:
    # Backward compatibility: prefer organized /data, then legacy root file.
    new_path = data_dir() / name
    if new_path.exists():
        return new_path
    return package_root() / name
