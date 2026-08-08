import os
from typing import Any, Dict, List, Optional

import pandas as pd
from config import Config
from src.log.Logger import Logger


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def normalize_instance_file_key(
    file_path: str, instance_dir: Optional[str] = None
) -> str:
    p = os.path.normpath(_as_str(file_path).strip())
    if not p:
        return ""

    instance_dir = (
        Config.get_nested("instance", "dir") if instance_dir is None else instance_dir
    )
    instance_dir = os.path.normpath(_as_str(instance_dir).strip())

    p2 = os.path.normcase(p)
    instance_dir2 = os.path.normcase(instance_dir)

    if instance_dir2 and p2.startswith(instance_dir2):
        rel = p2[len(instance_dir2) :].lstrip("\\/")
        return os.path.normcase(os.path.join(instance_dir2, rel))

    try:
        if instance_dir2:
            rel = os.path.relpath(p2, start=instance_dir2)
            if not rel.startswith(".."):
                return os.path.normcase(os.path.join(instance_dir2, rel))
    except Exception:
        pass

    return os.path.normcase(p2)


def load_targets_by_file(
    targets_path: str, log: Optional[Logger] = None
) -> Dict[str, Dict[str, Any]]:
    if not targets_path:
        return {}

    targets_path = os.path.normpath(targets_path)
    if not os.path.exists(targets_path):
        if log is not None:
            log.warning(
                f"targets.xlsx não encontrado em: {targets_path}. Prosseguindo sem targets."
            )
        return {}

    df = pd.read_excel(targets_path, engine="openpyxl")
    if "file" not in df.columns:
        raise ValueError("targets.xlsx inválido: coluna 'file' não encontrada")

    instance_dir = Config.get_nested("instance", "dir")

    targets_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        key_raw = _as_str(row.get("file")).strip()
        key = normalize_instance_file_key(key_raw, instance_dir=instance_dir)
        if not key:
            continue

        payload = row.to_dict()
        targets_by_file.setdefault(key, []).append(payload)

    return targets_by_file
