#!/usr/bin/env python3
"""Generate a consolidated results workbook without executing the optimizer."""

import argparse
import subprocess
import sys
from datetime import datetime
from hashlib import sha1
from pathlib import Path

# Allow execution as ``python scripts/generate_union_results.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.log.Logger import Logger
from src.process.PostProcessingProcess import PostProcessingProcess


def _get_git_commit_hash6():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:6]
    except Exception:
        return "000000"


def _build_run_tag(now: datetime):
    date_part = now.strftime("%Y-%m-%d")
    commit_part = _get_git_commit_hash6()
    ts_full = now.astimezone().isoformat()
    ts_part = sha1(ts_full.encode("utf-8")).hexdigest()[:6]
    return f"{date_part}-{commit_part}-{ts_part}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Consolida planilhas de resultados sem executar a otimizacao."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Pasta que contem os resultados .xlsx a serem consolidados.",
    )
    args = parser.parse_args(argv)

    output = Path(args.input_dir).expanduser().resolve()
    if not output.is_dir():
        print(f"Pasta de entrada inexistente ou invalida: {output}", file=sys.stderr)
        return 1

    log = Logger(
        log_dir=str(output / "logs"),
        log_file="generate_union_results.log",
        worker_id=0,
        task=0,
    )
    run_tag = _build_run_tag(datetime.now())
    union_path = PostProcessingProcess(log=log, output=str(output)).union_results(
        run_tag=run_tag,
        build_target=False,
        include_targets=False,
    )
    if union_path is None:
        log.error("Falha ao gerar union_results: nenhuma planilha Excel valida foi consolidada.")
        return 1

    log.info(f"Union results salvo em {union_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
