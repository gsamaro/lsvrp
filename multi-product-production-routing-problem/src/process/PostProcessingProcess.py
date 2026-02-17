import os

import pandas as pd
from config import Config
from constants import ALPHA


class PostProcessingProcess:
    def __init__(self, log, output):
        self.log = log
        self.output = output

    def union_results(self, run_tag=None, build_target=False):
        # Recursively collect all .xlsx files under output (including subfolders)
        excel_paths = []
        for root, _, files in os.walk(self.output):
            for fname in files:
                if (
                    fname.lower().endswith(".xlsx")
                    and not fname.startswith("~$")
                    and "target" not in fname
                    and "union_results" not in fname
                ):
                    excel_paths.append(os.path.join(root, fname))

        if not excel_paths:
            self.log.error("Nenhum arquivo .xlsx encontrado.")
            return None

        frames = []
        for path in excel_paths:
            try:
                df = pd.read_excel(path, engine="openpyxl")
                df["__source_file__"] = os.path.relpath(path, start=self.output)
                frames.append(df)
            except Exception as e:
                # Skip files that cannot be read; could log if needed
                self.log.error(f"Erro ao ler arquivo {path}: {e}")
                continue

        if not frames:
            self.log.error("Nenhum DataFrame lido.")
            return None

        try:
            union_df = pd.concat(frames, ignore_index=True, sort=False)

            # Enrich consolidated results with alpha and targets (if available)
            try:
                union_df["alpha"] = ALPHA[0]
            except Exception:
                union_df["alpha"] = None

            targets_dir = Config.get_nested("postprocessing", "output")
            targets_path = (
                os.path.join(targets_dir, "targets.xlsx") if targets_dir else None
            )

            if targets_path and os.path.exists(targets_path):
                try:
                    targets_df = pd.read_excel(targets_path, engine="openpyxl")
                    required_cols = [
                        "file",
                        "time",
                        "f1_target",
                        "f2_target",
                        "f3_target",
                        "f4_target",
                        "f5_target",
                    ]
                    if all(c in targets_df.columns for c in required_cols):
                        targets_df = targets_df[required_cols].copy()
                        union_df = union_df.merge(
                            targets_df,
                            how="left",
                            on=["file", "time"],
                            suffixes=("", "_target_file"),
                        )
                    else:
                        missing = [
                            c for c in required_cols if c not in targets_df.columns
                        ]
                        self.log.warning(
                            f"targets.xlsx encontrado, mas faltam colunas {missing}. Prosseguindo sem merge de targets."
                        )
                except Exception as e:
                    self.log.error(
                        f"Erro ao carregar/mesclar targets.xlsx ({targets_path}): {e}. Prosseguindo sem targets."
                    )
            else:
                for c in [
                    "f1_target",
                    "f2_target",
                    "f3_target",
                    "f4_target",
                    "f5_target",
                ]:
                    if c not in union_df.columns:
                        union_df[c] = pd.NA

            if not build_target:
                out_name = f"{run_tag}-union_results.xlsx"
            else:
                out_name = "union_results.xlsx"
            out_path = os.path.join(self.output, out_name)
            union_df.to_excel(out_path, index=False, engine="openpyxl")
        except Exception as e:
            self.log.error(f"Erro ao salvar arquivo {out_path}: {e}")
            return None
        return out_path

    def build_target(self, union_results_path=None):
        if union_results_path is None:
            union_results_path = os.path.join(self.output, "union_results.xlsx")
        df = pd.read_excel(union_results_path, engine="openpyxl")
        ideal_solution = pd.pivot_table(
            df,
            index=["file", "time"],
            aggfunc={
                "f1": "min",
                "f2": "min",
                "f3": "min",
                "f4": "min",
                "f5": "min",
            },
        )
        anti_ideal_solution = pd.pivot_table(
            df,
            index=["file", "time"],
            aggfunc={
                "f1": "max",
                "f2": "max",
                "f3": "max",
                "f4": "max",
                "f5": "max",
            },
        )
        joined = ideal_solution.join(
            anti_ideal_solution, lsuffix="_ideal", rsuffix="_nadir"
        )
        joined["f1_target"] = joined["f1_ideal"] + 0.3 * (
            joined["f1_nadir"] - joined["f1_ideal"]
        )
        joined["f2_target"] = joined["f2_ideal"] + 0.3 * (
            joined["f2_nadir"] - joined["f2_ideal"]
        )
        joined["f3_target"] = joined["f3_ideal"] + 0.3 * (
            joined["f3_nadir"] - joined["f3_ideal"]
        )
        joined["f4_target"] = joined["f4_ideal"] + 0.3 * (
            joined["f4_nadir"] - joined["f4_ideal"]
        )

        joined["f5_target"] = joined["f5_ideal"] + 0.3 * (
            joined["f5_nadir"] - joined["f5_ideal"]
        )

        # Save the target values
        output_dir = Config.get_nested("postprocessing", "output")
        os.makedirs(output_dir, exist_ok=True)
        target_output = os.path.join(output_dir, "targets.xlsx")
        joined.reset_index().to_excel(target_output, index=False)
        self.log.info(f"Target values saved to {target_output}")
        return target_output
