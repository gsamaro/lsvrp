import os

import pandas as pd
from config import Config
from constants import ALPHA
from openpyxl import load_workbook
from src.helpers.InstanceMetadata import enrich_with_instance_metadata


class PostProcessingProcess:
    def __init__(self, log, output):
        self.log = log
        self.output = output

    @staticmethod
    def _read_worksheet(worksheet, string_columns):
        rows = worksheet.iter_rows(values_only=True)
        headers = next(rows, None)
        if headers is None:
            return pd.DataFrame()

        data = list(rows)
        while data and all(value is None for value in data[-1]):
            data.pop()

        dataframe = pd.DataFrame(data, columns=list(headers))
        for column in string_columns:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype("string")
        return dataframe

    @staticmethod
    def _read_result_workbook(path):
        """Read the result sheet and the optional legacy metadata sheet.

        Older result workbooks do not have ``run_configs``.  Checking the
        workbook's sheet names before reading it keeps that case explicit,
        while errors in the workbook or in a present metadata sheet are not
        mistaken for a legacy workbook.
        """
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            result_sheet_names = [
                sheet_name
                for sheet_name in workbook.sheetnames
                if sheet_name != "run_configs"
            ]
            if not result_sheet_names:
                raise ValueError(
                    f"O workbook {path} nao possui uma aba de resultados; "
                    "apenas 'run_configs' foi encontrada."
                )

            result_sheet_name = result_sheet_names[0]
            result_df = PostProcessingProcess._read_worksheet(
                workbook[result_sheet_name],
                string_columns=("commit_hash", "config_hash"),
            )
            config_df = None
            if "run_configs" in workbook.sheetnames:
                config_df = PostProcessingProcess._read_worksheet(
                    workbook["run_configs"],
                    string_columns=("config_hash", "config_json"),
                )
        finally:
            workbook.close()

        return result_df, config_df

    def union_results(self, run_tag=None, build_target=False, include_targets=True):
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
        excel_paths.sort()

        if not excel_paths:
            if self.log is not None:
                self.log.error("Nenhum arquivo .xlsx encontrado.")
            return None

        if self.log is not None:
            self.log.info(f"Arquivos encontrados: {len(excel_paths)}")
            self.log.debug(
                f"Arquivos selecionados para consolidacao: {excel_paths}"
            )
        frames = []
        config_frames = []
        if self.log is not None:
            self.log.debug("config_frames inicializado vazio.")
        for path in excel_paths:
            if self.log is not None:
                self.log.debug(f"Iniciando leitura do arquivo: {path}")
            df, config_df = self._read_result_workbook(path)
            if self.log is not None:
                self.log.debug(
                    f"Resultado lido de {path}: linhas={len(df)}, "
                    f"colunas={df.columns.tolist()}, "
                    f"run_configs={'presente' if config_df is not None else 'ausente'}"
                )
            df["__source_file__"] = os.path.relpath(path, start=self.output)
            frames.append(df)
            if self.log is not None:
                self.log.debug(f"frames agora contem {len(frames)} DataFrame(s).")

            if config_df is None:
                if self.log is not None:
                    self.log.debug(
                        f"{path} nao possui a aba run_configs; seguindo sem metadados."
                    )
                continue

            if self.log is not None:
                self.log.debug(
                    f"run_configs lido de {path}: linhas={len(config_df)}, "
                    f"colunas={config_df.columns.tolist()}"
                )
            required_config_columns = {"config_hash", "config_json"}
            missing_config_columns = required_config_columns.difference(
                config_df.columns
            )
            if missing_config_columns:
                missing = sorted(missing_config_columns)
                raise ValueError(
                    f"A aba 'run_configs' de {path} nao possui as colunas obrigatorias: "
                    f"{missing}"
                )

            config_frames.append(config_df[sorted(required_config_columns)].copy())
            if self.log is not None:
                self.log.debug(
                    f"config_frames agora contem {len(config_frames)} DataFrame(s)."
                )

        if self.log is not None:
            self.log.info("Fim da leitura dos arquivos.")
            self.log.debug(
                f"Leitura concluida: frames={len(frames)}, "
                f"config_frames={len(config_frames)}"
            )

        if not frames:
            if self.log is not None:
                self.log.error("Nenhum DataFrame lido.")
            return None

        union_df = pd.concat(frames, ignore_index=True, sort=False)
        if self.log is not None:
            self.log.debug(
                f"union_df concatenado: linhas={len(union_df)}, "
                f"colunas={union_df.columns.tolist()}"
            )
        if "config_hash" not in union_df.columns:
            union_df["config_hash"] = pd.NA

        if config_frames:
            run_configs_df = pd.concat(config_frames, ignore_index=True, sort=False)
            if self.log is not None:
                self.log.debug(
                    f"run_configs concatenado antes da limpeza: linhas={len(run_configs_df)}"
                )
            run_configs_df = run_configs_df.dropna(
                subset=["config_hash", "config_json"]
            )
            valid_config_rows = (
                run_configs_df["config_hash"].astype(str).str.strip().ne("")
                & run_configs_df["config_json"].astype(str).str.strip().ne("")
            )
            run_configs_df = run_configs_df.loc[valid_config_rows]
            run_configs_df = (
                run_configs_df.drop_duplicates(subset=["config_hash"], keep="first")
                .sort_values("config_hash")
                .reset_index(drop=True)
            )
            if self.log is not None:
                self.log.debug(
                    f"run_configs apos limpeza/deduplicacao: linhas={len(run_configs_df)}"
                )
        else:
            run_configs_df = pd.DataFrame(columns=["config_hash", "config_json"])
            if self.log is not None:
                self.log.debug("Nenhum run_configs valido foi encontrado.")

        if self.log is not None:
            self.log.info("Arquivos concatenados.")

        if not any(
            col in union_df.columns
            for col in [
                "instancia",
                "clientes",
                "produtos",
                "veiculos",
                "periodos",
                "seeds",
            ]
        ):
            union_df = enrich_with_instance_metadata(union_df, file_col="file")

        for c in [
            "commit_hash",
            "alpha",
            "weight_hash",
            "new_f1_target",
            "new_f2_target",
            "new_f3_target",
            "new_f4_target",
            "new_f5_target",
        ]:
            if c not in union_df.columns:
                union_df[c] = pd.NA

        targets_dir = Config.get_nested("postprocessing", "output")
        targets_path = (
            os.path.join(targets_dir, "targets.xlsx") if targets_dir else None
        )

        if self.log is not None:
            self.log.info("Colunas validadas.")

        if include_targets and targets_path and os.path.exists(targets_path):
            if self.log is not None:
                self.log.info("Construindo targets.")
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
                    missing = [c for c in required_cols if c not in targets_df.columns]
                    if self.log is not None:
                        self.log.warning(
                            f"targets.xlsx encontrado, mas faltam colunas {missing}. Prosseguindo sem merge de targets."
                        )
            except Exception as e:
                if self.log is not None:
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

        if self.log is not None:
            self.log.info("Preparando para salvar.")

        if not build_target:
            out_name = f"{run_tag}-union_results.xlsx"
        else:
            out_name = "union_results.xlsx"
        out_path = os.path.join(self.output, out_name)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            union_df.to_excel(writer, index=False)
            run_configs_df.to_excel(writer, sheet_name="run_configs", index=False)
        if self.log is not None:
            self.log.info("Salvo.")
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
