def _union_results(log, output):
    # Recursively collect all .xlsx files under output (including subfolders)
    excel_paths = []
    for root, _, files in os.walk(output):
        for fname in files:
            if fname.lower().endswith(".xlsx") and not fname.startswith("~$"):
                excel_paths.append(os.path.join(root, fname))

    if not excel_paths:
        log.error("Nenhum arquivo .xlsx encontrado.")
        return None

    frames = []
    for path in excel_paths:
        try:
            df = pd.read_excel(path, engine="openpyxl")
            df["__source_file__"] = os.path.relpath(path, start=output)
            frames.append(df)
        except Exception as e:
            # Skip files that cannot be read; could log if needed
            log.error(f"Erro ao ler arquivo {path}: {e}")
            continue

    if not frames:
        log.error("Nenhum DataFrame lido.")
        return None

    try:
        union_df = pd.concat(frames, ignore_index=True, sort=False)
        out_path = os.path.join(output, "union_results.xlsx")
        union_df.to_excel(out_path, index=False, engine="openpyxl")
    except Exception as e:
        log.error(f"Erro ao salvar arquivo {out_path}: {e}")
        return None
    return out_path
