import pandas as pd
from pathlib import Path

def extract_min_pka(analysis_dir, output_dir, name_file):
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_file = analysis_dir / f"pka_{name_file}.csv"
    if not all_file.exists():
        print(f"Error: {all_file} not found")
        return

    df_all = pd.read_csv(all_file, sep=";")

    if "pKa_calc" not in df_all.columns:
        print("Нет столбца 'pKa_calc' в файле")
        return

    if "Base_Molecule" not in df_all.columns:
        df_all["Base_Molecule"] = df_all["Molecule"].astype(str).str.split("_").str[0]

    df_min = (
        df_all.dropna(subset=["pKa_calc"])
              .groupby(["Base_Molecule", "Method", "Basis", "Calculation_Form"], as_index=False)
              .apply(lambda g: g.loc[g["pKa_calc"].idxmin()])
              .reset_index(drop=True)
    )

    df_min = df_min[[
        "Base_Molecule", "Molecule", "Method", "Basis", "Calculation_Form",
        "pKa (exp)", "pKa_calc"
    ]]

    out_file = output_dir / f"pka_min_{name_file}.csv"
    df_min.to_csv(out_file, sep=";", index=False)

    print(f"Сохранена таблица min-pKa в {out_file}")