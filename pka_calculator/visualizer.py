import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from pathlib import Path
import math

def visualize_results(analysis_dir, output_dir, name_file, calibration_file=None):
    """
    Visualization of calculated pKa versus experimental values ​​from pka_min_*.csv
    """
    print(f"Visualizing results from {analysis_dir}")

    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pka_file = analysis_dir / f"pka_min_{name_file}.csv"
    try:
        df = pd.read_csv(pka_file, sep=";")
    except FileNotFoundError:
        print(f"Error: File {pka_file} not found")
        return

    required_cols = {"pKa (exp)", "pKa_calc", "Method", "Basis", "Calculation_Form"}
    if not required_cols.issubset(df.columns):
        print(f"The file does not contain the required columns: {required_cols}")
        return

    calibration_params = {}
    if calibration_file is not None:
        calib_path = Path(calibration_file)
        if calib_path.exists():
            calib_df = pd.read_csv(calib_path, sep=";")
            for _, row in calib_df.iterrows():
                key = (row["Method"], row["Basis"], row["Calculation_Form"])
                calibration_params[key] = (row["Slope"], row["Intercept"])
            print(f"Loaded calibration parameters from {calibration_file}")
        else:
            print(f"Calibration file {calibration_file} not found. Will calculate new parameters.")

    combinations = df[["Method", "Basis", "Calculation_Form"]].drop_duplicates()
    colors = {"Calculated": "blue", "Calibrated": "green"}
    markers = {"Calculated": "s", "Calibrated": "^"}

    n_cols = 3
    n_rows = math.ceil(len(combinations) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    fig.suptitle("Comparison of Experimental and Calculated pKa Values", fontsize=16)

    for i, row in combinations.iterrows():
        method, basis, form = row["Method"], row["Basis"], row["Calculation_Form"]
        label_comb = f"{method} | {basis} | {form}"
        ax = axes[i]

        subset = df[(df["Method"] == method) & (df["Basis"] == basis) & (df["Calculation_Form"] == form)]
        subset = subset.dropna(subset=["pKa (exp)", "pKa_calc"])

        x = subset["pKa (exp)"].values
        y = subset["pKa_calc"].values

        ax.scatter(x, y, c=colors["Calculated"], marker=markers["Calculated"],
                   label="Calculated", alpha=0.7, s=80)

        if len(x) > 1:
            if (method, basis, form) not in calibration_params:
                slope, intercept, _, _, _ = linregress(x, y)
                calibration_params[(method, basis, form)] = (slope, intercept)
            else:
                slope, intercept = calibration_params[(method, basis, form)]

            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            calibrated_y = (y - intercept) / slope

            ax.scatter(x, calibrated_y, c=colors["Calibrated"], marker=markers["Calibrated"],
                       label=f"Calibrated (R²={r_squared:.2f})", alpha=0.7, s=80)

        min_val = min(x.min(), y.min()) - 1
        max_val = max(x.max(), y.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        ax.set_xlabel("Experimental pKa", fontsize=12)
        ax.set_ylabel("Calculated pKa", fontsize=12)
        ax.set_title(label_comb, fontsize=10)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(combinations), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / f"pka_min_{name_file}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Bar plot MAE ---
    mae_results = []
    for (method, basis, form), (slope, intercept) in calibration_params.items():
        subset = df[(df["Method"] == method) & (df["Basis"] == basis) & (df["Calculation_Form"] == form)]
        subset = subset.dropna(subset=["pKa (exp)", "pKa_calc"])

        x = subset["pKa (exp)"].values
        y = subset["pKa_calc"].values

        mae_calc = np.mean(np.abs(y - x))
        mae_results.append({"Combination": f"{method}|{basis}|{form}", "Source": "Calculated", "MAE": mae_calc})

        calibrated = (y - intercept) / slope
        mae_calib = np.mean(np.abs(calibrated - x))
        mae_results.append({"Combination": f"{method}|{basis}|{form}", "Source": "Calibrated", "MAE": mae_calib})

    mae_df = pd.DataFrame(mae_results)
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.35
    x_pos = np.arange(len(mae_df["Combination"].unique()))

    for i, source in enumerate(["Calculated", "Calibrated"]):
        values = mae_df[mae_df["Source"] == source]["MAE"]
        ax.bar(x_pos + i * width, values, width, label=source, color=colors[source])

    ax.set_xlabel("Combination (Method|Basis|Form)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_title("Method/Basis/Form Accuracy Comparison", fontsize=14)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(mae_df["Combination"].unique(), rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / f"pka_min_mae_{name_file}.png", dpi=300, bbox_inches="tight")
    plt.close()

    calib_save_path = output_dir / f"calibration_params_{name_file}.csv"
    calib_save_df = pd.DataFrame([
        {"Method": m, "Basis": b, "Calculation_Form": f, "Slope": s, "Intercept": i}
        for (m, b, f), (s, i) in calibration_params.items()
    ])
    calib_save_df.to_csv(calib_save_path, sep=";", index=False)
    print(f"Calibration parameters saved to {calib_save_path}")

    print("\nCalibration parameters (y = a*pKa + b):")
    for combo, (slope, intercept) in calibration_params.items():
        print(f"{combo}: a = {slope:.4f}, b = {intercept:.4f}")

    print(f"Visualization complete! Plots saved to {plot_path}")
