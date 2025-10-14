import pandas as pd
import numpy as np
from scipy.stats import linregress
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', str(s))]

def read_manual_coeffs(manual_coeffs):
    if manual_coeffs is None:
        return None
    if isinstance(manual_coeffs, dict):
        return manual_coeffs
    p = Path(manual_coeffs)
    if p.exists():
        df = pd.read_csv(p, sep=';')
        return {str(r['Method']): (float(r['Slope']), float(r['Intercept']))
                for _, r in df.iterrows()}
    else:
        raise FileNotFoundError(f"manual_coeffs file not found: {manual_coeffs}")

def choose_basis_method(calc_dir: Path):
    calc_dir = Path(calc_dir)

    combos = []
    for basis_dir in calc_dir.iterdir():
        if not basis_dir.is_dir():
            continue
        basis = basis_dir.name
        for mol_dir in basis_dir.iterdir():
            if not mol_dir.is_dir():
                continue
            for form_dir in mol_dir.iterdir():
                if not form_dir.is_dir():
                    continue
                for method_dir in form_dir.iterdir():
                    if not method_dir.is_dir():
                        continue
                    method = method_dir.name
                    if (basis, method) not in combos:
                        combos.append((basis, method))

    if not combos:
        print("No combinations available Basis/Method")
        return None

    print("\nAvailable combinations (Basis / Method):\n")
    for i, (basis, method) in enumerate(combos, 1):
        print(f"{i:2d}. {basis:15s} | {method:15s}")

    choice = input("\nSelect a combination (number): ")
    try:
        idx = int(choice) - 1
        selected = combos[idx]
        print(f"\nВыбрано: Basis={selected[0]}, Method={selected[1]}")
        return selected
    except (ValueError, IndexError):
        print("Invalid input")
        return None

def make_interactive_html(name_file,
                          analysis_dir,
                          output_dir='.',
                          manual_coeffs=None,
                          html_name=None):
    
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1. Selecting Basis/Method ===
    if analysis_dir:
        selection = choose_basis_method(analysis_dir)
        if selection is None:
            return
        selected_basis, selected_method = selection
    else:
        selected_basis = selected_method = None

    # === Step 2. Reading data ===
    pka_file = analysis_dir / f"pka_{name_file}.csv"
    if not pka_file.exists():
        raise FileNotFoundError(f"pKa file not found: {pka_file}")

    df = pd.read_csv(pka_file, sep=';')
    df['Molecule'] = df['Molecule'].astype(str)
    df['Base_Molecule'] = df['Molecule'].str.split('_').str[0]

    if selected_basis and 'Basis' in df.columns:
        df = df[df['Basis'] == selected_basis]
    if selected_method and 'Method' in df.columns:
        df = df[df['Method'] == selected_method]

    methods = ['pKa_calc']
    manual = read_manual_coeffs(manual_coeffs)

    # === Part A: Subplots per method ===
    agg_rows = []
    for base, group in df.groupby('Base_Molecule'):
        exp_vals = group['pKa (exp)'].dropna()
        exp_pka = exp_vals.iloc[0] if len(exp_vals) > 0 else np.nan
        row = {'Base_Molecule': base, 'pKa (exp)': exp_pka}
        for method in methods:
            vals = group[method].dropna()
            row[method] = vals.min() if len(vals) > 0 else np.nan
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    fig1 = make_subplots(rows=1, cols=len(methods),
                         subplot_titles=methods,
                         shared_yaxes=True)
    calibration_params = {}

    for i, method in enumerate(methods):
        dfm = agg_df.dropna(subset=['pKa (exp)', method]).copy()
        x = dfm['pKa (exp)'].values
        y = dfm[method].values

        if manual and method in manual:
            slope, intercept = manual[method]
        else:
            if len(x) >= 2:
                slope, intercept, _, _, _ = linregress(x, y)
            else:
                slope, intercept = np.nan, np.nan
        calibration_params[method] = (slope, intercept)

        r_squared = np.corrcoef(x, y)[0, 1]**2 if len(x) >= 2 else np.nan

        fig1.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Calculated',
                                  marker=dict(symbol='square', size=8),
                                  hovertemplate='Mol: %{text}<br>Exp: %{x:.2f}<br>Calc: %{y:.2f}',
                                  text=dfm['Base_Molecule']),
                       row=1, col=i+1)

        if not np.isnan(slope):
            calibrated_y = (y - intercept) / slope
            fig1.add_trace(go.Scatter(x=x, y=calibrated_y, mode='markers',
                                      name=f'Calibrated (R²={r_squared:.2f})',
                                      marker=dict(symbol='triangle-up', size=8),
                                      hovertemplate='Mol: %{text}<br>Exp: %{x:.2f}<br>Calib: %{y:.2f}',
                                      text=dfm['Base_Molecule']),
                           row=1, col=i+1)

        mn, mx = (min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1) if len(x) > 0 else (0, 14)
        fig1.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                  line=dict(color='black', dash='dash'), showlegend=False),
                       row=1, col=i+1)
        fig1.update_xaxes(title_text="Experimental pKa", row=1, col=i+1, range=[mn, mx])
        fig1.update_yaxes(title_text="Calculated pKa", row=1, col=i+1, range=[mn, mx])

    fig1.update_layout(height=500, width=800*len(methods),
                       title_text="Comparison of Experimental and Calculated pKa",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                       hovermode='closest')

    # === Part B: Slider plot per molecule ===
    molecules = sorted(df['Base_Molecule'].unique(), key=natural_sort_key)
    traces = []
    traces_per_mol = []

    for mol in molecules:
        group = df[df['Base_Molecule'] == mol].copy()
        forms = group['Molecule'].tolist()
        exp_vals = group['pKa (exp)'].dropna().unique()
        exp_line = float(exp_vals[0]) if len(exp_vals) > 0 else None

        for method in methods:
            y = group[method].tolist()
            trace = go.Scatter(x=forms, y=y, mode='markers+lines', name=method,
                               hovertemplate='Form: %{x}<br>pKa: %{y:.2f}<extra></extra>',
                               visible=(mol == molecules[0]))
            traces.append(trace)

        if exp_line is not None:
            traces.append(go.Scatter(x=[forms[0], forms[-1]], y=[exp_line, exp_line],
                                     mode='lines', line=dict(dash='dash', width=2, color='black'),
                                     name='Experimental', visible=(mol == molecules[0])))
        else:
            traces.append(go.Scatter(x=[None], y=[None], visible=False))

        min_val = group[method].min()
        min_form = group.loc[group[method].idxmin(), 'Molecule']
        traces.append(go.Scatter(x=[min_form], y=[min_val], mode='markers',
                                 marker=dict(size=14, color='red', symbol='star'),
                                 name='Min pKa', visible=(mol == molecules[0])))
        traces_per_mol.append(len(methods) + 2)

    fig2 = go.Figure(traces)
    total_traces = len(traces)
    steps = []
    for idx, mol in enumerate(molecules):
        vis = [False]*total_traces
        offset = idx * traces_per_mol[0]
        for j in range(traces_per_mol[0]):
            vis[offset+j] = True
        steps.append(dict(method="update", label=mol,
                          args=[{"visible": vis}, {"title": f"pKa forms for {mol}"}]))

    fig2.update_layout(sliders=[dict(active=0, pad={"t":50}, steps=steps,
                                     currentvalue={"prefix": "Molecule: "})],
                       title=f"pKa forms (molecule: {molecules[0]})",
                       xaxis_title="Form", yaxis_title="pKa",
                       height=600, width=1000,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # === HTML export ===
    html_path = output_dir / (html_name if html_name else f"pka_{name_file}_interactive.html")
    fig1_div = fig1.to_html(full_html=False, include_plotlyjs=False, div_id="fig1")
    fig2_div = fig2.to_html(full_html=False, include_plotlyjs=False, div_id="fig2")
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>pKa: {name_file}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style> body {{ font-family: Arial, Helvetica, sans-serif; margin: 10px 20px; }} </style>
</head>
<body>
  <h1>pKa interactive visualization: {name_file}</h1>
  <h2>1) pKa_exp/pKa_calc</h2>
  {fig1_div}
  <hr/>
  <h2>2) Molecule forms viewer</h2>
  {fig2_div}
  <hr/>
  <h3>Calibration parameters (y = a * pKa_exp + b):</h3>
  <pre>{calibration_params}</pre>
</body>
</html>
"""
    html_path.write_text(page, encoding='utf-8')
    print(f"\nInteractive HTML saved to: {html_path}")
    print("Calibration parameters (method: (slope, intercept)):")
    for m, (a, b) in calibration_params.items():
        print(f"  {m}: a = {a:.4f}, b = {b:.4f}")