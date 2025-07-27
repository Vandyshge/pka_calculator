import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from pathlib import Path

def visualize_results(analysis_dir, output_dir, name_file):
    """Visualize calculated pKa results compared to experimental values"""
    
    print(f"Visualizing results from {analysis_dir}")
    
    # Prepare directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    pka_file = Path(analysis_dir) / f"pka_{name_file}.csv"
    try:
        calculated_df = pd.read_csv(pka_file, sep=';')
    except FileNotFoundError:
        print(f"Error: File {pka_file} not found")
        return
    
    # Get methods from column names (pKa_HF, pKa_B3LYP, etc.)
    methods = [col.replace('pKa_', '') for col in calculated_df.columns if col.startswith('pKa_')]
    colors = {'Calculated': 'blue', 'Calibrated': 'green'}
    markers = {'Calculated': 's', 'Calibrated': '^'}
    
    if not methods:
        print("No calculated pKa data found in the file")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, len(methods), figsize=(8*len(methods), 6))
    if len(methods) == 1:
        axes = [axes]  # Make it iterable for single plot case
    fig.suptitle('Comparison of Experimental and Calculated pKa Values', fontsize=16)

    calibration_params = {}
    
    # Plot each method
    for i, method in enumerate(methods):
        ax = axes[i]

        # Calculated data
        calc_x = calculated_df['pKa (exp)']
        calc_y = calculated_df[f'pKa_{method}']
        
        mask = ~np.isnan(calc_x) & ~np.isnan(calc_y)
        calc_x = calc_x[mask]
        calc_y = calc_y[mask]
        
        ax.scatter(calc_x, calc_y, c=colors['Calculated'], marker=markers['Calculated'], 
                  label='Calculated', alpha=0.7, s=80)
        
        # Calculate calibration
        slope, intercept, _, _, _ = linregress(calc_x, calc_y)
        calibration_params[method] = (slope, intercept)

        # Calculate and display R-squared
        corr_matrix = np.corrcoef(calculated_df['pKa (exp)'], calculated_df[f'pKa_{method}'])
        r_squared = corr_matrix[0,1]**2
        # ax.text(0.05, 0.9, f'R² = {r_squared:.2f}', 
        #        transform=ax.transAxes, fontsize=12)
        
        # Apply calibration
        calibrated_y = (calc_y - intercept) / slope
        ax.scatter(calc_x, calibrated_y, c=colors['Calibrated'], marker=markers['Calibrated'], 
                  label=f'Calibrated (R² = {r_squared:.2f})', alpha=0.7, s=80)
        # Experimental vs Calculated
        
        # Perfect agreement line
        max_val = max(calculated_df['pKa (exp)'].max(), calculated_df[f'pKa_{method}'].max()) + 1
        min_val = min(calculated_df['pKa (exp)'].min(), calculated_df[f'pKa_{method}'].min()) - 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                # , label='Perfect agreement')
        
        # Plot formatting
        ax.set_xlabel('Experimental pKa', fontsize=14)
        ax.set_ylabel('Calculated pKa', fontsize=14)
        ax.set_title(f'{method} Method', fontsize=16)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"pka_{name_file}.png"
    plt.savefig(plot_path, dpi=50, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    mae_results = []
    
    for method in methods:
        # Calculated MAE
        errors = calculated_df[f'pKa_{method}'] - calculated_df['pKa (exp)']
        mae_calc = np.mean(np.abs(errors))
        mae_results.append({'Method': method, 'Source': 'Calculated', 'MAE': mae_calc})
        
        # Calibrated MAE
        slope, intercept = calibration_params[method]
        calibrated = (calculated_df[f'pKa_{method}'] - intercept) / slope
        errors_calib = calibrated - calculated_df['pKa (exp)']
        mae_calib = np.mean(np.abs(errors_calib))
        mae_results.append({'Method': method, 'Source': 'Calibrated', 'MAE': mae_calib})
    
    mae_df = pd.DataFrame(mae_results)
    
    # Bar plot
    width = 0.25
    x = np.arange(len(methods))
    
    for i, source in enumerate(['Calculated', 'Calibrated']):
        values = mae_df[mae_df['Source'] == source]['MAE']
        ax.bar(x + i*width, values, width, label=source, color=colors[source])
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('Method Accuracy Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'pka_mae_{name_file}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print calibration parameters
    print("\nCalibration parameters (y = a*pKa + b):")
    for method, (slope, intercept) in calibration_params.items():
        print(f"{method}: a = {slope:.4f}, b = {intercept:.4f}")
    
    print(f"Visualization complete! Plot saved to {plot_path}")