import argparse
from pathlib import Path
from .calculator import calculate_pka
from .processor import process_results
from .analyzer import analyze_results
from .visualizer import visualize_results
from .monitor import monitor_jobs
from .deprotonator import process_deprotonation
from .equilibrator import process_equilibrated
from .interactive import make_interactive_html
from .min_pka import extract_min_pka

def main():
    parser = argparse.ArgumentParser(description='pKa Calculator Tool')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Calculation command
    calc_parser = subparsers.add_parser('calculate', help='Run pKa calculations')
    calc_parser.add_argument('xyz_dir', default='molecules',
                           help='Directory with XYZ files')                        
    calc_parser.add_argument('-b', '--basis', default='def2-TZVPP', 
                           help='Basis set')
    calc_parser.add_argument('-m', '--methods', nargs='+', default=['B3LYP', 'HF', 'PBE0'],
                           help='Calculation methods')
    calc_parser.add_argument('-o', '--output', default='mycalculations',
                           help='Output directory (full)')
    calc_parser.add_argument('-f', '--forms', nargs='+', choices=['neutral', 'deprotonated'],
                           help='Forms to calculate (neutral, deprotonated, or both if not specified)')

    # Monitor command
    mon_parser = subparsers.add_parser('monitor', help='Monitor running jobs')
    mon_parser.add_argument('summary_path', default='mycalculations',
                           help='Path to calculations_summary.txt')
    mon_parser.add_argument('-u', '--user', default='vandyshev',
                           help='Name of user')

    # Deprotonation comand
    depro_parser = subparsers.add_parser('deprotonate', help='Interactive creation of deprotonated molecules')
    depro_parser.add_argument('calc_dir', 
                            help='Directory with calculations (contains basis set directories)')
    depro_parser.add_argument('-o', '--output', default='deprotonated',
                            help='Output directory for deprotonated molecules')

    # Processing command
    proc_parser = subparsers.add_parser('process', help='Process calculation results')
    proc_parser.add_argument('calc_dir', default='analysis',
                           help='Directory with calculations')
    proc_parser.add_argument('-o', '--output', default='results',
                           help='Output directory')
    proc_parser.add_argument('-n', '--name_file', default='basis',
                           help='Name of output file (basis)')

    # Analysis command
    anal_parser = subparsers.add_parser('analyze', help='Analyze results')
    anal_parser.add_argument('results_dir', default='results', 
                           help='Directory with results')
    anal_parser.add_argument('-e', '--experimental', required=True,
                           help='CSV file with experimental pKa values')
    anal_parser.add_argument('-o', '--output', default='analysis',
                           help='Output directory')
    anal_parser.add_argument('-n', '--name_file', default='basis',
                           help='Name of output file (basis)')

    # Visualization command
    vis_parser = subparsers.add_parser('visualize', help='Visualize results')
    vis_parser.add_argument('analysis_dir', default='analysis',
                           help='Directory with analysis results')
    vis_parser.add_argument('-o', '--output', default='plots',
                           help='Output directory')
    vis_parser.add_argument('-n', '--name_file', default='basis',
                           help='Name of output file (basis)')
    vis_parser.add_argument('-f', '--calibration_file', default='None',
                           help='File with parametrs of calibration')

    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full processing pipeline')
    pipeline_parser.add_argument('calc_dir', 
                               help='Directory with calculations')
    pipeline_parser.add_argument('-e', '--experimental', required=True,
                               help='CSV file with experimental pKa values')
    pipeline_parser.add_argument('-o', '--output', default='process',
                               help='Output directory')
    pipeline_parser.add_argument('-n', '--name_file', default='basis',
                               help='Base name for output files')

    # Equilibration command
    eq_parser = subparsers.add_parser('equilibrate', help='Save equilibrated (last-frame) XYZ molecules')
    eq_parser.add_argument('calc_dir',
                           help='Directory with calculations (contains basis set directories)')
    eq_parser.add_argument('-o', '--output', default='equilibrated',
                           help='Output directory for equilibrated molecules')

    # Interactive visualization command
    inter_parser = subparsers.add_parser('interactive', help='Build interactive HTML visualization')
    inter_parser.add_argument('analysis_dir', 
                              help='Directory with analysis results (contains pka_xxx.csv)')
    inter_parser.add_argument('-n', '--name_file', required=True,
                              help='Base name of pKa file (without prefix pka_)')
    inter_parser.add_argument('-o', '--output', default='.',
                              help='Output directory for HTML')
    inter_parser.add_argument('--manual_coeffs', default=None,
                              help='Dict or path to CSV with Method;Slope;Intercept')
    inter_parser.add_argument('--html_name', default=None,
                              help='Name of output HTML file (default auto)')

    # Extract minimal pKa
    minpka_parser = subparsers.add_parser("minpka", help="Extract minimal pKa per Base_Molecule and save to CSV")
    minpka_parser.add_argument("analysis_dir", help="Directory with analysis results (contains pka_all.csv)")
    minpka_parser.add_argument("-o", "--output", default=".", help="Output directory for CSV")
    minpka_parser.add_argument("-n", "--name_file", default="min", help="Name of output file")



    args = parser.parse_args()

    if args.command == 'calculate':
        calculate_pka(args.xyz_dir, args.basis, args.methods, args.output, args.forms)
    elif args.command == 'deprotonate':
        process_deprotonation(args.calc_dir, args.output)
    elif args.command == 'monitor':
        monitor_jobs(args.summary_path, args.user)
    elif args.command == 'process':
        process_results(args.calc_dir, args.output, args.name_file)
    elif args.command == 'analyze':
        analyze_results(args.results_dir, args.experimental, args.output, args.name_file)
    elif args.command == 'visualize':
        visualize_results(args.analysis_dir, args.output, args.name_file, args.calibration_file)
    elif args.command == 'pipeline':
        print("\n=== Processing calculation results ===")
        process_results(args.calc_dir, args.output, args.name_file)
        
        print("\n=== Analyzing results ===")
        analyze_results(args.output, args.experimental, args.output, args.name_file)
        
        # print("\n=== Generating visualizations ===")
        # visualize_results(args.output, args.output, args.name_file)
    
        print("\n=== Extracting minimal pKa values ===")
        extract_min_pka(args.output, args.output, args.name_file)
    
        print("\n=== Generating visualization for minimal pKa ===")
        visualize_results(args.output, args.output, args.name_file)
    
        print("\n=== Building interactive HTML visualization ===")
        make_interactive_html(
            name_file=args.name_file,
            analysis_dir=args.output,
            output_dir=args.output,
            manual_coeffs=None,
            html_name=f"pka_{args.name_file}_interactive.html"
        )
    
        print("\nPipeline completed successfully!")
    elif args.command == 'equilibrate':
        process_equilibrated(args.calc_dir, args.output)
    elif args.command == 'interactive':
        make_interactive_html(name_file=args.name_file,
                              analysis_dir=args.analysis_dir,
                              output_dir=args.output,
                              # manual_coeffs=args.manual_coeffs,
                              html_name=args.html_name)
    elif args.command == "minpka":
        extract_min_pka(args.analysis_dir, args.output, args.name_file)


if __name__ == '__main__':
    main()