from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis.metrics import metrics_table
from .analysis.calibration import apply_linear_calibration, fit_linear_calibration
from .analysis.plots import plot_predictions
from .chemistry.rdkit_tools import smiles_csv_to_manifest
from .config import write_default_config
from .manifest import infer_legacy_xyz_manifest, write_manifest
from .schedulers.slurm import query_slurm, submit_batches
from .thermodynamics.core import (
    calibrate_reference,
    load_experimental,
    predict_acid_base,
    save_reference_table,
)
from .workflow import (
    collect_project_results,
    prepare_project,
    project_progress,
    rebuild_project_jobs,
)
from .geometry import extract_final_geometries
from .analysis.interactive import make_interactive_html


def _load_results(path_or_project: str | Path) -> pd.DataFrame:
    path = Path(path_or_project).expanduser().resolve()
    if path.is_dir():
        path = path / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def _temperature_from_project(path_or_project: str | Path, default: float = 298.15) -> float:
    path = Path(path_or_project).expanduser().resolve()
    project = path if path.is_dir() else path.parent
    cfg = project / "config.toml"
    if cfg.exists():
        from .config import ProjectConfig

        return ProjectConfig.from_toml(cfg).thermodynamics.temperature_k
    return default


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_default_config(root / "pka.toml", overwrite=args.force)
    manifest = root / "molecules.csv"
    if args.force or not manifest.exists():
        pd.DataFrame(
            columns=[
                "molecule_id",
                "state_id",
                "state",
                "xyz",
                "charge",
                "multiplicity",
                "site_id",
                "parent_id",
                "smiles",
                "metadata",
            ]
        ).to_csv(manifest, index=False)
    print(f"Initialized: {root}")
    print(f"  config:   {root / 'pka.toml'}")
    print(f"  manifest: {manifest}")


def cmd_import_xyz(args: argparse.Namespace) -> None:
    states = infer_legacy_xyz_manifest(args.xyz_dir, multiplicity=args.multiplicity)
    out = Path(args.output).expanduser().resolve()
    write_manifest(states, out)
    print(f"Imported {len(states)} states -> {out}")
    print("WARNING: multiplicity was not inferred chemically. Review the manifest before calculations.")


def cmd_generate_smiles(args: argparse.Namespace) -> None:
    out = smiles_csv_to_manifest(
        args.csv,
        args.output_dir,
        enumerate_states=not args.neutral_only,
        tautomers=args.tautomers,
        max_tautomers=args.max_tautomers,
    )
    print(f"Generated manifest: {out}")


def cmd_prepare(args: argparse.Namespace) -> None:
    specs, batches = prepare_project(
        args.manifest,
        args.config,
        args.project_dir,
        overwrite=args.overwrite,
    )
    print(f"Prepared calculations: {len(specs)}")
    print(f"Prepared grouped Slurm jobs: {len(batches)}")
    if len(batches):
        cols = [
            "batch_id",
            "n_calculations",
            "cores_per_calculation",
            "worker_slots",
            "allocated_cores",
        ]
        print(batches[cols].to_string(index=False))


def cmd_rebuild_jobs(args: argparse.Namespace) -> None:
    batches = rebuild_project_jobs(args.project_dir)
    print(f"Rebuilt worker-pool Slurm jobs: {len(batches)}")
    if len(batches):
        cols = [
            "batch_id",
            "n_calculations",
            "cores_per_calculation",
            "worker_slots",
            "allocated_cores",
        ]
        print(batches[cols].to_string(index=False))


def cmd_submit(args: argparse.Namespace) -> None:
    df = submit_batches(args.project_dir, skip_submitted=not args.resubmit)
    if "array_job_id" in df and df["array_job_id"].fillna("").ne("").any():
        arrays = df[df["array_job_id"].fillna("").ne("")]
        summary = arrays.groupby("array_job_id", sort=False).agg(
            batches=("batch_id", "size"),
            calculations=("n_calculations", lambda x: x.astype(int).sum()),
        )
        print("Slurm arrays:")
        print(summary.to_string())
        return
    cols = [
        "batch_id",
        "n_calculations",
        "cores_per_calculation",
        "worker_slots",
        "allocated_cores",
        "job_id",
    ]
    print(df[cols].to_string(index=False))


def cmd_status(args: argparse.Namespace) -> None:
    print("Slurm batches:")
    try:
        print(query_slurm(args.project_dir).to_string(index=False))
    except Exception as exc:
        print(f"  Slurm query unavailable: {exc}")
    progress = project_progress(args.project_dir)
    print("\nCalculation progress:")
    if progress.empty:
        print("  no calculations")
    else:
        print(progress["status"].value_counts().rename_axis("status").to_string())


def cmd_collect(args: argparse.Namespace) -> None:
    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("PKA_COLLECT_WORKERS", "1"))
    df = collect_project_results(args.project_dir, workers=workers)
    print(f"Collected {len(df)} calculations -> {Path(args.project_dir).resolve() / 'results.csv'}")
    if "valid" in df.columns:
        print(df["valid"].value_counts(dropna=False).rename_axis("valid").to_string())


def cmd_calibrate(args: argparse.Namespace) -> None:
    results = _load_results(args.results)
    exp = load_experimental(
        args.experimental,
        value_column=args.value_column,
        property_hint=("pkb" if args.route == "pkb" else "pka"),
    )
    temperature = args.temperature or _temperature_from_project(args.results)
    refs, details = calibrate_reference(
        results,
        exp,
        route=args.route,
        temperature_k=temperature,
        estimator=args.estimator,
    )
    out = Path(args.output).expanduser().resolve()
    save_reference_table(refs, out)
    details_path = out.with_name(out.stem + "_training_details.csv")
    details.to_csv(details_path, index=False)
    print(refs.to_string(index=False))
    print(f"Reference table: {out}")
    print(f"Training details: {details_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    results = _load_results(args.results)
    refs = pd.read_csv(args.references)
    temperature = args.temperature or _temperature_from_project(args.results)
    pred = predict_acid_base(
        results,
        refs,
        route=args.route,
        temperature_k=temperature,
        pkw=args.pkw,
    )
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"Predictions: {out}")
    selected = pred[pred["selected"]] if "selected" in pred.columns else pred
    cols = [x for x in ["molecule_id", "level", "state_id", "site_id", "property", "value"] if x in selected]
    print(selected[cols].to_string(index=False))


def cmd_metrics(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.predictions)
    prop_hint = str(pred["property"].dropna().iloc[0]).lower() if "property" in pred and not pred["property"].dropna().empty else None
    exp = load_experimental(args.experimental, value_column=args.value_column, property_hint=prop_hint)
    metrics = metrics_table(pred, exp, selected_only=not args.all_sites)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(metrics.to_string(index=False))
    if args.fit_calibration:
        fit = fit_linear_calibration(pred, exp, selected_only=not args.all_sites)
        fit_path = out.with_name(out.stem + "_linear_calibration.csv")
        fit.to_csv(fit_path, index=False)
        print(f"Linear calibration (experimental = a*calculated + b): {fit_path}")


def cmd_linear_calibrate(args: argparse.Namespace) -> None:
    """Fit on explicit train data and apply frozen coefficients to all predictions."""
    pred = pd.read_csv(args.predictions)
    prop_hint = (
        str(pred["property"].dropna().iloc[0]).lower()
        if "property" in pred and not pred["property"].dropna().empty
        else None
    )
    exp = load_experimental(
        args.training_experimental,
        value_column=args.value_column,
        property_hint=prop_hint,
    )
    fit = fit_linear_calibration(
        pred,
        exp,
        selected_only=not args.all_sites,
    )
    corrected = apply_linear_calibration(pred, fit)

    fit_path = Path(args.output_calibration).expanduser().resolve()
    pred_path = Path(args.output_predictions).expanduser().resolve()
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    fit.to_csv(fit_path, index=False)
    corrected.to_csv(pred_path, index=False)
    print(fit.to_string(index=False))
    print(f"Calibration: {fit_path}")
    print(f"Corrected predictions: {pred_path}")


def cmd_plot(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.predictions)
    prop_hint = str(pred["property"].dropna().iloc[0]).lower() if "property" in pred and not pred["property"].dropna().empty else None
    exp = load_experimental(args.experimental, value_column=args.value_column, property_hint=prop_hint)
    paths = plot_predictions(pred, exp, args.output_dir, selected_only=not args.all_sites)
    for path in paths:
        print(path)



def cmd_extract_geometries(args: argparse.Namespace) -> None:
    df = extract_final_geometries(
        args.project_dir,
        args.output_dir,
        level=args.level,
        valid_only=not args.include_invalid,
    )
    print(f"Extracted geometries: {len(df)} -> {Path(args.output_dir).resolve()}")


def cmd_interactive(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.predictions)
    prop_hint = str(pred["property"].dropna().iloc[0]).lower() if "property" in pred and not pred["property"].dropna().empty else None
    exp = load_experimental(args.experimental, value_column=args.value_column, property_hint=prop_hint)
    out = make_interactive_html(
        pred, exp, args.output, selected_only=not args.all_sites
    )
    print(out)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pka-calculator",
        description="Reproducible pKa/pKb quantum-chemistry workflows with ORCA and Slurm",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a starter config and empty molecule manifest")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("import-xyz", help="migrate the legacy XYZ filename convention")
    p.add_argument("xyz_dir")
    p.add_argument("-o", "--output", default="molecules.csv")
    p.add_argument("--multiplicity", type=int, default=1)
    p.set_defaults(func=cmd_import_xyz)

    p = sub.add_parser("generate-smiles", help="generate 3D acid/base candidate states with RDKit")
    p.add_argument("csv", help="CSV with columns name,smiles")
    p.add_argument("output_dir")
    p.add_argument("--neutral-only", action="store_true")
    p.add_argument("--tautomers", action="store_true")
    p.add_argument("--max-tautomers", type=int, default=20)
    p.set_defaults(func=cmd_generate_smiles)

    p = sub.add_parser("prepare", help="prepare ORCA inputs and grouped Slurm jobs")
    p.add_argument("manifest")
    p.add_argument("-c", "--config", default="pka.toml")
    p.add_argument("-o", "--project-dir", default="calculations")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "rebuild-jobs",
        help="rebuild only Slurm worker-pool jobs for an existing project",
    )
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_rebuild_jobs)

    p = sub.add_parser("submit", help="submit prepared grouped Slurm jobs")
    p.add_argument("project_dir")
    p.add_argument("--resubmit", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="show Slurm batches and per-calculation progress")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="parse and validate ORCA results")
    p.add_argument("project_dir")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel ORCA output parsers (default: PKA_COLLECT_WORKERS or 1)")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("calibrate", help="fit an effective thermodynamic reference on a training set")
    p.add_argument("results", help="project directory or results.csv")
    p.add_argument("experimental")
    p.add_argument("--route", choices=["acid", "conjugate_acid", "pkb"], default="acid")
    p.add_argument("--value-column")
    p.add_argument("--temperature", type=float)
    p.add_argument("--estimator", choices=["mean", "median"], default="mean")
    p.add_argument("-o", "--output", default="references.csv")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("predict", help="calculate site-specific and selected pKa/pKb")
    p.add_argument("results", help="project directory or results.csv")
    p.add_argument("references")
    p.add_argument(
        "--route",
        choices=["acid", "conjugate_acid", "pkb", "pkb_from_pka"],
        default="acid",
    )
    p.add_argument("--temperature", type=float)
    p.add_argument("--pkw", type=float, default=14.0)
    p.add_argument("-o", "--output", default="predictions.csv")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("metrics", help="compare predictions with an independent experimental set")
    p.add_argument("predictions")
    p.add_argument("experimental")
    p.add_argument("--value-column")
    p.add_argument("--all-sites", action="store_true")
    p.add_argument("--fit-calibration", action="store_true")
    p.add_argument("-o", "--output", default="metrics.csv")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser(
        "linear-calibrate",
        help="fit experimental = a*calculated+b on train data and apply to predictions",
    )
    p.add_argument("predictions")
    p.add_argument("training_experimental")
    p.add_argument("--value-column")
    p.add_argument("--all-sites", action="store_true")
    p.add_argument("--output-calibration", default="linear_calibration.csv")
    p.add_argument("--output-predictions", default="predictions_linear_calibrated.csv")
    p.set_defaults(func=cmd_linear_calibrate)

    p = sub.add_parser("plot", help="plot experimental vs calculated values")
    p.add_argument("predictions")
    p.add_argument("experimental")
    p.add_argument("output_dir")
    p.add_argument("--value-column")
    p.add_argument("--all-sites", action="store_true")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("extract-geometries", help="extract final optimized XYZ structures")
    p.add_argument("project_dir")
    p.add_argument("output_dir")
    p.add_argument("--level")
    p.add_argument("--include-invalid", action="store_true")
    p.set_defaults(func=cmd_extract_geometries)

    p = sub.add_parser("interactive", help="create an interactive Plotly HTML report")
    p.add_argument("predictions")
    p.add_argument("experimental")
    p.add_argument("-o", "--output", default="pka_report.html")
    p.add_argument("--value-column")
    p.add_argument("--all-sites", action="store_true")
    p.set_defaults(func=cmd_interactive)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
