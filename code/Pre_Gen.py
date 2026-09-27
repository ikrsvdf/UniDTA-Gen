# -*- coding: utf-8 -*-
# for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
#   echo "========== ${fold} =========="

#   "$PY" Pre_Gen.py \
#     --dataset bindingdb \
#     --start-set warm_start \
#     --fold "$fold" \
#     --stage generate \
#     --only-tau 1.2 \
#     --only-lambda 2.0 \
#     --device cuda:2
# done
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import pickle
import sys
import traceback
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils import DataSet, collate_fn, Tokenizer  # noqa: E402
from model import DrugGenModel, KA_GAT, create_mask_from_tensor  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

import __main__ as _main_module  # noqa: E402
if not hasattr(_main_module, "Tokenizer"):
    setattr(_main_module, "Tokenizer", Tokenizer)

RDLogger.DisableLog("rdApp.*")


PROJECT_ROOT = "/media/ubuntu/disk_1/lcg/UniDTA-Gen"

DATASET = "bindingdb"
START_SET = "warm_start"
FOLD = "fold_0"
TEST_CSV = ""
WEIGHT_PATH = ""
SENSITIVITY_DIR = ""
OUTPUT_ROOT = ""
MOLECULE_DIR = ""
EVAL_DIR = ""

TAU_VALUES = [0.5, 0.7, 0.9, 1.2]
LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0]

DEFAULT_TAU = 1.2
DEFAULT_LAMBDA = 5.0


def _resolve_results_fold_dir(dataset: str, start_set: str, fold: str) -> str:
    """Support both old and current result layouts.

    Old:     code/results/<dataset>/<start_set>/<fold>/
    Current: code/<dataset>/<start_set>/<fold>/
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "code", "results", dataset, start_set, fold),
        os.path.join(PROJECT_ROOT, "code", dataset, start_set, fold),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def configure_fold(
    fold: str,
    dataset: str = "parasite",
    start_set: str = "warm_start",
) -> None:
    """Set all fold-specific input / output paths."""
    global DATASET, START_SET, FOLD
    global TEST_CSV, WEIGHT_PATH, SENSITIVITY_DIR
    global OUTPUT_ROOT, MOLECULE_DIR, EVAL_DIR

    DATASET = dataset
    START_SET = start_set
    FOLD = fold

    TEST_CSV = os.path.join(
        PROJECT_ROOT, dataset, "data_folds", start_set, fold, "test.csv"
    )
    results_fold_dir = _resolve_results_fold_dir(dataset, start_set, fold)
    WEIGHT_PATH = os.path.join(results_fold_dir, "best_Gen_model_total.pth")
    SENSITIVITY_DIR = os.path.join(results_fold_dir, "sensitivity")
    OUTPUT_ROOT = os.path.join(SENSITIVITY_DIR, "lambda_tau_4x4_grid")
    MOLECULE_DIR = os.path.join(OUTPUT_ROOT, "generated_molecules")
    EVAL_DIR = os.path.join(OUTPUT_ROOT, "evaluation")


# Initialise the default fold_0 paths at import time.
configure_fold(FOLD)


def configure_cells(only_tau=None, only_lambda=None) -> None:
    """Limit the grid to one (tau, lambda) cell if requested."""
    global TAU_VALUES, LAMBDA_VALUES

    if (only_tau is None) != (only_lambda is None):
        raise SystemExit("--only-tau and --only-lambda must be used together.")

    if only_tau is not None:
        TAU_VALUES = [float(only_tau)]
        LAMBDA_VALUES = [float(only_lambda)]
    else:
        TAU_VALUES = [0.5, 0.7, 0.9, 1.2]
        LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0]

# Metrics saved/evaluated for each grid cell.
RATIO_METRICS = [
    "validity_ratio",
    "uniqueness_ratio",
    "novelty_ratio",
    "ratio_of_available_molecules",
]
PROPERTY_METRICS = ["mean_QED", "mean_LogP", "mean_SAS"]
ALL_METRICS = RATIO_METRICS + PROPERTY_METRICS

METRIC_LABELS = {
    "validity_ratio": "Validity",
    "uniqueness_ratio": "Uniqueness",
    "novelty_ratio": "Novelty",
    "ratio_of_available_molecules": "Available molecules",
    "mean_QED": "Mean QED",
    "mean_LogP": "Mean LogP",
    "mean_SAS": "Mean SAS",
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def _fmt_value(value: float) -> str:
    """Format 0.5/1.0/2.0/5.0 nicely for file names and labels."""
    return f"{float(value):.1f}"


def _cell_path(tau: float, lam: float) -> str:
    return os.path.join(
        MOLECULE_DIR, f"tau_{_fmt_value(tau)}_lambda_{_fmt_value(lam)}.csv"
    )


def format_smiles(smiles: str):
    """Canonicalize a SMILES string; return None if RDKit cannot parse it."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string; fall back to CPU if CUDA is unavailable."""
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except Exception as exc:
        raise SystemExit(f"Invalid --device {device_str!r}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4x4 lambda-tau sensitivity analysis for UniDTA-Gen."
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "evaluate", "all"],
        default="all",
        help="generate CSV files, evaluate them, or both (default: all).",
    )
    parser.add_argument(
        "--dataset",
        default="parasite",
        help="Dataset name (default: parasite).",
    )
    parser.add_argument(
        "--start-set",
        default="warm_start",
        help="Split folder under data_folds (default: warm_start).",
    )
    parser.add_argument(
        "--fold",
        default="fold_0",
        help="Fold name, e.g. fold_0 ... fold_4 (default: fold_0).",
    )
    parser.add_argument(
        "--device",
        default="cuda:2",
        help="Torch device, e.g. cuda:2, cuda:0, cpu (default: cuda:2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Base random seed.  Each newly generated cell uses seed + cell_index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite CSV files that already exist in the output folder.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Do not generate heatmap PNG files; CSV summaries are still saved.",
    )
    parser.add_argument(
        "--only-tau",
        type=float,
        default=None,
        help="Generate/evaluate only one tau value (single-cell mode). "
        "Must be used together with --only-lambda.",
    )
    parser.add_argument(
        "--only-lambda",
        type=float,
        default=None,
        help="Generate/evaluate only one lambda value (single-cell mode). "
        "Must be used together with --only-tau.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
def load_tokenizer():
    tokenizer_path = os.path.join(PROJECT_ROOT, DATASET, f"{DATASET}_tokenizer.pkl")
    with open(tokenizer_path, "rb") as handle:
        return pickle.load(handle)


def build_dataset_and_loader(tokenizer):
    """Build the test DataSet/DataLoader in the same order as Pre_Gen.py."""
    test_dataset = DataSet(
        TEST_CSV,
        os.path.join(PROJECT_ROOT, DATASET, "esm2"),
        os.path.join(PROJECT_ROOT, DATASET, "3di_embeddings"),
        tokenizer=tokenizer,
        grapg_dir=os.path.join(PROJECT_ROOT, DATASET, "saved_graphs"),
        dataset=DATASET,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return test_dataset, test_loader


def build_model(tokenizer, device: torch.device):
    hidden_dim = 128
    out_1 = 32
    out_2 = 16
    grid_size = 2
    head = 2
    layer_num = 2
    pooling = "avg"
    in_node_dim = 92
    in_edge_dim = 21

    ka_gat_model = KA_GAT(
        in_node_dim,
        in_edge_dim,
        hidden_dim,
        out_1,
        out_2,
        grid_size,
        head,
        layer_num,
        pooling,
    ).to(device)

    model = DrugGenModel(tokenizer=tokenizer, ka_gat_model=ka_gat_model).to(device)

    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    # Be tolerant of DataParallel checkpoints that contain a "module." prefix.
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if any(str(key).startswith("module.") for key in state_dict.keys()):
            state_dict = OrderedDict(
                (str(key).replace("module.", "", 1), value)
                for key, value in state_dict.items()
            )
            model.load_state_dict(state_dict)
        else:
            raise exc

    model.eval()
    model.tokenizer = tokenizer
    print(f"Loaded generator weights: {WEIGHT_PATH}")
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_one_cell(
    model,
    test_loader,
    output_base: pd.DataFrame,
    tau: float,
    lam: float,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run generation for one (tau, lambda) cell and return a CSV-ready frame."""
    set_seed(seed)
    model.eval()
    all_generated = []

    pbar = tqdm(
        test_loader,
        desc=f"generate tau={_fmt_value(tau)} lambda={_fmt_value(lam)}",
        leave=False,
    )
    with torch.no_grad():
        for batch in pbar:
            try:
                smiles_seq = batch["smiles_seq"].to(device)
                affinity = batch["affinity"].to(device)
                esm = batch["esm_feat"].to(device)
                di = batch["3di_feat"].to(device)
                qed = batch["qed"].to(device)
                sas = batch["sas"].to(device)
                logp = batch["logp"].to(device)
                batched_graph = batch["graph"].to(device)

                mask = create_mask_from_tensor(esm)
                esm = esm.float()
                di = di.float()

                # model.generate() prints a "Constrained generation succeeded"
                # line for every molecule; suppress it so the tqdm progress bar
                # remains readable.
                with contextlib.redirect_stdout(io.StringIO()):
                    generated_tokens, retry_count = model.generate(
                        smiles_seq,
                        esm,
                        di,
                        mask,
                        affinity,
                        qed,
                        sas,
                        logp,
                        random_sample=True,
                        max_retry=10,
                        require_valid=True,
                        graphs=batched_graph,
                        apply_chemical_constraints=True,
                        chem_check_interval=3,
                        temperature=float(tau),
                        top_k=50,
                        top_p=0.99,
                        lambda_chem=float(lam),
                    )

                generated = model.tokenizer.get_text(generated_tokens)
                # Same post-processing as Pre_Gen.py: canonicalize and drop
                # molecules that are not RDKit-parseable.
                generated = [format_smiles(smi) for smi in generated]
                generated = [smi for smi in generated if smi]

                all_generated.append(";".join(generated) if generated else "")
                pbar.set_postfix(
                    {
                        "retry": retry_count,
                        "valid": f"{len(generated)}/{len(generated_tokens)}",
                    }
                )
            except Exception:
                traceback.print_exc()
                all_generated.append("")
                continue

    if len(all_generated) != len(output_base):
        raise RuntimeError(
            "Generated row count does not match the DataSet row count: "
            f"{len(all_generated)} vs {len(output_base)}. "
            "The DataSet may have dropped rows without graphs."
        )

    output_df = output_base.copy()
    output_df["generated_molecules"] = all_generated
    return output_df


def run_generation(args: argparse.Namespace) -> None:
    os.makedirs(MOLECULE_DIR, exist_ok=True)

    reference_df = pd.read_csv(TEST_CSV)
    reference_row_count = len(reference_df)
    plan_rows = []
    cells_to_generate = []

    for tau in TAU_VALUES:
        for lam in LAMBDA_VALUES:
            out_path = _cell_path(tau, lam)
            action = ""
            source = ""

            if os.path.exists(out_path) and not args.overwrite:
                action = "skipped_existing_output"
            else:
                action = "generated"
                source = WEIGHT_PATH
                cells_to_generate.append((float(tau), float(lam), out_path))

            plan_rows.append(
                {
                    "tau": float(tau),
                    "lambda": float(lam),
                    "file": out_path,
                    "action": action,
                    "source": source,
                }
            )

    if cells_to_generate:
        print(
            f"\n{len(cells_to_generate)} cell(s) need generation. "
            f"{16 - len(cells_to_generate)} cell(s) will be skipped."
        )
        tokenizer = load_tokenizer()
        test_dataset, test_loader = build_dataset_and_loader(tokenizer)
        output_base = test_dataset.data.reset_index(drop=True).copy()

        if len(output_base) != reference_row_count:
            print(
                f"[warn] DataSet length ({len(output_base)}) != test.csv length "
                f"({reference_row_count}); generated CSVs will contain only the "
                "samples with usable graphs."
            )

        device = resolve_device(args.device)
        model = build_model(tokenizer, device)

        for cell_index, (tau, lam, out_path) in enumerate(cells_to_generate):
            print(
                f"\n========== [{cell_index + 1}/{len(cells_to_generate)}] "
                f"tau={_fmt_value(tau)}  lambda={_fmt_value(lam)} =========="
            )
            generated_df = generate_one_cell(
                model=model,
                test_loader=test_loader,
                output_base=output_base,
                tau=tau,
                lam=lam,
                seed=args.seed + cell_index,
                device=device,
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            generated_df.to_csv(out_path, index=False)
            print(f"Saved -> {out_path}")
    else:
        print("\nNo cell needs generation (all requested CSVs already exist).")

    plan_df = pd.DataFrame(plan_rows).sort_values(["tau", "lambda"]).reset_index(drop=True)
    plan_path = os.path.join(OUTPUT_ROOT, "grid_plan.csv")
    plan_df.to_csv(plan_path, index=False)
    print(f"\nGrid plan saved -> {plan_path}")


# ---------------------------------------------------------------------------
# Evaluation: validity / uniqueness / novelty / properties
# ---------------------------------------------------------------------------
def is_valid_smiles(smiles: str) -> bool:
    if smiles is None:
        return False
    text = str(smiles).strip()
    if text == "" or text.lower() == "nan":
        return False
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        return False
    return mol is not None and mol.GetNumAtoms() > 0


def load_property_calculator():
    """Load data_process/add_properties.py to keep QED/LogP/SAS definitions."""
    ap_path = os.path.join(PROJECT_ROOT, "data_process", "add_properties.py")
    fp_path = os.path.join(PROJECT_ROOT, "data_process", "fpscores.pkl.gz")
    if not (os.path.exists(ap_path) and os.path.exists(fp_path)):
        print("[warn] add_properties.py / fpscores.pkl.gz not found; "
              "property metrics will be left as NaN.")
        return None, None

    try:
        spec = importlib.util.spec_from_file_location("add_properties_mod", ap_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sa_calculator = module.SAScoreCalculator(fp_path)
        return module, sa_calculator
    except Exception as exc:
        print(f"[warn] Failed to load property calculator: {exc}")
        return None, None


def evaluate_file(path: str, ap_module=None, sa_calculator=None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "generated_molecules" not in df.columns:
        raise ValueError(f"'generated_molecules' not found in {path}")

    # Keep the same convention as sensitivity_evaluation.py: one CSV cell =
    # one generated molecule.  This script generates with batch_size=1, so no
    # semicolon-joined molecules appear.
    generated = (
        df["generated_molecules"].fillna("").astype(str).str.strip().tolist()
    )

    reference_set = set(df["Drug"].fillna("").astype(str).str.strip()) if "Drug" in df.columns else set()
    reference_set = {s for s in reference_set if s}

    valid = [s for s in generated if is_valid_smiles(s)]
    valid_unique = set(valid)
    novel = [s for s in valid_unique if s not in reference_set]
    novel_set = set(novel)

    total_generated = len(generated)
    validity_ratio = len(valid) / total_generated if total_generated else 0.0
    uniqueness_ratio = len(valid_unique) / len(valid) if valid else 0.0
    novelty_ratio = len(novel_set) / len(valid_unique) if valid_unique else 0.0
    ratio_available = len(novel_set) / total_generated if total_generated else 0.0

    result = {
        "total_generated": total_generated,
        "valid_count": len(valid),
        "unique_count": len(valid_unique),
        "novel_count": len(novel_set),
        "validity_ratio": validity_ratio,
        "uniqueness_ratio": uniqueness_ratio,
        "novelty_ratio": novelty_ratio,
        "ratio_of_available_molecules": ratio_available,
    }

    # Property means over valid generated molecules.
    for metric in PROPERTY_METRICS:
        result[metric] = float("nan")
    if ap_module is not None and sa_calculator is not None and valid:
        rows = []
        for smi in valid:
            try:
                qed, logp, sas = ap_module.compute_chem_props(smi, sa_calculator)
            except Exception:
                qed = logp = sas = float("nan")
            rows.append((qed, logp, sas))
        prop_df = pd.DataFrame(rows, columns=["QED", "LogP", "SAS"]).dropna(how="all")
        if len(prop_df):
            result["props_count"] = int(len(prop_df))
            result["mean_QED"] = float(prop_df["QED"].mean())
            result["mean_LogP"] = float(prop_df["LogP"].mean())
            result["mean_SAS"] = float(prop_df["SAS"].mean())
        else:
            result["props_count"] = 0
    else:
        result["props_count"] = 0

    return result


def pivot_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        summary.pivot(index="tau", columns="lambda", values=metric)
        .reindex(index=TAU_VALUES, columns=LAMBDA_VALUES)
    )


def save_heatmap(
    summary: pd.DataFrame,
    metric: str,
    out_path: str,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
) -> None:
    matrix = pivot_metric(summary, metric)
    values = matrix.values.astype(float)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(LAMBDA_VALUES)))
    ax.set_xticklabels([_fmt_value(v) for v in LAMBDA_VALUES])
    ax.set_yticks(np.arange(len(TAU_VALUES)))
    ax.set_yticklabels([_fmt_value(v) for v in TAU_VALUES])
    ax.set_xlabel("lambda (chemical penalty weight)")
    ax.set_ylabel("tau (temperature coefficient)")
    ax.set_title(METRIC_LABELS.get(metric, metric))

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if np.isnan(value) else f"{value:.3f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if (not np.isnan(value) and value > (vmax or 0.5) * 0.6) else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap -> {out_path}")


def run_evaluation(skip_plots: bool = False) -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)
    ap_module, sa_calculator = load_property_calculator()

    rows = []
    missing = []
    for tau in TAU_VALUES:
        for lam in LAMBDA_VALUES:
            csv_path = _cell_path(tau, lam)
            if not os.path.exists(csv_path):
                missing.append(csv_path)
                continue
            metrics = evaluate_file(csv_path, ap_module, sa_calculator)
            rows.append(
                {
                    "tau": float(tau),
                    "lambda": float(lam),
                    "file": csv_path,
                    **metrics,
                }
            )

    if missing:
        print("[warn] Missing generated CSVs, skipped:")
        for path in missing:
            print(f"  {path}")

    if not rows:
        raise SystemExit("No generated CSV files found to evaluate.")

    summary = pd.DataFrame(rows).sort_values(["tau", "lambda"]).reset_index(drop=True)
    preferred_cols = [
        "tau",
        "lambda",
        "total_generated",
        "valid_count",
        "unique_count",
        "novel_count",
        "validity_ratio",
        "uniqueness_ratio",
        "novelty_ratio",
        "ratio_of_available_molecules",
        "props_count",
        "mean_QED",
        "mean_LogP",
        "mean_SAS",
        "file",
    ]
    summary = summary[[c for c in preferred_cols if c in summary.columns]]

    summary_path = os.path.join(EVAL_DIR, "grid_evaluation_results.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nEvaluation summary saved -> {summary_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Heatmaps for all ratio and property metrics.
    if not skip_plots:
        for metric in RATIO_METRICS:
            out_path = os.path.join(EVAL_DIR, f"heatmap_{metric}.png")
            save_heatmap(
                summary,
                metric,
                out_path,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )

        for metric in PROPERTY_METRICS:
            if metric in summary.columns and summary[metric].notna().any():
                out_path = os.path.join(EVAL_DIR, f"heatmap_{metric}.png")
                save_heatmap(summary, metric, out_path, cmap="magma")
    else:
        print("Skipped heatmap generation because --no-plots was given.")

    # A small run-info file for reproducibility.
    info_path = os.path.join(OUTPUT_ROOT, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write("UniDTA-Gen lambda-tau 4x4 sensitivity analysis\n")
        handle.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"test_csv: {TEST_CSV}\n")
        handle.write(f"weights: {WEIGHT_PATH}\n")
        handle.write(f"tau_values: {TAU_VALUES}\n")
        handle.write(f"lambda_values: {LAMBDA_VALUES}\n")
        handle.write(f"default_tau: {DEFAULT_TAU}\n")
        handle.write(f"default_lambda: {DEFAULT_LAMBDA}\n")
    print(f"Run info saved -> {info_path}")


def main() -> None:
    args = parse_args()
    configure_fold(args.fold, args.dataset, args.start_set)
    configure_cells(args.only_tau, args.only_lambda)

    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"Test CSV not found: {TEST_CSV}")
    if not os.path.exists(WEIGHT_PATH):
        raise SystemExit(f"Generator weight not found: {WEIGHT_PATH}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(MOLECULE_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # The DataSet class uses project-root-relative graph/embedding paths.
    os.chdir(PROJECT_ROOT)

    print(f"Dataset  : {DATASET}")
    print(f"Start set: {START_SET}")
    print(f"Fold     : {FOLD}")
    print(f"Test CSV : {TEST_CSV}")
    print(f"Weights  : {WEIGHT_PATH}")
    print(f"Output   : {OUTPUT_ROOT}")

    if args.stage in ("generate", "all"):
        run_generation(args)
    if args.stage in ("evaluate", "all"):
        run_evaluation(skip_plots=args.no_plots)


if __name__ == "__main__":
    main()
