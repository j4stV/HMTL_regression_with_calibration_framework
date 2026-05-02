"""A/B test: effect of aux n_bins on HMTL regression metrics.

Compares n_bins = 5 (current default) vs larger values on a local dataset.
Uses reduced ensemble/epochs so it finishes quickly on a laptop.

Usage:
  .venv/bin/python scripts/ab_n_bins.py --dataset wine --n_bins 5 20 --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATASET_CONFIGS = {
    "wine": {"data_cfg_base": "configs/data.yaml"},
    "superconductor": {"data_cfg_base": "configs/data_superconductor.yaml"},
}

OPENML_DATASETS = {
    "mip2016": 43071,      # target std ~28600, heavy-tailed - V2 catastrophic
    "topo_2_1": 422,       # V2 +679% relΔRMSE
    "abalone": 42726,      # neutral control (V2 ~0% delta)
}


def write_temp_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _apply_target_transforms(data_cfg: dict, filter_y_quantile: float, log1p_target: bool, tmpdir: Path, yeo_johnson: bool = False) -> dict:
    """Rewrite train/valid/test CSVs with filtered and/or log1p-ed target. Returns new data_cfg."""
    import numpy as np
    import pandas as pd

    paths = data_cfg["paths"]
    target = paths["target"]
    out_dir = tmpdir / "data_transformed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # compute filter thresholds from train only
    tr = pd.read_csv(paths["train_csv"])
    if filter_y_quantile > 0:
        lo = tr[target].quantile(filter_y_quantile)
        hi = tr[target].quantile(1.0 - filter_y_quantile)
    else:
        lo, hi = -np.inf, np.inf

    yj = None
    yj_clip = None
    if yeo_johnson:
        from sklearn.preprocessing import PowerTransformer
        yj = PowerTransformer(method="yeo-johnson", standardize=False)
        y_train = tr[target].values.astype(float).reshape(-1, 1)
        if filter_y_quantile > 0:
            mask = (y_train.ravel() >= lo) & (y_train.ravel() <= hi)
            yj.fit(y_train[mask])
        else:
            yj.fit(y_train)
        # record training-y range in YJ space to clip predictions on inverse and avoid blow-ups
        yt_train = yj.transform(y_train)
        pad = float(yt_train.std()) * 3.0
        yj_clip = (float(yt_train.min()) - pad, float(yt_train.max()) + pad)

    new_paths = {}
    for split in ["train_csv", "valid_csv", "test_csv"]:
        df = pd.read_csv(paths[split]).copy()
        if split == "train_csv" and filter_y_quantile > 0:
            df = df[(df[target] >= lo) & (df[target] <= hi)].reset_index(drop=True)
        if log1p_target:
            y = df[target].values
            df[target] = np.sign(y) * np.log1p(np.abs(y))
        if yj is not None:
            df[target] = yj.transform(df[target].values.astype(float).reshape(-1, 1)).ravel()
        out_path = out_dir / Path(paths[split]).name
        df.to_csv(out_path, index=False)
        new_paths[split] = str(out_path)

    new_cfg = {**data_cfg, "paths": {**paths, **new_paths}}
    # attach inverse-info to the dict but keep it out of the written YAML
    new_cfg.setdefault("_meta", {})
    new_cfg["_meta"]["_inverse_transform"] = {"yj": yj, "yj_clip": yj_clip, "log1p": log1p_target}
    return new_cfg


def _compute_raw_metrics(y_true: "np.ndarray", y_pred: "np.ndarray", inverse_info: dict) -> dict:
    """Compute RMSE/R-AUC in raw (un-transformed) target space."""
    import numpy as np
    yt = y_true.copy().astype(float)
    yp = y_pred.copy().astype(float)
    yj = inverse_info.get("yj")
    if yj is not None:
        clip = inverse_info.get("yj_clip")
        if clip is not None:
            yt = np.clip(yt, *clip)
            yp = np.clip(yp, *clip)
        yt = yj.inverse_transform(yt.reshape(-1, 1)).ravel()
        yp = yj.inverse_transform(yp.reshape(-1, 1)).ravel()
    if inverse_info.get("log1p"):
        # signed log1p inverse: sign(x)*(exp(|x|)-1)
        yt = np.sign(yt) * (np.exp(np.abs(yt)) - 1)
        yp = np.sign(yp) * (np.exp(np.abs(yp)) - 1)
    errs = (yt - yp) ** 2
    rmse = float(np.sqrt(errs.mean()))
    # simple R-AUC MSE: MSE on sorted-by-|residual| cumulative
    order = np.argsort(np.abs(yt - yp))
    cum_mse = np.cumsum(errs[order]) / np.arange(1, len(errs) + 1)
    r_auc_mse = float(np.mean(cum_mse))
    return {"raw_rmse": rmse, "raw_r_auc_mse": r_auc_mse}


def prepare_openml_dataset(key: str, split_seed: int = 42) -> str:
    """Download OpenML dataset, split 80/10/10, write CSVs, return data-yaml path."""
    import openml
    import numpy as np
    import pandas as pd

    did = OPENML_DATASETS[key]
    out_dir = Path("data/ab_openml") / key
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = out_dir / "data.yaml"
    if data_yaml.exists() and all((out_dir / f).exists() for f in ["train.csv", "valid.csv", "test.csv"]):
        return str(data_yaml)

    ds = openml.datasets.get_dataset(did, download_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    # drop rows with NaN target
    mask = ~pd.isna(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    # coerce sparse dtypes
    for c in X.columns:
        if str(X[c].dtype).startswith("Sparse"):
            X[c] = X[c].sparse.to_dense()
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    n = len(X)
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    tr = idx[:n_tr]; va = idx[n_tr:n_tr + n_va]; te = idx[n_tr + n_va:]
    target_col = ds.default_target_attribute
    for name, subset in [("train", tr), ("valid", va), ("test", te)]:
        df = X.iloc[subset].copy()
        df[target_col] = y.iloc[subset].values
        df.to_csv(out_dir / f"{name}.csv", index=False)

    cfg = {
        "paths": {
            "train_csv": str(out_dir / "train.csv"),
            "valid_csv": str(out_dir / "valid.csv"),
            "cal_csv": None,
            "test_csv": str(out_dir / "test.csv"),
            "target": target_col,
        },
        "preprocess": {
            "impute_const": -1.0,
            "use_dynamic_binning": True,
            "quantile_binning": {"enabled": False, "bins": 5},
            "standardize": True,
            "pca": {"enabled": True, "n_components": None},
            "target_standardize": True,
        },
    }
    with open(data_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[prepared] {key}: train={len(tr)}, valid={len(va)}, test={len(te)} -> {data_yaml}")
    return str(data_yaml)


def build_configs(tmpdir: Path, n_bins: int, seed: int, n_models: int, epochs: int, patience: int, base_data_cfg: str, use_residual: bool = True, filter_y_quantile: float = 0.0, log1p_target: bool = False, yeo_johnson: bool = False) -> dict:
    with open(base_data_cfg) as f:
        data = yaml.safe_load(f)
    with open("configs/model_snn.yaml") as f:
        model = yaml.safe_load(f)
    with open("configs/train.yaml") as f:
        train = yaml.safe_load(f)
    with open("configs/ensemble.yaml") as f:
        ens = yaml.safe_load(f)

    model["hmtl"]["n_bins"] = int(n_bins)
    model["hmtl"]["aux_task"] = "contrastive"
    model["encoder"]["residual"] = bool(use_residual)

    inv_info = {}
    if filter_y_quantile > 0 or log1p_target or yeo_johnson:
        data = _apply_target_transforms(data, filter_y_quantile, log1p_target, tmpdir, yeo_johnson=yeo_johnson)
        inv_info = data.pop("_meta", {}).get("_inverse_transform", {})

    train["training"]["seed"] = int(seed)
    train["training"]["epochs"] = int(epochs)
    train["training"]["batch_size"] = 256
    train["training"]["early_stop"]["patience"] = int(patience)
    train["training"]["adversarial"]["enabled"] = False
    train["conformal"]["method"] = "symmetric"
    train["amp"]["enabled"] = False
    train["training"]["amp"]["enabled"] = False

    ens["ensemble"]["n_models"] = int(n_models)

    paths = {}
    for name, cfg in [("data", data), ("model", model), ("train", train), ("ensemble", ens)]:
        p = tmpdir / f"{name}.yaml"
        write_temp_yaml(p, cfg)
        paths[name] = str(p)
    return paths, inv_info


def run_once(n_bins: int, seed: int, n_models: int, epochs: int, patience: int, base_data_cfg: str, tag: str, use_residual: bool = True, filter_y_quantile: float = 0.0, log1p_target: bool = False, yeo_johnson: bool = False) -> dict:
    from scripts.main import run_experiment

    tmpdir = Path("experiments/ab_n_bins") / tag
    paths, inv_info = build_configs(tmpdir, n_bins, seed, n_models, epochs, patience, base_data_cfg, use_residual=use_residual, filter_y_quantile=filter_y_quantile, log1p_target=log1p_target, yeo_johnson=yeo_johnson)
    t0 = time.time()
    res = run_experiment(
        data_config=paths["data"],
        model_config=paths["model"],
        train_config=paths["train"],
        ensemble_config=paths["ensemble"],
    )
    elapsed = time.time() - t0
    m = res["metrics"]
    # raw-space metrics (only meaningful when a target transform is applied — matches base otherwise)
    raw_m = {}
    tr = res.get("test_results")
    if tr is not None and getattr(tr, "y_true", None) is not None:
        raw_m = _compute_raw_metrics(tr.y_true, tr.y_pred, inv_info)
    return {
        "n_bins": n_bins,
        "seed": seed,
        "use_residual": use_residual,
        "filter_y_quantile": filter_y_quantile,
        "log1p_target": log1p_target,
        "yeo_johnson": yeo_johnson,
        "elapsed_s": round(elapsed, 1),
        "test_rmse": m.get("test_rmse"),
        "test_r_auc_mse": m.get("test_r_auc_mse"),
        "test_mean_uncertainty": m.get("test_mean_uncertainty"),
        "val_rmse": m.get("val_rmse"),
        "val_r_auc_mse": m.get("val_r_auc_mse"),
        "test_coverage@90": m.get("test_coverage@90"),
        "test_width@90": m.get("test_width@90"),
        **raw_m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_CONFIGS) + list(OPENML_DATASETS), default="wine")
    ap.add_argument("--n_bins", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--n_models", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="experiments/ab_n_bins/results.json")
    args = ap.parse_args()

    if args.dataset in OPENML_DATASETS:
        base_data_cfg = prepare_openml_dataset(args.dataset)
    else:
        base_data_cfg = DATASET_CONFIGS[args.dataset]["data_cfg_base"]
    runs: list[dict] = []
    for n_bins in args.n_bins:
        for seed in args.seeds:
            tag = f"{args.dataset}_nbins{n_bins}_seed{seed}"
            print(f"\n{'=' * 80}\n### RUN: {tag}\n{'=' * 80}")
            try:
                r = run_once(n_bins, seed, args.n_models, args.epochs, args.patience, base_data_cfg, tag)
                r["dataset"] = args.dataset
                r["status"] = "ok"
            except Exception as e:
                r = {"dataset": args.dataset, "n_bins": n_bins, "seed": seed, "status": "fail", "error": str(e)}
            runs.append(r)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(runs, f, indent=2)
            print(f"\n--> saved {out}")

    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(f"{'dataset':<16} {'n_bins':>6} {'seed':>4} {'test_rmse':>11} {'test_rauc':>11} {'cov@90':>7} {'time_s':>7}")
    for r in runs:
        if r.get("status") == "ok":
            print(f"{r['dataset']:<16} {r['n_bins']:>6} {r['seed']:>4} {r['test_rmse']:>11.4f} {r['test_r_auc_mse']:>11.4f} {r['test_coverage@90']:>7.3f} {r['elapsed_s']:>7.1f}")
        else:
            print(f"{r['dataset']:<16} {r['n_bins']:>6} {r['seed']:>4} FAILED: {r.get('error','')[:60]}")


if __name__ == "__main__":
    main()
