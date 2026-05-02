#!/usr/bin/env python3
"""Diagnostic script for worst-performing HMTL datasets in AutoML benchmark."""

import json
import os
import sys

BASE = "/userspace/vvn/HMTL_with_calibration/experiments/automl_newfeatures_v2"

DATASETS = [
    "dataset_42572_santander_transaction_value",
    "dataset_3050_qsar_tid_11",
    "dataset_3277_qsar_tid_10980",
    "dataset_422_topo_2_1",
    "dataset_42570_mercedes_benz_greener_manufacturing",
]

def fmt(v, width=14):
    """Format a value for display."""
    if v is None:
        return "N/A".rjust(width)
    if isinstance(v, float):
        if v == float("inf") or v == float("-inf"):
            return "Inf".rjust(width)
        if abs(v) > 1e6:
            return f"{v:.2e}".rjust(width)
        if abs(v) > 100:
            return f"{v:.2f}".rjust(width)
        if abs(v) > 1:
            return f"{v:.4f}".rjust(width)
        return f"{v:.6f}".rjust(width)
    return str(v).rjust(width)


def analyze_dataset(path):
    with open(path) as f:
        data = json.load(f)

    name = data["dataset_name"]
    n_features = data["n_features"]
    n_total = data["n_samples_total"]
    n_train = data["n_samples_train"]
    n_test = data["n_samples_test"]
    seeds = data["run_meta"]["seed_list"]

    print("=" * 130)
    print(f"DATASET: {name}")
    print(f"  Samples: total={n_total}, train={n_train}, test={n_test}  |  Features: {n_features}")
    print(f"  Seeds: {seeds}")
    print("=" * 130)

    sizes_data = data["sizes"]
    size_keys = sorted(sizes_data.keys(), key=lambda x: int(x))

    # =====================================================
    # 1. PER-SIZE BREAKDOWN: HMTL RMSE + baselines
    # =====================================================
    print("\n--- 1. PER-SIZE RMSE BREAKDOWN (HMTL vs Baselines) ---")
    header = f"{'Size%':>6} {'N_train':>8} | {'HMTL_RMSE':>14} {'CatBoost_RMSE':>14} {'MLP_RMSE':>14} | {'HMTL/CB ratio':>14}"
    print(header)
    print("-" * len(header))

    for sk in size_keys:
        sd = sizes_data[sk]
        n_tr = sd.get("n_train_samples", "?")

        # Get aggregated HMTL RMSE
        hmtl_rmse = None
        cb_rmse = None
        mlp_rmse = None

        # Try aggregated first
        if "aggregate_over_seeds" in sd and sd["aggregate_over_seeds"]:
            agg = sd["aggregate_over_seeds"]
            if "hmtl" in agg and agg["hmtl"]:
                hmtl_rmse = agg["hmtl"].get("rmse_mean")
            if "baselines" in agg and agg["baselines"]:
                if "catboost" in agg["baselines"]:
                    cb_rmse = agg["baselines"]["catboost"].get("rmse_mean")
                if "single_mlp" in agg["baselines"]:
                    mlp_rmse = agg["baselines"]["single_mlp"].get("rmse_mean")

        # Fallback: compute from per_seed
        if hmtl_rmse is None and "per_seed" in sd:
            vals = []
            for seed_key, sv in sd["per_seed"].items():
                if sv.get("status") == "ok" and "hmtl" in sv and sv["hmtl"]:
                    r = sv["hmtl"].get("rmse")
                    if r is not None:
                        vals.append(r)
            if vals:
                hmtl_rmse = sum(vals) / len(vals)

        if cb_rmse is None and "per_seed" in sd:
            vals = []
            for seed_key, sv in sd["per_seed"].items():
                if sv.get("status") == "ok" and "baselines" in sv:
                    bl = sv["baselines"]
                    if "catboost" in bl:
                        r = bl["catboost"].get("rmse")
                        if r is not None:
                            vals.append(r)
            if vals:
                cb_rmse = sum(vals) / len(vals)

        if mlp_rmse is None and "per_seed" in sd:
            vals = []
            for seed_key, sv in sd["per_seed"].items():
                if sv.get("status") == "ok" and "baselines" in sv:
                    bl = sv["baselines"]
                    if "single_mlp" in bl:
                        r = bl["single_mlp"].get("rmse")
                        if r is not None:
                            vals.append(r)
            if vals:
                mlp_rmse = sum(vals) / len(vals)

        ratio = ""
        if hmtl_rmse is not None and cb_rmse is not None and cb_rmse > 0:
            ratio = f"{hmtl_rmse / cb_rmse:.1f}x"

        print(f"{sk:>6} {n_tr:>8} | {fmt(hmtl_rmse)} {fmt(cb_rmse)} {fmt(mlp_rmse)} | {ratio:>14}")

    # =====================================================
    # 2. PER-SEED VARIANCE
    # =====================================================
    print("\n--- 2. PER-SEED VARIANCE (HMTL RMSE by seed) ---")
    header2 = f"{'Size%':>6} | "
    for s in seeds:
        header2 += f"{'seed=' + str(s) + '_RMSE':>18} "
    header2 += f"| {'diff%':>10} {'worse_seed':>12}"
    print(header2)
    print("-" * len(header2))

    for sk in size_keys:
        sd = sizes_data[sk]
        line = f"{sk:>6} | "
        seed_rmse = {}
        for s in seeds:
            s_str = str(s)
            if "per_seed" in sd and s_str in sd["per_seed"]:
                sv = sd["per_seed"][s_str]
                if sv.get("status") == "ok" and "hmtl" in sv and sv["hmtl"]:
                    r = sv["hmtl"].get("rmse")
                    seed_rmse[s] = r
                    line += f"{fmt(r, 18)} "
                else:
                    status = sv.get("status", "?")
                    line += f"{'[' + status + ']':>18} "
            else:
                line += f"{'[missing]':>18} "

        if len(seed_rmse) == 2:
            vals = list(seed_rmse.values())
            if min(vals) > 0:
                diff_pct = abs(vals[0] - vals[1]) / min(vals) * 100
                worse = max(seed_rmse, key=seed_rmse.get)
                line += f"| {diff_pct:>9.1f}% {worse:>12}"
            else:
                line += f"| {'N/A':>10} {'N/A':>12}"
        else:
            line += f"| {'N/A':>10} {'N/A':>12}"

        print(line)

    # =====================================================
    # 3. TRAINING METRICS (val score, val r_auc_mse)
    # =====================================================
    print("\n--- 3. TRAINING METRICS (ensemble_avg_val_score & ensemble_avg_val_r_auc_mse) ---")
    header3 = f"{'Size%':>6} | "
    for s in seeds:
        header3 += f"{'s' + str(s) + '_val_score':>18} {'s' + str(s) + '_val_rauc':>18} "
    print(header3)
    print("-" * len(header3))

    for sk in size_keys:
        sd = sizes_data[sk]
        line = f"{sk:>6} | "
        for s in seeds:
            s_str = str(s)
            if "per_seed" in sd and s_str in sd["per_seed"]:
                sv = sd["per_seed"][s_str]
                if sv.get("status") == "ok" and "hmtl" in sv and sv["hmtl"]:
                    val_score = sv["hmtl"].get("ensemble_avg_val_score")
                    val_rauc = sv["hmtl"].get("ensemble_avg_val_r_auc_mse")
                    line += f"{fmt(val_score, 18)} {fmt(val_rauc, 18)} "
                else:
                    line += f"{'[skipped]':>18} {'[skipped]':>18} "
            else:
                line += f"{'[missing]':>18} {'[missing]':>18} "
        print(line)

    # =====================================================
    # 4. UNCERTAINTY VALUES
    # =====================================================
    print("\n--- 4. UNCERTAINTY VALUES (mean per seed, averaged over seeds) ---")
    header4 = f"{'Size%':>6} | {'mean_uncert':>14} {'mean_epist':>14} {'mean_aleat':>14} {'epist/aleat':>12} | {'CB_uncert':>14} {'CB_epist':>14} {'CB_aleat':>14}"
    print(header4)
    print("-" * len(header4))

    for sk in size_keys:
        sd = sizes_data[sk]
        h_uncerts = []
        h_epists = []
        h_aleats = []
        cb_uncerts = []
        cb_epists = []
        cb_aleats = []

        if "per_seed" in sd:
            for s_str, sv in sd["per_seed"].items():
                if sv.get("status") != "ok":
                    continue
                if "hmtl" in sv and sv["hmtl"]:
                    h = sv["hmtl"]
                    if h.get("mean_uncertainty") is not None:
                        h_uncerts.append(h["mean_uncertainty"])
                    if h.get("mean_epistemic") is not None:
                        h_epists.append(h["mean_epistemic"])
                    if h.get("mean_aleatoric") is not None:
                        h_aleats.append(h["mean_aleatoric"])
                if "baselines" in sv and "catboost" in sv["baselines"]:
                    cb = sv["baselines"]["catboost"]
                    if cb.get("mean_uncertainty") is not None:
                        cb_uncerts.append(cb["mean_uncertainty"])
                    if cb.get("mean_epistemic") is not None:
                        cb_epists.append(cb["mean_epistemic"])
                    if cb.get("mean_aleatoric") is not None:
                        cb_aleats.append(cb["mean_aleatoric"])

        avg = lambda lst: sum(lst) / len(lst) if lst else None
        h_u = avg(h_uncerts)
        h_e = avg(h_epists)
        h_a = avg(h_aleats)
        cb_u = avg(cb_uncerts)
        cb_e = avg(cb_epists)
        cb_a = avg(cb_aleats)

        ratio = ""
        if h_e is not None and h_a is not None and h_a > 1e-9:
            ratio = f"{h_e / h_a:.1f}x"

        print(f"{sk:>6} | {fmt(h_u)} {fmt(h_e)} {fmt(h_a)} {ratio:>12} | {fmt(cb_u)} {fmt(cb_e)} {fmt(cb_a)}")

    # =====================================================
    # 5. SUMMARY FLAGS
    # =====================================================
    print("\n--- 5. DIAGNOSTIC FLAGS ---")

    # Check for Infinity val scores
    inf_count = 0
    total_count = 0
    all_hmtl_rmse = []
    all_cb_rmse = []

    for sk in size_keys:
        sd = sizes_data[sk]
        if "per_seed" not in sd:
            continue
        for s_str, sv in sd["per_seed"].items():
            if sv.get("status") != "ok":
                continue
            total_count += 1
            if "hmtl" in sv and sv["hmtl"]:
                h = sv["hmtl"]
                vs = h.get("ensemble_avg_val_score")
                if vs is not None and (vs == float("inf") or vs == float("-inf")):
                    inf_count += 1
                r = h.get("rmse")
                if r is not None:
                    all_hmtl_rmse.append(r)
            if "baselines" in sv and "catboost" in sv["baselines"]:
                r = sv["baselines"]["catboost"].get("rmse")
                if r is not None:
                    all_cb_rmse.append(r)

    print(f"  Infinity val_score count: {inf_count}/{total_count}")

    if all_hmtl_rmse and all_cb_rmse:
        hmtl_median = sorted(all_hmtl_rmse)[len(all_hmtl_rmse) // 2]
        cb_median = sorted(all_cb_rmse)[len(all_cb_rmse) // 2]
        print(f"  HMTL RMSE  - min: {min(all_hmtl_rmse):.4f}, median: {hmtl_median:.4f}, max: {max(all_hmtl_rmse):.4f}")
        print(f"  CatBoost RMSE - min: {min(all_cb_rmse):.6f}, median: {cb_median:.6f}, max: {max(all_cb_rmse):.6f}")
        if cb_median > 0:
            print(f"  Median HMTL/CatBoost ratio: {hmtl_median / cb_median:.1f}x")

    # Check if epistemic >> aleatoric everywhere
    epist_dom_count = 0
    for sk in size_keys:
        sd = sizes_data[sk]
        if "per_seed" not in sd:
            continue
        for s_str, sv in sd["per_seed"].items():
            if sv.get("status") != "ok" or "hmtl" not in sv or not sv["hmtl"]:
                continue
            h = sv["hmtl"]
            e = h.get("mean_epistemic", 0)
            a = h.get("mean_aleatoric", 0)
            if a > 0 and e / a > 10:
                epist_dom_count += 1
    print(f"  Epistemic >> Aleatoric (>10x): {epist_dom_count}/{total_count}")

    # Check adaptive policy
    print("\n  Adaptive policies used:")
    for sk in size_keys:
        sd = sizes_data[sk]
        policy = sd.get("adaptive_policy")
        eff_cfg = sd.get("effective_config")
        if policy or eff_cfg:
            cfg_str = ""
            if eff_cfg:
                cfg_str = f" | hidden={eff_cfg.get('hidden_width','?')}, low={eff_cfg.get('low_layer','?')}, high={eff_cfg.get('high_layer','?')}, n_models={eff_cfg.get('n_models','?')}, bagging={eff_cfg.get('bagging','?')}"
            print(f"    Size {sk}%: policy={policy}{cfg_str}")

    print()


def main():
    for ds in DATASETS:
        path = os.path.join(BASE, ds, "results.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        try:
            analyze_dataset(path)
        except Exception as e:
            print(f"ERROR analyzing {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
