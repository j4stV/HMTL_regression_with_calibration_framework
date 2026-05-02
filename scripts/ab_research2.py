"""Round 2: add Yeo-Johnson and combine-with-residual-off conditions.

Reports RAW-space RMSE/R-AUC (valid cross-condition comparison).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ab_n_bins import prepare_openml_dataset, run_once

COMMON = dict(n_models=3, epochs=60, patience=10)

# (tag, n_bins, use_residual, filter_y_q, log1p, yeo_johnson)
CONDITIONS = [
    ("base_ref_on",     5, True,  0.0,   False, False),  # reproduce base
    ("residual_off",    5, False, 0.0,   False, False),  # confirmed winner
    ("yj_only",         5, True,  0.0,   False, True),
    ("yj_residual_off", 5, False, 0.0,   False, True),
    ("log1p_residual_off", 5, False, 0.0, True, False),
    ("log1p_only",      5, True,  0.0,   True,  False),
]

DATASETS = ["mip2016", "topo_2_1", "abalone"]
SEEDS = [42, 43]


def main() -> None:
    base_cfgs = {ds: prepare_openml_dataset(ds) for ds in DATASETS}

    runs: list[dict] = []
    out = Path("experiments/ab_research2/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        for cond_name, n_bins, use_residual, filter_q, log1p, yj in CONDITIONS:
            for seed in SEEDS:
                tag = f"r2_{ds}_{cond_name}_seed{seed}"
                print(f"\n{'='*80}\n### {tag}\n{'='*80}")
                try:
                    r = run_once(
                        n_bins=n_bins,
                        seed=seed,
                        use_residual=use_residual,
                        filter_y_quantile=filter_q,
                        log1p_target=log1p,
                        yeo_johnson=yj,
                        base_data_cfg=base_cfgs[ds],
                        tag=tag,
                        **COMMON,
                    )
                    r.update({"dataset": ds, "cond": cond_name, "status": "ok"})
                except Exception as e:
                    r = {
                        "dataset": ds, "cond": cond_name, "seed": seed, "status": "fail",
                        "error": str(e)[:200],
                    }
                runs.append(r)
                with open(out, "w") as f:
                    json.dump(runs, f, indent=2)

    print("\n" + "=" * 100 + "\nSUMMARY (raw-space RMSE / R-AUC MSE)\n" + "=" * 100)
    print(f"{'dataset':<10} {'cond':<22} {'seed':>4} {'raw_rmse':>12} {'raw_rauc':>12} {'std_rmse':>9} {'time':>6}")
    for r in runs:
        if r.get("status") == "ok":
            print(f"{r['dataset']:<10} {r['cond']:<22} {r['seed']:>4} {r.get('raw_rmse', float('nan')):>12.4f} {r.get('raw_r_auc_mse', float('nan')):>12.4f} {r.get('test_rmse', float('nan')):>9.4f} {r.get('elapsed_s', 0):>6.1f}")
        else:
            print(f"{r['dataset']:<10} {r['cond']:<22} {r.get('seed','?'):>4} FAILED: {r.get('error','')[:60]}")


if __name__ == "__main__":
    main()
