"""Multi-hypothesis A/B: test a matrix of tweaks vs current defaults.

Hypotheses (after n_bins confirmed):
  H1: use_residual=False (matches Bondarenko reference)
  H2: filter top/bot 0.1% of y on train (matches Bondarenko filter_dataset)
  H3: log1p(target) for heavy-tailed targets
  H4: combine n_bins=20 + use_residual=False (stacking best)
  H5: all combined (n_bins=20 + residual=False + filter 0.1%)

Datasets:
  - mip2016 (heavy-tailed, catastrophic case)
  - topo_2_1 (bounded target, neutral)
  - abalone (small, neutral control)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ab_n_bins import prepare_openml_dataset, run_once

# Fast local config
COMMON = dict(n_models=3, epochs=60, patience=10)

# (tag_suffix, n_bins, use_residual, filter_y_quantile, log1p_target)
CONDITIONS = [
    ("base",                 5,  True,  0.0,   False),  # current defaults (reproduce)
    ("nbins20",              20, True,  0.0,   False),  # hypothesis 0 (confirmed)
    ("residual_off",         5,  False, 0.0,   False),  # H1
    ("filter001",            5,  True,  0.001, False),  # H2
    ("log1p",                5,  True,  0.0,   True),   # H3
    ("combined_20_res",      20, False, 0.0,   False),  # H4
    ("combined_all",         20, False, 0.001, False),  # H5
]

DATASETS = ["mip2016", "topo_2_1", "abalone"]
SEEDS = [42, 43]


def main() -> None:
    base_cfgs = {ds: prepare_openml_dataset(ds) for ds in DATASETS}

    runs: list[dict] = []
    out = Path("experiments/ab_research/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        for cond_name, n_bins, use_residual, filter_q, log1p in CONDITIONS:
            # Skip log1p on datasets where target has negatives that span near zero
            # topo_2_1 is in [0,1] and abalone is positive; log1p is safe on all three
            for seed in SEEDS:
                tag = f"res_{ds}_{cond_name}_seed{seed}"
                print(f"\n{'='*80}\n### {tag}\n{'='*80}")
                try:
                    r = run_once(
                        n_bins=n_bins,
                        seed=seed,
                        use_residual=use_residual,
                        filter_y_quantile=filter_q,
                        log1p_target=log1p,
                        base_data_cfg=base_cfgs[ds],
                        tag=tag,
                        **COMMON,
                    )
                    r.update({"dataset": ds, "cond": cond_name, "status": "ok"})
                except Exception as e:
                    r = {
                        "dataset": ds, "cond": cond_name, "seed": seed,
                        "n_bins": n_bins, "use_residual": use_residual,
                        "filter_y_quantile": filter_q, "log1p_target": log1p,
                        "status": "fail", "error": str(e)[:200],
                    }
                runs.append(r)
                with open(out, "w") as f:
                    json.dump(runs, f, indent=2)

    # summary
    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    hdr = f"{'dataset':<10} {'cond':<20} {'seed':>4} {'rmse':>9} {'r_auc':>9} {'cov@90':>7} {'time':>6}"
    print(hdr)
    for r in runs:
        if r.get("status") == "ok":
            print(f"{r['dataset']:<10} {r['cond']:<20} {r['seed']:>4} {r['test_rmse']:>9.4f} {r['test_r_auc_mse']:>9.4f} {r['test_coverage@90']:>7.3f} {r['elapsed_s']:>6.1f}")
        else:
            print(f"{r['dataset']:<10} {r['cond']:<20} {r.get('seed','?'):>4} FAILED: {r.get('error','')[:60]}")


if __name__ == "__main__":
    main()
