#!/usr/bin/env python3
import json, os
base = "/userspace/vvn/HMTL_with_calibration/experiments/automl_newfeatures_v2"
datasets = [
    "dataset_42572_santander_transaction_value",
    "dataset_3050_qsar_tid_11",
    "dataset_3277_qsar_tid_10980",
    "dataset_422_topo_2_1",
    "dataset_42570_mercedes_benz_greener_manufacturing",
]
for ds in datasets:
    path = os.path.join(base, ds, "results.json")
    with open(path) as f:
        data = json.load(f)
    name = data["dataset_name"]
    print(f"\n=== {name} (features={data['n_features']}, samples={data['n_samples_total']}) ===")
    print(f"{'Size':>5} {'aux_task':>14} {'RMSE':>12} {'val_score':>12} {'regime':>8} {'aux_policy':>16} {'aleatoric':>12} {'epistemic':>12}")
    for sk in sorted(data["sizes"].keys(), key=int):
        sd = data["sizes"][sk]
        if "per_seed" in sd and "42" in sd["per_seed"]:
            sv = sd["per_seed"]["42"]
            if sv.get("status") == "ok" and "hmtl" in sv and sv["hmtl"]:
                h = sv["hmtl"]
                aux = h.get("resolved_aux_task", "?")
                rmse = h.get("rmse", 0)
                vs = h.get("ensemble_avg_val_score", 0)
                aleat = h.get("mean_aleatoric", 0)
                epist = h.get("mean_epistemic", 0)
                pol = sd.get("adaptive_policy", {})
                if isinstance(pol, dict):
                    regime = pol.get("regime", "?")
                    aux_pol = pol.get("auto_aux_policy", "?")
                else:
                    regime = "?"
                    aux_pol = "?"
                vs_str = "Inf" if vs == float("inf") else f"{vs:.4f}"
                print(f"{sk:>5} {aux:>14} {rmse:>12.4f} {vs_str:>12} {regime:>8} {aux_pol:>16} {aleat:>12.4f} {epist:>12.4f}")
