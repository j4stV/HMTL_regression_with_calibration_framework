#!/usr/bin/env python3
"""
Comprehensive analysis: HMTL vs CatBoost across 11 datasets.
Investigates why HMTL wins on some datasets but loses on most.
"""
import json
import sys
from pathlib import Path
import statistics

BASE_DIR = Path("/userspace/vvn/HMTL_with_calibration/experiments/automl_newfeatures_v2")

WINNING_DATASETS = {
    "dataset_201_pol": "pol",
    "dataset_41980_sat11_hand_runtime_regression": "SAT11-HAND",
    "dataset_42688_brazilian_houses": "Brazilian_houses",
    "dataset_42726_abalone": "abalone",
    "dataset_574_house_16h": "house_16H",
}

CLOSE_LOSING_DATASETS = {
    "dataset_541_socmob": "socmob",
    "dataset_41021_moneyball": "Moneyball",
    "dataset_42571_allstate_claims_severity": "Allstate",
    "dataset_4549_buzzinsocialmedia_twitter": "Buzz",
    "dataset_43071_mip_2016_regression": "MIP-2016",
    "dataset_531_boston": "boston",
}

ALL_DATASETS = {**WINNING_DATASETS, **CLOSE_LOSING_DATASETS}
SIZES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]


def load_results(dataset_dir):
    path = BASE_DIR / dataset_dir / "results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def get_metric(size_data, model, metric, per_seed_fallback=True):
    """Extract a metric from size_data, with per_seed averaging fallback."""
    val = None

    if model == "hmtl":
        if "hmtl" in size_data and isinstance(size_data["hmtl"], dict):
            val = size_data["hmtl"].get(metric)
    elif model == "catboost":
        if "baselines" in size_data and isinstance(size_data["baselines"], dict):
            val = size_data["baselines"].get("catboost", {}).get(metric)

    if val is None and per_seed_fallback:
        per_seed = size_data.get("per_seed", {})
        vals = []
        for seed_key, seed_data in per_seed.items():
            if seed_data.get("status") != "ok":
                continue
            if model == "hmtl":
                if "hmtl" in seed_data and not seed_data.get("hmtl_skipped", False):
                    v = seed_data["hmtl"].get(metric)
                    if v is not None:
                        vals.append(v)
            elif model == "catboost":
                if "baselines" in seed_data:
                    v = seed_data["baselines"].get("catboost", {}).get(metric)
                    if v is not None:
                        vals.append(v)
        if vals:
            val = sum(vals) / len(vals)

    return val


def extract_scaling(results):
    scaling = {}
    for sk in SIZES:
        if sk not in results.get("sizes", {}):
            continue
        sd = results["sizes"][sk]
        scaling[sk] = {
            "n_train": sd.get("n_train_samples"),
            "h_rmse": get_metric(sd, "hmtl", "rmse"),
            "cb_rmse": get_metric(sd, "catboost", "rmse"),
            "h_unc": get_metric(sd, "hmtl", "mean_uncertainty"),
            "h_rauc": get_metric(sd, "hmtl", "r_auc_mse"),
            "cb_rauc": get_metric(sd, "catboost", "r_auc_mse"),
            "h_epi": get_metric(sd, "hmtl", "mean_epistemic"),
            "h_alea": get_metric(sd, "hmtl", "mean_aleatoric"),
            "cb_unc": get_metric(sd, "catboost", "mean_uncertainty"),
        }
    return scaling


def sep(c="=", w=120):
    print(c * w)


def f(v, w=12, d=6):
    return f"{v:>{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"


def main():
    sep()
    print("HMTL vs CatBoost: COMPREHENSIVE ANALYSIS")
    sep()

    all_data = {}
    for ddir, short in ALL_DATASETS.items():
        results = load_results(ddir)
        if results is None:
            continue
        cat = "WIN" if ddir in WINNING_DATASETS else "LOSS"
        all_data[short] = {
            "dir": ddir,
            "cat": cat,
            "nt": results.get("n_samples_total"),
            "ntr": results.get("n_samples_train"),
            "nf": results.get("n_features"),
            "scaling": extract_scaling(results),
        }

    # ========== 1. DATASET CHARACTERISTICS ==========
    print("\n")
    sep()
    print("1. DATASET CHARACTERISTICS")
    sep()
    header = f"{'Dataset':<22} {'Cat':<6} {'N_total':>8} {'N_train':>8} {'N_feat':>7} {'Samp/Feat':>10}"
    print(header)
    print("-" * len(header))
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sf = info["ntr"] / info["nf"] if info["nf"] else 0
        print(f"{nm:<22} {info['cat']:<6} {info['nt']:>8} {info['ntr']:>8} {info['nf']:>7} {sf:>10.1f}")

    for c in ["WIN", "LOSS"]:
        entries = [v for v in all_data.values() if v["cat"] == c]
        ns = [e["nt"] for e in entries]
        fs = [e["nf"] for e in entries]
        rs = [e["ntr"]/e["nf"] for e in entries if e["nf"]]
        print(f"\n  {c} group: N_total avg={sum(ns)/len(ns):.0f} ({min(ns)}-{max(ns)}), "
              f"N_feat avg={sum(fs)/len(fs):.1f} ({min(fs)}-{max(fs)}), "
              f"S/F avg={sum(rs)/len(rs):.1f} ({min(rs):.1f}-{max(rs):.1f})")

    # ========== 2. SCALING CURVES ==========
    print("\n")
    sep()
    print("2. PER-SIZE SCALING CURVES (RMSE)")
    sep()

    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        print(f"\n--- {nm} ({info['cat']}) | N_train={info['ntr']} | N_feat={info['nf']} ---")
        print(f"  {'Size':>5} {'N':>7} {'H_RMSE':>11} {'CB_RMSE':>11} {'Delta':>11} {'H/CB':>7} {'Win':>5}")
        print("  " + "-" * 60)
        sw = sl = st_s = st_l = 0
        for sk in SIZES:
            if sk not in sc:
                continue
            s = sc[sk]
            h, c = s["h_rmse"], s["cb_rmse"]
            if h is None or c is None:
                continue
            d = h - c
            r = h / c if c else 999
            w = "H" if d < 0 else "CB"
            si = int(sk)
            if si <= 50:
                st_s += 1
                if d < 0: sw += 1
            else:
                st_l += 1
                if d < 0: sl += 1
            print(f"  {sk:>5}% {str(s['n_train']):>7} {h:>11.6f} {c:>11.6f} {d:>+11.6f} {r:>7.4f} {w:>5}")
        print(f"  HMTL wins: small(<=50%)={sw}/{st_s}, large(>50%)={sl}/{st_l}")

    # ========== 3. WIN PATTERN ==========
    print("\n")
    sep()
    print("3. HMTL WIN PATTERN: SMALL vs LARGE DATA")
    sep()
    print(f"\n{'Dataset':<22} {'Cat':<6} {'Small':>8} {'Large':>8} {'Pattern':>28}")
    print("-" * 80)
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        sw = sl = st_s = st_l = 0
        for sk in SIZES:
            if sk not in sc: continue
            s = sc[sk]
            h, c = s.get("h_rmse"), s.get("cb_rmse")
            if h is None or c is None: continue
            if int(sk) <= 50:
                st_s += 1
                if h < c: sw += 1
            else:
                st_l += 1
                if h < c: sl += 1
        sr = sw/st_s if st_s else 0
        lr = sl/st_l if st_l else 0
        if sr > 0.6 and lr > 0.6: pat = "HMTL dominates"
        elif sr > 0.6 and lr <= 0.4: pat = "HMTL better SMALL only"
        elif sr <= 0.4 and lr > 0.6: pat = "HMTL better LARGE only"
        elif sr <= 0.4 and lr <= 0.4: pat = "CatBoost dominates"
        else: pat = "Mixed"
        print(f"{nm:<22} {info['cat']:<6} {sw}/{st_s}({sr:.0%}){'':<1} {sl}/{st_l}({lr:.0%}){'':<1} {pat:>28}")

    # ========== 4. FULL-DATA COMPARISON ==========
    print("\n")
    sep()
    print("4. FULL-DATA (100%) RMSE COMPARISON")
    sep()
    print(f"{'Dataset':<22} {'Cat':<6} {'H_RMSE':>11} {'CB_RMSE':>11} {'H/CB':>7} {'Delta%':>9}")
    print("-" * 70)
    items = []
    for nm, info in all_data.items():
        s = info["scaling"].get("100", {})
        h, c = s.get("h_rmse"), s.get("cb_rmse")
        if h and c:
            items.append((nm, info["cat"], h, c, h/c, (h-c)/c*100))
    for nm, cat, h, c, r, dp in sorted(items, key=lambda x: x[4]):
        mk = " <-WIN" if r < 1.0 else ""
        print(f"{nm:<22} {cat:<6} {h:>11.6f} {c:>11.6f} {r:>7.4f} {dp:>+9.2f}%{mk}")

    # ========== 5. UNCERTAINTY QUALITY ==========
    print("\n")
    sep()
    print("5. UNCERTAINTY QUALITY (100% data)")
    sep()
    print(f"{'Dataset':<22} {'Cat':<6} {'H_RMSE':>9} {'H_Unc':>9} {'U/RMSE':>7} {'Epi':>9} {'Alea':>9} {'E/A':>6} {'H_RAUC':>9} {'CB_RAUC':>9}")
    print("-" * 100)
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        s = info["scaling"].get("100", {})
        hr = s.get("h_rmse"); u = s.get("h_unc"); e = s.get("h_epi"); a = s.get("h_alea")
        hra = s.get("h_rauc"); cra = s.get("cb_rauc")
        ur = u/hr if (u and hr and hr > 0) else None
        ea = e/a if (e and a and a > 0) else None
        print(f"{nm:<22} {info['cat']:<6} {f(hr,9)} {f(u,9)} {f(ur,7,3)} {f(e,9)} {f(a,9)} {f(ea,6,3)} {f(hra,9)} {f(cra,9)}")

    # ========== 6. UNCERTAINTY CALIBRATION ACROSS SIZES ==========
    print("\n")
    sep()
    print("6. UNCERTAINTY/RMSE RATIO ACROSS SIZES")
    sep()
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        print(f"\n  --- {nm} ({info['cat']}) ---")
        print(f"  {'Size':>5} {'H_RMSE':>11} {'H_Unc':>11} {'U/RMSE':>8} {'Epi':>11} {'Alea':>11}")
        print("  " + "-" * 58)
        rl = []
        for sk in SIZES:
            if sk not in sc: continue
            s = sc[sk]
            h = s.get("h_rmse"); u = s.get("h_unc"); e = s.get("h_epi"); a = s.get("h_alea")
            r = u/h if (u and h and h > 0) else None
            if r is not None: rl.append(r)
            rs = f"{r:>8.4f}" if r is not None else f"{'N/A':>8}"
            print(f"  {sk:>5}% {f(h,11)} {f(u,11)} {rs} {f(e,11)} {f(a,11)}")
        if len(rl) > 1:
            m = statistics.mean(rl); sd = statistics.stdev(rl)
            cv = sd/m if m > 0 else 0
            print(f"  Stats: mean={m:.4f} std={sd:.4f} CV={cv:.4f}")

    # ========== 7. SCALING EFFICIENCY ==========
    print("\n")
    sep()
    print("7. SCALING EFFICIENCY: RMSE reduction 10% -> 100%")
    sep()
    print(f"{'Dataset':<22} {'Cat':<6} {'H10':>9} {'H100':>9} {'H_Impr':>7} {'CB10':>9} {'CB100':>9} {'CB_Impr':>7} {'H>CB':>5}")
    print("-" * 85)
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        s10 = sc.get("10", {}); s100 = sc.get("100", {})
        h10, h100 = s10.get("h_rmse"), s100.get("h_rmse")
        c10, c100 = s10.get("cb_rmse"), s100.get("cb_rmse")
        hi = (h10-h100)/h10*100 if (h10 and h100 and h10 > 0) else None
        ci = (c10-c100)/c10*100 if (c10 and c100 and c10 > 0) else None
        b = ""
        if hi is not None and ci is not None:
            b = "YES" if hi > ci else "NO"
        hip = f"{hi:>7.1f}%" if hi is not None else f"{'N/A':>7}"
        cip = f"{ci:>7.1f}%" if ci is not None else f"{'N/A':>7}"
        print(f"{nm:<22} {info['cat']:<6} {f(h10,9)} {f(h100,9)} {hip} {f(c10,9)} {f(c100,9)} {cip} {b:>5}")

    # ========== 8. CROSSOVER ANALYSIS ==========
    print("\n")
    sep()
    print("8. CROSSOVER ANALYSIS")
    sep()
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        prev = None; xovers = []
        for sk in SIZES:
            if sk not in sc: continue
            s = sc[sk]
            h, c = s.get("h_rmse"), s.get("cb_rmse")
            if h is None or c is None: continue
            cur = "HMTL" if h < c else "CB"
            if prev and cur != prev:
                xovers.append(f"at {sk}%: {prev}->{cur}")
            prev = cur
        print(f"\n  {nm} ({info['cat']}): ", end="")
        if xovers:
            print("; ".join(xovers))
        else:
            fw = None
            for sk in SIZES:
                if sk in sc:
                    s = sc[sk]; h, c = s.get("h_rmse"), s.get("cb_rmse")
                    if h is not None and c is not None:
                        fw = "HMTL" if h < c else "CatBoost"
                        break
            print(f"No crossover - {fw} wins ALL sizes")

    # ========== 9. R-AUC-MSE ACROSS SIZES ==========
    print("\n")
    sep()
    print("9. R-AUC-MSE ACROSS SIZES (lower=better)")
    sep()
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        print(f"\n  --- {nm} ({info['cat']}) ---")
        print(f"  {'Size':>5} {'H_RAUC':>11} {'CB_RAUC':>11} {'Win':>5}")
        print("  " + "-" * 35)
        for sk in SIZES:
            if sk not in sc: continue
            s = sc[sk]
            hr, cr = s.get("h_rauc"), s.get("cb_rauc")
            w = ""
            if hr is not None and cr is not None:
                w = "H" if hr < cr else "CB"
            print(f"  {sk:>5}% {f(hr,11)} {f(cr,11)} {w:>5}")

    # ========== 10. SUMMARY ==========
    print("\n")
    sep()
    print("10. SUMMARY: KEY FINDINGS")
    sep()

    print("\nH1: HMTL wins on datasets with more samples?")
    wn = [v["nt"] for v in all_data.values() if v["cat"] == "WIN"]
    ln = [v["nt"] for v in all_data.values() if v["cat"] == "LOSS"]
    print(f"  WIN: avg={sum(wn)/len(wn):.0f} ({min(wn)}-{max(wn)})")
    print(f"  LOSS: avg={sum(ln)/len(ln):.0f} ({min(ln)}-{max(ln)})")

    print("\nH2: HMTL wins on higher-dim datasets?")
    wf = [v["nf"] for v in all_data.values() if v["cat"] == "WIN"]
    lf = [v["nf"] for v in all_data.values() if v["cat"] == "LOSS"]
    print(f"  WIN: avg={sum(wf)/len(wf):.1f} ({min(wf)}-{max(wf)})")
    print(f"  LOSS: avg={sum(lf)/len(lf):.1f} ({min(lf)}-{max(lf)})")

    print("\nH3: HMTL better at small data?")
    for c in ["WIN", "LOSS"]:
        entries = {k: v for k, v in all_data.items() if v["cat"] == c}
        sw = st_s = sl_ = st_l = 0
        for nm, info in entries.items():
            for sk in SIZES:
                if sk not in info["scaling"]: continue
                s = info["scaling"][sk]
                h, cb = s.get("h_rmse"), s.get("cb_rmse")
                if h is None or cb is None: continue
                if int(sk) <= 50:
                    st_s += 1
                    if h < cb: sw += 1
                else:
                    st_l += 1
                    if h < cb: sl_ += 1
        sr = sw/st_s if st_s else 0; lr = sl_/st_l if st_l else 0
        print(f"  {c}: small={sw}/{st_s}({sr:.0%}), large={sl_}/{st_l}({lr:.0%})")

    print("\nH4: Uncertainty calibration on winning vs losing?")
    for c in ["WIN", "LOSS"]:
        entries = {k: v for k, v in all_data.items() if v["cat"] == c}
        urs = []
        for nm, info in entries.items():
            s = info["scaling"].get("100", {})
            h = s.get("h_rmse"); u = s.get("h_unc")
            if h and u and h > 0:
                urs.append((nm, u/h))
        if urs:
            avg = sum(r for _, r in urs) / len(urs)
            print(f"  {c}: avg Unc/RMSE={avg:.4f}")
            for n, r in urs:
                print(f"    {n}: {r:.4f}")

    print("\nFinal RMSE ratios (H/CB) at 100%, sorted:")
    r100 = []
    for nm, info in all_data.items():
        s = info["scaling"].get("100", {})
        h, c = s.get("h_rmse"), s.get("cb_rmse")
        if h and c:
            r100.append((nm, info["cat"], h/c, (h-c)/c*100))
    for nm, cat, r, dp in sorted(r100, key=lambda x: x[2]):
        mk = "<-WIN" if r < 1 else ""
        print(f"  {nm:<22} {cat:<6} ratio={r:.4f} ({dp:+.2f}%) {mk}")

    # ========== 11. TREND ==========
    print("\n")
    sep()
    print("11. H/CB RATIO TREND ACROSS SIZES")
    sep()
    print(f"{'Dataset':<22}", end="")
    for sk in SIZES:
        print(f" {sk+'%':>6}", end="")
    print(f"  {'Trend':>14}")
    print("-" * 105)
    for nm, info in sorted(all_data.items(), key=lambda x: (x[1]["cat"], x[0])):
        sc = info["scaling"]
        print(f"{nm:<22}", end="")
        rft = []
        for sk in SIZES:
            if sk in sc:
                s = sc[sk]
                h, c = s.get("h_rmse"), s.get("cb_rmse")
                if h is not None and c is not None and c != 0:
                    r = h/c; rft.append((int(sk), r))
                    print(f" {r:>6.4f}", end="")
                else:
                    print(f" {'N/A':>6}", end="")
            else:
                print(f" {'N/A':>6}", end="")
        if len(rft) >= 2:
            fh = [r for sz, r in rft if sz <= 50]
            sh = [r for sz, r in rft if sz > 50]
            if fh and sh:
                af = sum(fh)/len(fh); ash = sum(sh)/len(sh)
                if ash < af - 0.01: t = "HMTL improves"
                elif ash > af + 0.01: t = "HMTL degrades"
                else: t = "Stable"
            else: t = "?"
        else: t = "?"
        print(f"  {t:>14}")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)

if __name__ == "__main__":
    main()
