"""
BCICIV_2a (BCI Competition IV-2a) Left-vs-Right motor imagery benchmark.

What it does:
- Loads BCICIV_2a via MOABB dataset wrapper BNCI2014_001
- For each of 9 subjects:
    - Uses official train/test split if available (session contains 'train'/'test')
      otherwise uses a stratified 70/30 split
    - Computes baselines: chance + majority
    - Trains CSP+GaussianNB and CSP+LDA (classic MI baseline)
- Writes:
    results/bciciv2a_per_subject_results.csv
    results/bciciv2a_summary_results.csv
    results/bciciv2a_mean_std_bar.png
    results/bciciv2a_per_subject_lines.png
- Ablation grid:
    bands: (8–12), (13–30), (8–30)
    CSP components: 2, 4, 6, 8
  Writes:
    results/bciciv2a_ablation_grid.csv
    results/bciciv2a_ablation_grid.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

# ----------------------------
# MOABB + MNE imports
# ----------------------------
try:
    import moabb
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery
    from mne.decoding import CSP
    MOABB_OK = True
    try:
        import mne
        mne.set_log_level("WARNING")
    except Exception:
        pass
except Exception as e:
    MOABB_OK = False
    MOABB_IMPORT_ERROR = e

# ============================================================
# CONFIG
# ============================================================
SEED = 42
SUBJECTS = list(range(1, 10))     # subjects 1..9
DEFAULT_BAND = (8, 30)
TMIN = 0.0                        # 0..4s after cue
TMAX = 4.0

CSP_N_COMPONENTS = 6              # "main" setting

# Ablation grid
ABL_BANDS = [(8, 12), (13, 30), (8, 30)]
ABL_COMPONENTS = [2, 4, 6, 8]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Helpers
# ============================================================
def band_to_str(band: tuple[int, int]) -> str:
    return f"{band[0]}-{band[1]}"

def chance_baseline_accuracy(y_test: np.ndarray) -> float:
    classes = np.unique(y_test)
    return 1.0 / len(classes)

def majority_baseline_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    vals, counts = np.unique(y_train, return_counts=True)
    majority = vals[np.argmax(counts)]
    y_pred = np.full_like(y_test, fill_value=majority)
    return float(accuracy_score(y_test, y_pred))

def train_test_split_from_meta(y: np.ndarray, meta: pd.DataFrame):
    """
    Use official train/test session split if available (often '0train'/'1test').
    Otherwise do a stratified random split (reproducible).
    """
    if "session" in meta.columns:
        session = meta["session"].astype(str)
        train_mask = session.str.contains("train", case=False, na=False).to_numpy()
        test_mask = session.str.contains("test", case=False, na=False).to_numpy()
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            return train_mask, test_mask

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, train_size=0.7, random_state=SEED, stratify=y)
    train_mask = np.zeros(len(y), dtype=bool); train_mask[tr_idx] = True
    test_mask  = np.zeros(len(y), dtype=bool);  test_mask[te_idx] = True
    return train_mask, test_mask

# ============================================================
# Loading BCICIV_2a via MOABB (BNCI2014_001)
# ============================================================
def load_bciciv2a_left_right(subject: int, band=(8, 30)):
    if not MOABB_OK:
        raise ImportError(
            "MOABB/MNE not available. Install with:\n"
            "  pip install moabb mne pandas scikit-learn numpy scipy matplotlib\n"
            f"Import error: {MOABB_IMPORT_ERROR}"
        )

    moabb.set_download_dir(str(OUT_DIR / "moabb_data"))
    moabb.set_log_level("warning")

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=band[0], fmax=band[1], tmin=TMIN, tmax=TMAX)

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    return X, np.asarray(y), meta

# ============================================================
# Evaluation per subject
# ============================================================
def evaluate_subject(subject: int, band=(8, 30), n_components=6):
    X, y, meta = load_bciciv2a_left_right(subject, band=band)
    train_mask, test_mask = train_test_split_from_meta(y, meta)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    acc_chance = float(chance_baseline_accuracy(y_test))
    acc_maj = float(majority_baseline_accuracy(y_train, y_test))

    pipe_nb = make_pipeline(
        CSP(n_components=n_components, log=True, norm_trace=False),
        GaussianNB()
    )
    pipe_nb.fit(X_train, y_train)
    acc_nb = float(accuracy_score(y_test, pipe_nb.predict(X_test)))

    pipe_lda = make_pipeline(
        CSP(n_components=n_components, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    pipe_lda.fit(X_train, y_train)
    acc_lda = float(accuracy_score(y_test, pipe_lda.predict(X_test)))

    return {
        "subject": subject,
        "band": band_to_str(band),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "chance": acc_chance,
        "majority": acc_maj,
        "csp_gaussian_nb": acc_nb,
        "csp_lda": acc_lda,
    }

def run_all_subjects_main():
    rows = []
    for s in SUBJECTS:
        print(f"[Main] Evaluating subject {s} (band={band_to_str(DEFAULT_BAND)}, CSP={CSP_N_COMPONENTS})")
        rows.append(evaluate_subject(s, band=DEFAULT_BAND, n_components=CSP_N_COMPONENTS))

    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    df.to_csv(OUT_DIR / "bciciv2a_per_subject_results.csv", index=False)

    metric_cols = ["chance", "majority", "csp_gaussian_nb", "csp_lda"]
    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [df[c].mean() for c in metric_cols],
        "std":  [df[c].std(ddof=1) for c in metric_cols],
    })
    summary.to_csv(OUT_DIR / "bciciv2a_summary_results.csv", index=False)

    # Plot: mean ± std bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["metric"], summary["mean"], yerr=summary["std"], capsize=5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("BCICIV_2a Left/Right — Mean ± Std across subjects (main setting)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bciciv2a_mean_std_bar.png", dpi=200)
    plt.close(fig)

    # Plot: per-subject lines
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = df["subject"].to_numpy()
    for c in metric_cols:
        ax.plot(x, df[c].to_numpy(), marker="o", label=c)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title("BCICIV_2a Left/Right — Per-subject accuracies (main setting)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bciciv2a_per_subject_lines.png", dpi=200)
    plt.close(fig)

    print("\n[Main] Wrote:")
    print(" - results/bciciv2a_per_subject_results.csv")
    print(" - results/bciciv2a_summary_results.csv")
    print(" - results/bciciv2a_mean_std_bar.png")
    print(" - results/bciciv2a_per_subject_lines.png")

# ============================================================
# Ablation grid
# ============================================================
def run_ablation_grid(bands, components_list):
    """
    Writes:
      - results/bciciv2a_ablation_grid.csv (per-config subject accuracies + mean±std)
      - results/bciciv2a_ablation_grid.png (one figure)
    """
    methods = ["csp_lda", "csp_gaussian_nb"]
    records = []

    for band in bands:
        for n_comp in components_list:
            print(f"[Ablation] band={band_to_str(band)} CSP={n_comp}")

            acc_by_method = {m: [] for m in methods}
            chance_list = []
            maj_list = []

            for s in SUBJECTS:
                res = evaluate_subject(s, band=band, n_components=n_comp)
                chance_list.append(res["chance"])
                maj_list.append(res["majority"])
                for m in methods:
                    acc_by_method[m].append(res[m])

            for m in methods:
                row = {
                    "band": band_to_str(band),
                    "band_low": band[0],
                    "band_high": band[1],
                    "csp_components": n_comp,
                    "method": m,
                    "chance_mean": float(np.mean(chance_list)),
                    "majority_mean": float(np.mean(maj_list)),
                    "mean_acc": float(np.mean(acc_by_method[m])),
                    "std_acc": float(np.std(acc_by_method[m], ddof=1)),
                }
                for s_idx, s in enumerate(SUBJECTS):
                    row[f"acc_s{s}"] = float(acc_by_method[m][s_idx])
                records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values(["method", "band_low", "band_high", "csp_components"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "bciciv2a_ablation_grid.csv", index=False)

        # Plot: mean accuracy vs CSP components, separate lines per band (two panels: LDA and NB)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, method in zip(axes, ["csp_lda", "csp_gaussian_nb"]):
        dsub = df[df["method"] == method].copy()
        for band in bands:
            bstr = band_to_str(band)
            db = dsub[dsub["band"] == bstr].sort_values("csp_components")
            ax.plot(db["csp_components"], db["mean_acc"], marker="o", label=f"band {bstr}")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean accuracy")
        ax.set_title(f"Ablation — {method} (mean across 9 subjects)")
        ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("CSP n_components")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bciciv2a_ablation_grid.png", dpi=200)
    plt.close(fig)

    print("\n[Ablation] Wrote:")
    print(" - results/bciciv2a_ablation_grid.csv")
    print(" - results/bciciv2a_ablation_grid.png")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_all_subjects_main()
    run_ablation_grid(bands=ABL_BANDS, components_list=ABL_COMPONENTS)

