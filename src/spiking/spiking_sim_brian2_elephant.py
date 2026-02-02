"""
Spiking simulation + decoding demo (Brian2 + Elephant-friendly outputs).

What it does:
1) Simulates 2 classes (state 0 vs 1) across many trials.
   - For each trial, uses a Brian2 PoissonGroup with class-dependent firing rates.
2) Extracts spike-count features (counts per neuron per trial).
3) Trains a simple Poisson Naive Bayes classifier (implemented here; sklearn has no PoissonNB).
4) Saves outputs to: results/spiking/
   - spiking_trial_features.csv
   - spiking_summary.txt
   - spiking_raster_example.png
   - spiking_psth_example.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Brian2 (simulation)
# ----------------------------
from brian2 import Hz, ms, second, PoissonGroup, SpikeMonitor, defaultclock, prefs, run, start_scope

# ----------------------------
# sklearn (train/test + metrics)
# ----------------------------
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIG
# ============================================================
SEED = 42
RNG = np.random.default_rng(SEED)

N_TRIALS = 200
TRIAL_DUR = 1.0  # seconds
N_NEURONS = 30

# Baseline firing + class-dependent modulation
BASE_RATE_HZ = 12.0
MOD_MAX_HZ = 8.0

# Feature split
TEST_SIZE = 0.3

# Output
OUT_DIR = Path("results") / "spiking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make Brian2 run without needing a C++ compiler (important on Windows)
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms


# ============================================================
# A simple Poisson Naive Bayes (since sklearn has NO PoissonNB)
# log p(x|c) ~ sum_j [ x_j * log(lambda_cj) - lambda_cj ]  (drop const log(x!))
# ============================================================
class PoissonNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha: float = 1e-9):
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if np.any(X < 0):
            raise ValueError("PoissonNB expects non-negative count features.")

        self._le = LabelEncoder().fit(y)
        y_enc = self._le.transform(y)
        self.classes_ = self._le.classes_

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # log prior with simple smoothing
        class_counts = np.bincount(y_enc, minlength=n_classes).astype(float)
        self.class_log_prior_ = np.log((class_counts + 1.0) / (class_counts.sum() + n_classes))

        # Poisson rate per class/feature = mean count (+ alpha)
        self.lam_ = np.zeros((n_classes, n_features), dtype=float)
        for c in range(n_classes):
            Xc = X[y_enc == c]
            mu = Xc.mean(axis=0) if Xc.shape[0] else np.zeros(n_features)
            self.lam_[c] = np.maximum(mu + self.alpha, self.alpha)

        self.log_lam_ = np.log(self.lam_)
        self.lam_sum_ = self.lam_.sum(axis=1)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # (n_samples, n_features) @ (n_features, n_classes) -> (n_samples, n_classes)
        return self.class_log_prior_ + X @ self.log_lam_.T - self.lam_sum_

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        idx = np.argmax(jll, axis=1)
        return self.classes_[idx]


# ============================================================
# Simulation helpers
# ============================================================
def make_class_rates(n_neurons: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two firing-rate vectors (Hz) for class 0 and class 1.
    We generate a modulation pattern and flip it between classes.
    """
    mod = RNG.uniform(0.0, MOD_MAX_HZ, size=n_neurons)
    signs = RNG.choice([-1.0, 1.0], size=n_neurons)
    pattern = mod * signs

    r0 = np.clip(BASE_RATE_HZ + pattern, 0.1, None)
    r1 = np.clip(BASE_RATE_HZ - pattern, 0.1, None)
    return r0, r1


def simulate_one_trial(rates_hz: np.ndarray, trial_dur_s: float, n_neurons: int):
    """
    Run a single PoissonGroup simulation and return spike indices + times (seconds).
    """
    start_scope()
    defaultclock.dt = 1 * ms

    G = PoissonGroup(n_neurons, rates=rates_hz * Hz)
    M = SpikeMonitor(G)
    run(trial_dur_s * second)

    spike_i = np.asarray(M.i, dtype=int)
    spike_t = np.asarray(M.t / second, dtype=float)
    return spike_i, spike_t


# ============================================================
# Main
# ============================================================
def main():
    rates0, rates1 = make_class_rates(N_NEURONS)

    labels = RNG.integers(0, 2, size=N_TRIALS)  # 0 or 1
    features = np.zeros((N_TRIALS, N_NEURONS), dtype=int)

    # For plotting examples
    raster_trials_to_plot = min(25, N_TRIALS)
    raster_store = []  # list of (trial_idx, spike_i, spike_t, label)

    neuron_for_psth = 0
    psth_times_class0 = []
    psth_times_class1 = []

    for tr in range(N_TRIALS):
        y = int(labels[tr])
        rates = rates0 if y == 0 else rates1

        spike_i, spike_t = simulate_one_trial(rates, TRIAL_DUR, N_NEURONS)
        counts = np.bincount(spike_i, minlength=N_NEURONS)
        features[tr] = counts

        # save a few trials for raster
        if tr < raster_trials_to_plot:
            raster_store.append((tr, spike_i, spike_t, y))

        # store one neuron's spike times for PSTH
        t0 = spike_t[spike_i == neuron_for_psth]
        if y == 0:
            psth_times_class0.append(t0)
        else:
            psth_times_class1.append(t0)

    # ----------------------------
    # Decode from spike counts
    # ----------------------------
    X = features.astype(float)
    y = labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    clf = PoissonNaiveBayes(alpha=1e-6)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Save features
    df = pd.DataFrame(X, columns=[f"count_n{j}" for j in range(N_NEURONS)])
    df.insert(0, "label", y)
    df.insert(0, "trial", np.arange(N_TRIALS))
    df.to_csv(OUT_DIR / "spiking_trial_features.csv", index=False)

    # Summary text
    summary_text = (
        f"Spiking simulation + decoding (Brian2)\n"
        f"- trials: {N_TRIALS}\n"
        f"- neurons: {N_NEURONS}\n"
        f"- trial duration: {TRIAL_DUR:.3f}s\n"
        f"- base rate: {BASE_RATE_HZ:.2f} Hz\n"
        f"- modulation max: {MOD_MAX_HZ:.2f} Hz\n"
        f"- classifier: Poisson Naive Bayes (custom)\n"
        f"- test size: {TEST_SIZE:.2f}\n"
        f"- accuracy: {acc:.4f}\n"
    )
    (OUT_DIR / "spiking_summary.txt").write_text(summary_text, encoding="utf-8")

    # ----------------------------
    # Raster plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for tr, spike_i, spike_t, ytr in raster_store:
        # y-axis: neuron index + trial offset to separate trials visually
        y_offset = tr * (N_NEURONS + 2)
        ax.scatter(spike_t, spike_i + y_offset, s=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index (stacked by trial)")
    ax.set_title("Raster (first trials; each trial stacked)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spiking_raster_example.png", dpi=200)
    plt.close(fig)

    # ----------------------------
    # PSTH (manual, robust)
    # ----------------------------
    bin_w = 0.02  # 20 ms
    bins = np.arange(0.0, TRIAL_DUR + bin_w, bin_w)

    def psth_rate_hz(times_list: list[np.ndarray]) -> np.ndarray:
        if len(times_list) == 0:
            return np.zeros(len(bins) - 1)
        all_times = np.concatenate(times_list) if any(len(t) for t in times_list) else np.array([])
        counts, _ = np.histogram(all_times, bins=bins)
        rate = counts / (len(times_list) * bin_w)  # spikes/s = Hz
        return rate

    r0 = psth_rate_hz(psth_times_class0)
    r1 = psth_rate_hz(psth_times_class1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(bins[:-1], r0, where="post", label="class 0")
    ax.step(bins[:-1], r1, where="post", label="class 1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"PSTH (neuron {neuron_for_psth})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spiking_psth_example.png", dpi=200)
    plt.close(fig)

    print("[Spiking] Wrote:")
    print(f" - {OUT_DIR / 'spiking_trial_features.csv'}")
    print(f" - {OUT_DIR / 'spiking_summary.txt'}")
    print(f" - {OUT_DIR / 'spiking_raster_example.png'}")
    print(f" - {OUT_DIR / 'spiking_psth_example.png'}")
    print(f"[Spiking] Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
