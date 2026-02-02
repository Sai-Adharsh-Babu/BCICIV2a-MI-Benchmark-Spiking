# Reproducible Motor-Imagery EEG Decoding on BCICIV-2a with Strong Baselines, Per-Subject Analysis, and Ablations + Synthetic Spiking Simulation & Bayesian Decoding (Brian2 + Elephant)

## Overview
This repository contains a **reproducible benchmark** for decoding **left-vs-right motor imagery** using the **BCI Competition IV-2a (BCICIV_2a)** EEG dataset, implemented with **MOABB/MNE** for standardized access and preprocessing.  
It also includes a **separate simulation module** that generates synthetic spike trains using **Brian2**, analyzes them with **Elephant**, and performs **Bayesian decoding** (Naive Bayes) to recover known ground-truth states.

To keep claims clear and credible, the project is organized into:
- **EEG decoding pipeline (real data):** BCICIV_2a left vs right MI classification  
- **Synthetic spiking pipeline (simulation study):** spiking activity + decoding  
- **Bridge is exploratory:** spiking simulation is **not** claiming physiological ground truth from EEG

---

## Key Features (What makes it “strong”)
### EEG (BCICIV_2a)
- **Chance baseline** (sanity check)
- **Majority-class baseline** (guards against imbalance)
- **CSP + LDA** (classic MI-BCI baseline)
- **CSP + Gaussian Naive Bayes** (probabilistic baseline)
- **Per-subject results for all 9 subjects**
- **Mean ± Std across subjects**
- **Ablation study** over:
  - frequency bands: **8–12**, **13–30**, **8–30** Hz  
  - CSP components: **2, 4, 6, 8**

### Spiking Simulation (Brian2 + Elephant)
- Synthetic spike trains generated at the population level
- Feature extraction from spike counts (total/early/late bins)
- Bayesian decoding to recover known trial labels
- Spike-train visualization (raster, PSTH) + summary outputs

---

## Objectives
### EEG Benchmark (BCICIV_2a)
- Load and preprocess BCICIV_2a using MOABB/MNE
- Extract CSP spatial features
- Train strong classical baselines (LDA, GaussianNB)
- Report per-subject performance + mean ± std
- Run ablations to test sensitivity to preprocessing choices

### Spiking Simulation
- Generate synthetic spike trains using Brian2-style modeling
- Analyze firing patterns and PSTH with Elephant
- Apply Bayesian decoding to infer simulated neural states
- Compare predicted vs actual labels to evaluate accuracy

---

## Tools & Libraries
- **Python 3.10+ (recommended: 3.11 works)**
- EEG:
  - `mne`, `moabb`
  - `numpy`, `scipy`, `pandas`, `matplotlib`
  - `scikit-learn`
- Spiking:
  - `brian2`, `elephant`, `neo`, `quantities`
  - `numpy`, `pandas`, `matplotlib`
  - `scikit-learn`

---

## Dataset (BCICIV_2a)
This project uses the **BCI Competition IV Dataset 2a** (motor imagery EEG).

**The dataset is not included** in this repository due to licensing/distribution restrictions.  
However, the EEG pipeline uses **MOABB**, which can automatically download the dataset when you run the script.

> If MOABB fails to download (network/firewall), download from the official competition source and configure MOABB accordingly.

---

## Repository Structure  
```
Reproducible Motor-Imagery EEG Decoding on BCICIV-2a with Strong Baselines, Per-Subject Analysis, and Ablations + Synthetic Spiking Simulation & Bayesian Decoding (Brian2 + Elephant)/
│
├── src/
│ ├── eeg/
│ │ └── project_bciciv2a_baselines_ablation.py              # EEG pipeline: baselines + per-subject + ablations
│ │
│ └── spiking/
│ └── spiking_sim_brian2_elephant.py                        # Spiking simulation + Elephant analysis + decoding
│
├── figures/
│   ├── eeg/
│   │   ├── bciciv2a_mean_std_bar.png
│   │   ├── bciciv2a_per_subject_lines.png
│   │   └── bciciv2a_ablation_grid.png
│   └── spiking/
│       ├── spiking_raster_example.png
│       └── spiking_psth_example.png
│
├── reports/
│   ├── eeg/
│   │   ├── abstract.md
│   │   └── tables/
│   │       ├── bciciv2a_per_subject_results.csv
│   │       ├── bciciv2a_summary_results.csv
│   │       └── bciciv2a_ablation_grid.csv
│   └── spiking/
│       ├── abstract.md
│       ├── spiking_trial_features.csv
│       └── spiking_summary.txt
│
├── requirements/
│   ├── eeg/
│   │   └── requirements.txt
│   └── spiking/
│       └── requirements.txt
│
├── data/
│   └── README_DATA.md
│
├── .gitignore
├── .gitattributes
├── LICENSE
└── README.md

```

## Installation
You can set this up in **one environment (simpler)** or **two environments (recommended)**.

### Option A — Single environment (simpler)
```powershell
python -m pip install -U pip
pip install numpy scipy pandas matplotlib scikit-learn mne moabb
pip install brian2 elephant neo quantities
```

### Option B — Two environments (recommended: avoids dependency conflicts)
- **EEG environment (BCICIV_2a decoding)**
  ```powershell
  py -m venv .venv_eeg
  .\.venv_eeg\Scripts\Activate.ps1
  python -m pip install -U pip
  pip install numpy scipy pandas matplotlib scikit-learn mne moabb
  ```

- **Spiking environment (Brian2 + Elephant)**
  ```powershell
  py -m venv .venv_spiking
  .\.venv_spiking\Scripts\Activate.ps1
  python -m pip install -U pip
  pip install numpy pandas matplotlib scikit-learn
  pip install brian2 elephant neo quantities
  ```
  
---

## Usage
### Run EEG benchmark (BCICIV_2a: baselines + per-subject + ablations)
python .\src\eeg\project_bciciv2a_baselines_ablation.py

### Run spiking simulation + Bayesian decoding
python .\src\spiking\spiking_sim_brian2_elephant.py

---

## Expected Results
### EEG (BCICIV_2a)
- Per-subject accuracies for all **9 subjects**
- Summary performance as **mean ± std** across subjects
- Strong baselines:
  - chance baseline
  - majority-class baseline
  - CSP+LDA
  - CSP+Gaussian Naive Bayes
- Ablation grid showing sensitivity to:
  - frequency band (8–12, 13–30, 8–30 Hz)
  - CSP components (2, 4, 6, 8)

### Spiking (Simulation)
- Synthetic spike trains (raster plot)
- PSTH/firing rate visualization
- Trial-wise spike-count features (CSV)
- Bayesian decoding accuracy vs known simulated labels

---

## Notes (Skeptic-friendly / Clear claims)
- **EEG pipeline (real data):** Decodes left vs right motor imagery from BCICIV_2a using MOABB/MNE + CSP features.
- **Synthetic spiking pipeline (simulation study):** Generates spike trains with known labels and decodes them; this validates the decoding method under controlled ground truth.
- **No physiological claim:** The spiking simulation is **not** presented as a biological ground-truth model of EEG—it's an exploratory modeling and decoding demonstration.

---

## Current Status
- EEG benchmark complete (baselines + per-subject + mean±std + ablations)
- Spiking simulation complete (Brian2 + Elephant + Bayesian decoding + plots)
- Potential upgrades:
  - add confidence intervals/statistical tests
  - add more classifiers (logistic regression, SVM)
  - add nested CV / hyperparameter selection per subject
  - add a final written report PDF in `reports/`

---

## License  
This project is licensed under the terms specified in the **[LICENSE](./LICENSE)** file.  

---

## Citation / Acknowledgments
- **BCI Competition IV Dataset 2a (BCICIV_2a)** for motor imagery EEG data
- **MOABB** and **MNE** for standardized EEG dataset handling and preprocessing
- **Brian2** for spiking simulation
- **Elephant** and **Neo** for spike-train analysis tooling

