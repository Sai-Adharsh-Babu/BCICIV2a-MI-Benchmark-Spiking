# Reproducible Motor-Imagery EEG Decoding on BCICIV-2a with Strong Baselines, Per-Subject Analysis, and Ablations

## Overview
This project explores how spiking neural activity can be modeled and decoded using computational methods.  
Neural firing data are simulated using Brian2, analyzed with Elephant, and decoded using a Naive Bayesian classifier to infer neural states.  
The study aims to bridge theoretical neuroscience and computational modeling, contributing to future non-invasive brain–machine interface (BMI) research.

## Objectives
- Generate synthetic spike train data using Brian2.  
- Analyze firing patterns and rate distributions with Elephant.  
- Apply Bayesian decoding to classify simulated neural activity.  
- Compare predicted vs. actual neural states to evaluate decoding accuracy.

## Tools and Libraries
- Python 3.9+
- Brian2 – Spiking neuron simulation  
- Elephant – Spike train analysis  
- Neo – Data management for electrophysiology  
- NumPy, Matplotlib, SciPy  
- Scikit-learn – Bayesian classifier implementation

## Dataset
The project uses the **EEG Motor Imagery Dataset (BCICIV_2a)** in `.csv` format.  
Due to licensing, the dataset is **not included** here. You can request or download it from the official [BCI Competition IV Dataset 2a page](http://www.bbci.de/competition/iv/).

## Repository Structure  
```
neural-spike-simulation-bayes/
│
├── src/
│   ├── neural_spike_bayes.py          # Simulation + Bayesian decoding
│
├── data/
│   ├── README_DATA.md                 # Current dataset in use
│
├── results/
│   └── (to be added after experiment)
│
├── requirements.txt
└── README.md

```

## Usage
Run the main simulation file:
python src/neural_spike_bayes.py
Output plots (raster plots, rate histograms, confusion matrices) will be generated in the results/ folder after experimental runs.

## Expected Results
- Simulated spiking neuron activity (raster plots)  
- Firing rate histograms and ISI distributions  
- Bayesian decoding accuracy visualization  

## Current Status
- Code and dataset prepared  
- Simulation and Bayesian decoding setup complete  
- Experimental runs and result analysis pending  
- Planned updates: performance testing, parameter tuning, and documentation of findings

## License  
This project is licensed under the terms specified in the **[LICENSE](./LICENSE)** file.  

## Notes
This project is currently ongoing as part of a neuroscience research portfolio.  
Updates will include final analysis, results, and documentation once experiments are completed.
