# Uncertainty Quantification for Clinical Risk Prediction

A careful evaluation of uncertainty methods on a small clinical dataset (UCI Heart Disease, Cleveland, n = 297). The question: **do MC Dropout, Bayes-by-Backprop variational inference and Deep Ensembles give better-calibrated and more useful uncertainty than a plain neural network, in distribution and under age-based distribution shift?**

The heart disease data is a testbed. The focus is the evaluation: calibration metrics, paired bootstrap intervals, and controls for every surprising result.

## Key findings

1. **In distribution, no method clearly beats the plain baseline.** All four reach about 0.60 accuracy (majority-class rate is 0.54) and are overconfident by roughly 4 to 7 points. ECE intervals overlap almost completely.
2. **Predictive entropy flags wrong predictions equally well for every method** (AUROC about 0.80 to 0.82), including the plain network. No method is significantly better than the baseline's softmax entropy.
3. **Under age shift, every method degrades and becomes badly overconfident** (confidence exceeds accuracy by roughly 15 to 28 points).
4. **MC Dropout's apparent gain under shift is mostly "being less confident".** A network trained with dropout but evaluated deterministically recovers most of the gain, and a baseline with a single temperature parameter (fitted on the young training data only) matches or beats MC Dropout.
5. **An untuned epoch count made VI look far worse** (600 epochs: ECE 0.157, Brier 0.562). With the epoch count chosen on an inner validation split (200), VI is indistinguishable from the baseline. The 600-epoch run is kept as an ablation, not as a result.

## Methods

| Method | Description |
|---|---|
| Baseline | MLP (2 x 64, ReLU), cross-entropy, Adam, 100 full-batch epochs |
| MC Dropout [1] | Same MLP with dropout p = 0.2, kept on at test time, 50 stochastic passes averaged |
| Deep Ensemble [2] | 10 MLPs differing only in random initialisation, probabilities averaged |
| VI (Bayes by Backprop) [3] | Gaussian weight posteriors, N(0, 1) prior, KL / N term, 200 epochs, 50 samples at test time |

Controls added after the main comparison: dropout network evaluated with dropout off, and the baseline with temperature scaling [4].

Metrics: expected calibration error (10 equal-width bins in distribution, 5 under shift) [4, 5], Brier score [6], negative log-likelihood, and AUROC of predictive entropy for separating correct from incorrect predictions. Confidence intervals use the nonparametric bootstrap over patients [7]; method differences use paired resamples. The shift experiments follow the spirit of evaluating uncertainty under dataset shift [8].

## Data and setup

- UCI Heart Disease (Cleveland) [9, 10], `fetch_ucirepo(id=45)`. 303 rows, 13 features, 6 rows with missing `ca` or `thal` dropped, leaving 297.
- All 5 severity levels (0 to 4) kept. Class counts: 160 / 54 / 35 / 35 / 13. Class 4 is never predicted by any method except VI (rarely), so no class-4 conclusions are drawn.
- `cp`, `restecg`, `slope`, `thal` one-hot encoded. Continuous features standardised, with the scaler fit on training data only inside each fold.
- Evaluation: stratified 5-fold CV (seed 42), out-of-fold probabilities for every patient.
- Epoch counts chosen on an inner split of the training data (never on the evaluation fold): 100 for baseline, MC Dropout and ensemble; 200 for VI.

## Results

### In distribution (5-fold CV, out-of-fold predictions)

| Method | Acc | ECE | Brier | NLL | Mean conf - acc | Entropy AUROC |
|---|---|---|---|---|---|---|
| Baseline | 0.596 | 0.074 | 0.500 | 0.957 | +0.067 | 0.811 |
| MC Dropout | 0.599 | 0.059 | 0.486 | 0.938 | +0.042 | 0.820 |
| Deep Ensemble | 0.603 | 0.081 | 0.503 | 0.965 | +0.059 | 0.800 |
| VI (BBB, 200 ep) | 0.596 | 0.066 | 0.505 | 0.972 | +0.065 | 0.816 |

Paired bootstrap vs baseline (negative Brier / NLL is better): MC Dropout Brier -0.014 [-0.027, -0.001], NLL -0.019 [-0.047, 0.008]; Deep Ensemble and VI intervals include 0. A deterministic dropout network scored Brier 0.493 / NLL 0.945, so most of MC Dropout's small edge is regularisation.

### Under age shift (mean over 5 seeds)

Split 1: train age <= 55 (n = 146), test age >= 61 (n = 79). Split 2: train age <= 50 (n = 92), test age >= 60 (n = 91). Both cut-offs were fixed before seeing results.

| Method | Split 1 Brier | Split 1 NLL | Split 2 Brier | Split 2 NLL |
|---|---|---|---|---|
| Baseline | 0.708 | 1.538 | 0.753 | 1.562 |
| Baseline + temperature scaling | **0.657** | **1.346** | 0.698 | 1.377 |
| Dropout (eval mode) | 0.688 | 1.449 | 0.708 | 1.444 |
| MC Dropout | 0.676 | 1.407 | **0.692** | 1.400 |
| Deep Ensemble | 0.716 | 1.540 | 0.731 | 1.470 |
| VI (BBB) | 0.700 | 1.491 | 0.734 | 1.427 |

Temperature scaling beats the plain baseline in both splits (Brier about -0.05, NLL about -0.19, intervals exclude 0). Against MC Dropout it is better in split 1 and indistinguishable in split 2. Fitted temperatures were 1.2 to 1.5.

## Figures

- `results/figures/reliability_indist_vs_shift.png`: reliability diagrams, in distribution vs both shifted splits
- `results/figures/rejection_curve.png`: accuracy vs fraction of patients kept (most confident first)
- `results/figures/class_distribution.png`, `results/figures/feature_distributions.png`: EDA

## Limitations

- **Small data.** 297 patients; the shifted test sets have 79 and 91 patients; class 4 has 13 patients (2 to 3 in the shifted training sets).
- **Weak classifier.** About 60% accuracy in distribution (majority-class rate 0.54), and 40 to 50% under shift (majority-class rate about 0.42 to 0.44). Calibration gains are on a weak model.
- **One training seed per fold** in the main CV comparison, so the bootstrap intervals do not capture seed variance. The shift experiments average 5 seeds.
- **The two shift splits are not independent.** Their test sets overlap heavily and the training sets are nested.
- **Label shift is mixed with covariate shift.** Older patients have more disease, so the class mix also changes.
- **Temperature scaling was fitted on young data only**, so it cannot fully correct the shift (confidence still exceeds accuracy by about 15 to 22 points).
- **Not run:** MC Dropout combined with temperature scaling, so the contribution of MC averaging beyond recalibration is untested. VI's prior and initial variance were not tuned.
- Many intervals are reported without multiple-comparison correction; marginal results (lower bounds near 0) should not be over-read.

## Reproducing

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install ucimlrepo pandas numpy matplotlib scikit-learn scipy torch jupyter
jupyter notebook   # open notebooks/ and Restart & Run All
```

The raw data is downloaded and saved to `data/raw/heart_disease_raw.csv`. Out-of-fold probabilities and metric tables are written to `results/metrics/`.

## References

1. Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*. arXiv:1506.02142.
2. Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*. arXiv:1612.01474.
3. Blundell, C., Cornebise, J., Kavukcuoglu, K. and Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML*. arXiv:1505.05424.
4. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*. arXiv:1706.04599.
5. Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles. *AAAI*.
6. Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability. *Monthly Weather Review*, 78(1), 1-3.
7. Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
8. Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift. *NeurIPS*. arXiv:1906.02530.
9. Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
10. Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.

## Repository layout

```
data/raw/          raw CSV
notebooks/         analysis notebook(s)
results/metrics/   saved probabilities and summary CSVs
results/figures/   plots
```
