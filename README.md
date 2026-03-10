# MSIN0097 – Predictive Analytics Coursework

### Turbofan Engine RUL Prediction (NASA C-MAPSS FD001)

**Key documents**

-   Main analysis notebook → `rul_prediction_cmapss.ipynb`
-   Written report & appendix → submitted here and on Moodle (PDF) MSIN0097-coursework-report-appendix.pdf (logs are in the appendix)

  My Candidate number - VGKQ6

------------------------------------------------------------------------

## Project Overview

This repository contains my individual coursework submission for MSIN0097 (Predictive Analytics, 2025–26).

The objective is to build a fully reproducible end-to-end predictive system that predicts the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor telemetry. The system is designed to demonstrate:

-   Clear problem framing with safety-critical success metrics (RMSE, MAE, R²)
-   Rigorous data exploration and validation (outlier analysis, leakage risk assessment)
-   Careful evaluation design with leakage prevention (engine-level splits, train-only scaling)
-   Transparent agent-assisted development using Claude Code
-   Full reproducibility via environment specification and deterministic seeds
-   Governed hyperparameter tuning and robust post-tuning evaluation

This project does **not** focus on maximising predictive performance at the expense of validity. Instead, the emphasis is placed on methodological correctness, auditability, and responsible use of AI coding assistants.

------------------------------------------------------------------------

## Dataset

The project uses the **NASA C-MAPSS FD001** dataset (Saxena et al., 2008) — an industry-standard benchmark for predictive maintenance research.

**Subset**: FD001 — single operating condition, single fault mode.

| Split | Engines | Rows   | Description                                      |
|-------|---------|--------|--------------------------------------------------|
| Train | 100     | 20,631 | Run-to-failure (full degradation trajectories)   |
| Test  | 100     | 13,096 | Truncated before failure (RUL must be predicted) |

Each engine records **21 sensor channels** and **3 operational settings** per cycle.

### Data Access

Download from the [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) and place these three files in the project root:

```         
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### Data Source

NASA Prognostics Center of Excellence (2008). Turbofan Engine Degradation Simulation Data Set (C-MAPSS). Accessed February 2026.

------------------------------------------------------------------------

## Predictive Task

**Regression**: predict how many operating cycles remain before engine failure, given multivariate sensor readings at a single time step.

Target definition:

-   RUL = max_cycle − current_cycle (per engine)
-   RUL capped at 125 cycles — sensors show no distinguishable degradation above this threshold (Heimes, 2008)
-   Capping focuses model capacity on the safety-critical near-failure window

Grouping constraint:

-   `unit_id` (engine ID) is used for grouped train/validation splitting to prevent temporal leakage across engines

Primary evaluation metrics:

-   **RMSE** (penalises dangerous large errors in safety-critical context)
-   **MAE** (interpretable average error in cycles)
-   **R²** (variance explained)

Final selected model:

-   Gradient Boosting Regressor (n_estimators=100, max_depth=5, learning_rate=0.05)
-   Hyperparameters tuned via grid search over 12 configurations, selected by validation RMSE

------------------------------------------------------------------------

## Results

| Model                         | Val RMSE  | Test RMSE | Test MAE  | Test R²   |
|-------------------------------|-----------|-----------|-----------|-----------|
| Linear Regression             | 20.00     | 22.00     | 17.78     | 0.720     |
| Random Forest                 | 16.46     | 18.98     | 13.44     | 0.791     |
| **Gradient Boosting (Tuned)** | **16.25** | **18.43** | **13.12** | **0.803** |
| Neural Network (MLP)          | 16.33     | 18.60     | 13.41     | 0.800     |

-   Best performance in the safety-critical 0–25 cycle window (RMSE = 14.2)
-   71% of test engines predicted within ±20 cycles
-   Published FD001 benchmarks report RMSE 12–20 (Ramasso & Saxena, 2014); our result (18.43) falls within this competitive range
-   Test set evaluated exactly once, after all model selection decisions were finalised

------------------------------------------------------------------------

## Modelling and Evaluation

The modelling pipeline follows strict evaluation discipline throughout.

Implemented components:

-   Engine-level GroupShuffleSplit (80/20) with zero engine overlap
-   MinMaxScaler fitted on training engines only (leakage prevention)
-   7 data readiness assertions verified before any modelling
-   Baseline linear model (Linear Regression)
-   Ensemble model (Random Forest, 100 trees, max_depth=10)
-   Boosted ensemble (Gradient Boosting with grid search tuning)
-   Neural network (PyTorch MLP: 45→128→64→32→1 with BatchNorm, Dropout, early stopping)
-   5-fold GroupKFold cross-validation for robust model ranking
-   Ablation study isolating rolling feature contributions
-   Permutation importance for model-agnostic feature ranking
-   Error CDF, residual analysis, and failure mode analysis by RUL range

All preprocessing is applied consistently: scaler fit on training data only, applied without refitting to validation and test sets.

------------------------------------------------------------------------

## Reproducibility

This repository is designed to be fully reproducible.

Key principles:

-   Environment specified via `requirements.txt` (Python 3.11.x)
-   All random seeds fixed (`random_state=42`, `torch.manual_seed(42)`)
-   Deterministic train/validation split via `GroupShuffleSplit(random_state=42)`
-   Test set touched exactly once after final model selection
-   All figures saved to `figures/`
-   Clear separation between raw data and derived outputs

------------------------------------------------------------------------

## Agent Governance

Claude Code (Anthropic) was used as a development aid throughout the project for code scaffolding, debugging, and documentation drafting.

AI tools were used strictly as development assistants. All modelling decisions, evaluation design, and interpretations were made and verified by the author.

Three material agent mistakes were identified and corrected:

1.  **Scaler leakage** (★) — Agent fitted MinMaxScaler on entire dataset before train/val split. Corrected to train-only fitting.
2.  **Missing neural network** (★) — Agent proposed only 3 classical models, omitting the brief's requirement for a modern approach. PyTorch MLP added manually.
3.  **Test set exposure** (★) — Agent computed test metrics during exploration and grid search. Corrected to evaluate test set exactly once.

Four analytical additions went beyond agent suggestions:

-   5-fold GroupKFold cross-validation
-   Permutation importance for MLP
-   Error CDF for distributional analysis
-   Ablation study on rolling features

Full interaction logs and verification decisions are documented in the appendix of the written report (submitted on Moodle).

------------------------------------------------------------------------

## Repository Structure

```         
.
├── rul_prediction_cmapss.ipynb   # Main notebook (Steps 1–6)
├── requirements.txt              # Python dependencies (pip install -r requirements.txt)
├── README.md                     # This file
├── LICENSE                       # MIT licence
│
├── train_FD001.txt               # Training data (100 engines, run-to-failure)
├── test_FD001.txt                # Test data (100 engines, truncated)
└── RUL_FD001.txt                 # Ground truth RUL for test set
```

Running the notebook generates a `figures/` directory with all plots (15 figures saved automatically).

------------------------------------------------------------------------

## Notebook Workflow

The analysis follows a structured six-step end-to-end workflow within a single notebook:

| Step | Section | Key Activities |
|----|----|----|
| 1 | Problem Framing | Target definition, success metrics, agent tooling plan |
| 2 | EDA | Sensor correlations, degradation trends, outlier & leakage checks |
| 3 | Data Preparation | Rolling features (window=5), RUL cap at 125, engine-level GroupShuffleSplit |
| 4 | Model Exploration | 4 models compared on validation set + 5-fold GroupKFold CV |
| 5 | Fine-Tuning | GB grid search (12 configs), ablation study on rolling features |
| 6 | Evaluation | Pred vs actual, error analysis, feature importance, failure modes |

------------------------------------------------------------------------

## Environment Setup

``` bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### Quick Reproduction

After setting up the environment and downloading the dataset:

1.  Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in the project root
2.  Run `rul_prediction_cmapss.ipynb` cell-by-cell
3.  All figures are saved to `figures/` automatically

------------------------------------------------------------------------

## References

-   Saxena, A., Goebel, K., Simon, D. and Eklund, N. (2008) 'Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation', *Proceedings of PHM08*, Denver, CO.
-   Heimes, F.O. (2008) 'Recurrent Neural Networks for Remaining Useful Life Estimation', *Proceedings of PHM08*, Denver, CO.
-   Ramasso, E. and Saxena, A. (2014) 'Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets', *IJPHM*, 5(2), pp. 1–15.
-   Kingma, D.P. and Ba, J. (2015) 'Adam: A Method for Stochastic Optimization', *Proceedings of ICLR*, San Diego, CA.
-   Li, X., Ding, Q. and Sun, J.Q. (2018) 'Remaining useful life estimation in prognostics using deep convolution neural networks', *Reliability Engineering & System Safety*, 172, pp. 1–11.
-   Zheng, S., Ristovski, K., Farahat, A. and Gupta, C. (2017) 'Long short-term memory network for remaining useful life estimation', *Proceedings of IEEE ICPHM*, Dallas, TX, pp. 88–95.
