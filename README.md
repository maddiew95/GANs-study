# GANs-study

A reproducibility-oriented study of fault diagnosis for a GPVS dataset under data scarcity, with a focus on whether GAN-generated synthetic samples help downstream classification/anomaly detection.

Most experiment logic lives in notebooks under `GPVS-Faults/` and model definitions in `GPVS-Faults/models/` and `GPVS-Faults/GANs/`.

## Table of contents
- [Project purpose and context](#project-purpose-and-context)
- [Repository layout](#repository-layout)
- [Data and preprocessing pipeline](#data-and-preprocessing-pipeline)
- [Implemented GAN models and configurations](#implemented-gan-models-and-configurations)
- [Implemented classifier/anomaly models and configurations](#implemented-classifieranomaly-models-and-configurations)
- [Evaluation approach](#evaluation-approach)
- [How to reproduce](#how-to-reproduce)
- [Results/reporting outputs](#resultsreporting-outputs)
- [Caveats and limitations](#caveats-and-limitations)

## Project purpose and context
The repository evaluates GPVS fault diagnosis across:
- **Full-data baseline** runs (`fault diagnoses with full/`),
- **Scarcity scene** runs (`fault diagnoses with scenes/`), and
- **GAN-augmented scarcity** runs (`fault diagnoses with GANs/`).

Fault labels are multiclass (`F0`..`F7`), where `F0` is treated as normal operation in the anomaly-detection notebook (`ae.ipynb`).

## Repository layout
- `requirements.txt` — Python dependencies used by notebooks.
- `GPVS-Faults/eda.ipynb` — data inventory/EDA.
- `GPVS-Faults/subsample.ipynb` — creates shared eval split plus per-seed/per-scene training splits.
- `GPVS-Faults/subsample_timegan.ipynb` — creates contiguous multi-chunk train splits for TimeGAN.
- `GPVS-Faults/GANs/`
  - `dcgans.py`, `cgans.py`, `wgans.py`, `timegans.py` — GAN implementations.
  - `dcgans.ipynb`, `cgans.ipynb`, `wgans.ipynb`, `gen_timegan.ipynb` — generate augmented CSVs.
- `GPVS-Faults/models/`
  - `cnn.py`, `cnn_lstm.py`, `lstm_xgb.py`, `transformer_lstm.py`, `transformer_lstm_svm.py`, `ae.py`.
- `GPVS-Faults/fault diagnoses with full/` — baseline notebooks.
- `GPVS-Faults/fault diagnoses with scenes/` — scarcity-only notebooks.
- `GPVS-Faults/fault diagnoses with GANs/` — GAN comparison notebooks.
- `GPVS-Faults/util.py` — shared preprocessing/utilities.
- `GPVS-Faults/ReadMe.pdf` — project report artifact.

## Data and preprocessing pipeline
Expected raw inputs (not versioned in this repo) are CSV files like:
- `GPVS-Faults/CSV_Files/F0M.csv` ... `F7M.csv`

Pipeline shown in notebooks:
1. `subsample.ipynb`
   - fixed holdout per class: `VAL_N=200`, `TEST_N=200`, `HOLDOUT_SEED=0`
   - writes `CSV_Files/eval/val.csv` and `CSV_Files/eval/test.csv`
   - writes `CSV_Files/seed<1-5>/scene<0-6>/train.csv` with class budgets
     `{0:69567, 1:1000, 2:500, 3:100, 4:50, 5:30, 6:10}`.
2. `subsample_timegan.ipynb`
   - builds contiguous multi-chunk TimeGAN train sets for scenes 1..6 under `CSV_Files/TimeGANs_csv/...`
   - preserves shared holdout and performs leakage checks.
3. `util.py`
   - `preprocess_scenario(...)` applies z-score normalization (fit on train, applied to val/test).
   - outlier-removal utility exists (`outlier_remove`) but is commented out in active preprocessing flow.

## Implemented GAN models and configurations

> If a value is not explicitly set in repository code/notebooks, it is marked **Not specified**.

| GAN | Where implemented/evaluated | Core architecture | Input/latent dims | Key layers & activations | Loss / objective | Optimizer | LR | Batch size | Epochs | Other settings | Seeds / scenes / ratios |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DCGAN (tabular MLP, per-class unconditional) | `GPVS-Faults/GANs/dcgans.py`, `GPVS-Faults/GANs/dcgans.ipynb`; evaluated by notebooks in `fault diagnoses with GANs/` | One generator + discriminator per class (`F0..F7`) | features=13, latent_dim=64 (default) | G: Linear 64→128→256→256→13 + LeakyReLU + BatchNorm + `tanh`; D: Linear 13→128→128→1 + LeakyReLU + Dropout(0.1) + `sigmoid` | BCE adversarial loss | Adam (`betas=(0.5,0.999)`) | default `2e-4`; notebook uses `1e-3` | 64 (or min with class count) | default 300; notebook uses 1000 | MinMax scaler to `[-1,1]`; deterministic seeding; `drop_last` for BN-safe batches | `SEEDS=[1..5]`, `SCENES=[1..6]`, `RATIOS=[0,0.5,1,2]` |
| cGAN (conditional MLP) | `GPVS-Faults/GANs/cgans.py`, `GPVS-Faults/GANs/cgans.ipynb`; evaluated in `fault diagnoses with GANs/` | Single conditional generator/discriminator with label embeddings | features=13, latent_dim=64, classes=8 | G: concat(label_emb,z) → Linear→128→256→13 + LeakyReLU + BatchNorm + `tanh`; D: concat(x,label_emb) → Linear→256→128→1 + LeakyReLU + Dropout(0.4) | Least-squares GAN (`MSELoss`) | Adam (`betas=(0.5,0.999)`) | default `2e-4`; notebook uses `1e-3` | 64 | default 300; notebook uses 1000 | Global MinMax scaler to `[-1,1]` on pooled features | `SEEDS=[1..5]`, `SCENES=[1..6]`, `RATIOS=[0,0.5,1,2]` |
| WGAN-GP (tabular MLP critic) | `GPVS-Faults/GANs/wgans.py`, `GPVS-Faults/GANs/wgans.ipynb`; evaluated in `fault diagnoses with GANs/` | Per-class unconditional generator + critic | features=13, latent_dim=64 | G similar to DCGAN MLP; Critic: Linear 13→128→128→1 (no sigmoid, no BN) | Wasserstein with gradient penalty | Adam (`betas=(0.5,0.9)`) | default `2e-4`; notebook uses `1e-3` | 64 | default 300; notebook uses 1000 | `n_critic=5`, `lambda_gp=10` (defaults) | `SEEDS=[1..5]`, `SCENES=[1..6]`, `RATIOS=[0,0.5,1,2]` |
| TimeGAN (5 GRU networks; 3-phase training) | `GPVS-Faults/GANs/timegans.py`, `GPVS-Faults/GANs/gen_timegan.ipynb`; evaluated in `fault diagnoses with GANs/` | Embedder, recovery, generator, supervisor, discriminator GRU stacks | features=13; `seq_len` scene-dependent (`{1:24,2:24,3:24,4:24,5:7,6:5}` in notebook) | GRU + linear heads; sigmoid on latent/feature nets, identity on discriminator output | Phase 1 recon + phase 2 supervised + phase 3 adversarial/joint terms | Adam (four optimizers in module) | `1e-3` | 64 | `n_epochs_emb=300`, `n_epochs_sup=300`, `n_epochs_joint=400` | `hidden_dim=24`, `num_layers=3`, `gamma=1.0`; contiguous training sets from `TimeGANs_csv` then augmentation written to standard `CSV_Files` tree | `SEEDS=[1..5]`, `SCENES=[1..6]`, `RATIOS=[0,0.5,1,2]` |

## Implemented classifier/anomaly models and configurations

| Model | Where implemented/evaluated | Task role | Architecture / features | Input shape | Loss/objective | Optimizer / scheduler | Typical train config in notebooks | Notable model settings |
|---|---|---|---|---|---|---|---|---|
| CNN | `GPVS-Faults/models/cnn.py`; `GPVS-Faults/fault diagnoses with full/cnn.ipynb`, `GPVS-Faults/fault diagnoses with scenes/cnn.ipynb`, `GPVS-Faults/fault diagnoses with GANs/cnn.ipynb` | Multiclass fault classification | Conv1d(1→12,k=2), BN, MaxPool, FC→8 logits | `(B,1,13)` | CrossEntropyLoss | Adam + StepLR(step=20,gamma=0.5) | typically epochs=70, lr=1e-2, weight_decay=1e-4, batch_size=50, seed=0 | Xavier init for Conv/Linear in notebook helpers |
| CNN_LSTM_v1 / CNN_LSTM_v2 (v2 used in notebooks) | `GPVS-Faults/models/cnn_lstm.py`; `GPVS-Faults/fault diagnoses with full/cnn-lstm.ipynb`, `GPVS-Faults/fault diagnoses with scenes/cnn-lstm.ipynb`, `GPVS-Faults/fault diagnoses with GANs/cnn-lstm.ipynb` | Multiclass classification | CNN frontend + LSTM over pooled sequence + FC | `(B,1,13)` | CrossEntropyLoss | Adam + StepLR(step=20,gamma=0.5) | typically epochs=70, lr=1e-2, weight_decay=1e-4, batch_size=50, seed=0 | `CNN_LSTM_v2` selected in notebooks (`MODEL_CLS=CNN_LSTM_v2`) |
| LSTM_XGB | `GPVS-Faults/models/lstm_xgb.py`; `GPVS-Faults/fault diagnoses with full/lstm-xgb.ipynb`, `GPVS-Faults/fault diagnoses with scenes/lstm-xgb.ipynb`, `GPVS-Faults/fault diagnoses with GANs/lstm-xgb.ipynb` | Two-stage multiclass classifier | Stage 1: LSTM backbone (hidden=32); Stage 2: XGBoost on extracted features | `(B,1,13)` | Stage1 CE; Stage2 XGBoost multi-class softprob | Stage1 Adam + StepLR(step=20,gamma=0.5); Stage2 XGBClassifier | stage1 typically epochs=70, lr=1e-2, weight_decay=1e-4, batch_size=50 | XGB params include `learning_rate=0.1`, `n_estimators=56`, `max_depth=6`, `tree_method="hist"`, `objective="multi:softprob"` |
| Transformer_LSTM | `GPVS-Faults/models/transformer_lstm.py`; `GPVS-Faults/fault diagnoses with full/transformer-lstm.ipynb`, `GPVS-Faults/fault diagnoses with scenes/transformer-lstm.ipynb`, `GPVS-Faults/fault diagnoses with GANs/transformer-lstm.ipynb` | Multiclass classification | Linear input projection + positional embedding + TransformerEncoder + LSTM + FC | `(B,1,13)` | CrossEntropyLoss | Adam (+ StepLR in full/scenes notebooks; disabled in GAN compare notebook) | full/scenes: multi-seed runs with epochs=200, lr=1e-3; GAN compare notebook sets `EPOCHS=200`, `LR=1e-3`, `WD=0`, `USE_SCHED=False` | Module defaults: `d_model=13`, `n_heads=1`, `n_attn_layers=2`, `dropout=0.28`, `lstm_hidden=8` |
| Transformer_LSTM_SVM | `GPVS-Faults/models/transformer_lstm_svm.py`; `GPVS-Faults/fault diagnoses with full/transformer-lstm-svm.ipynb`, `GPVS-Faults/fault diagnoses with scenes/transformer-lstm-svm.ipynb`, `GPVS-Faults/fault diagnoses with GANs/transformer-lstm-svm.ipynb` | Two-stage multiclass classifier | Stage 1 Transformer+LSTM backbone; Stage 2 RBF SVM on backbone features | `(B,1,13)` | Stage1 CE; Stage2 SVM classification | Stage1 Adam (no scheduler in model helper); Stage2 `sklearn.svm.SVC` | typically stage1 epochs=200, lr=1e-3, batch_size=50; scenes/full notebooks aggregate over `SEEDS=range(10)` | SVM: `kernel="rbf"`, `gamma="scale"`, `decision_function_shape="ovr"`; backbone defaults include `dropout=0.41`, `lstm_hidden=32`, `d_model=13` |
| AnomalyAE / AnomalyDetector | `GPVS-Faults/models/ae.py`; `GPVS-Faults/fault diagnoses with full/ae.ipynb`, `GPVS-Faults/fault diagnoses with scenes/ae.ipynb`, `GPVS-Faults/fault diagnoses with GANs/ae.ipynb` | Binary anomaly detection (`F0` vs faults) | MLP autoencoder + score calibration | tensor built from same 13 features (`(B,1,13)` accepted) | Reconstruction-based anomaly score (`score_mode` supports `mse`, `mahalanobis`, `combined`) | Adam + StepLR(step=20,gamma=0.5) in detector `fit` | notebook uses detector config `in_dim=13`, `hidden_dims=(64,32)`, `latent_dim=16`, `score_mode="mse"`; train call typically epochs=70, batch_size=50 | Threshold calibrated on healthy validation data with `target_far=0.05` |

## Evaluation approach
- Metrics in classifier notebooks: accuracy, macro precision, macro recall, macro F1, confusion matrices.
- GAN-comparison notebooks aggregate F1 over `(gan, scene, ratio)` and compute deltas vs ratio `0.0` baseline.
- Full/scenes Transformer-based notebooks run multi-seed summaries (typically `SEEDS=range(10)` in those notebooks).
- AE notebooks evaluate binary anomaly detection (including AUC and per-class detection rates).

## How to reproduce

### Required
1. **Create environment and install dependencies**
   ```bash
   cd /home/runner/work/GANs-study/GANs-study
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Provide raw CSV data** under `GPVS-Faults/CSV_Files/` as expected by notebooks (`F0M.csv` ... `F7M.csv`).
3. **Run notebook pipeline in order** (from `GPVS-Faults/`):
   - `eda.ipynb`
   - `subsample.ipynb`
   - (for TimeGAN path) `subsample_timegan.ipynb`
   - GAN generation notebooks in `GPVS-Faults/GANs/`
   - evaluation notebooks in:
     - `fault diagnoses with full/`
     - `fault diagnoses with scenes/`
     - `fault diagnoses with GANs/`

### Optional
- Skip TimeGAN generation (`gen_timegan.ipynb`) if only reproducing DCGAN/cGAN/WGAN experiments.
- Use only `fault diagnoses with full/` for full-data baselines without scarcity/GAN analysis.

## Results/reporting outputs
Notebooks write summary CSV artifacts (file names vary by notebook), including patterns such as:
- `*_results_by_scenario.csv`
- `*_gan_compare_raw.csv`
- `*_gan_compare_agg.csv`
- `*_gan_compare_delta.csv`

Recommended reporting flow:
1. Run notebook(s) for a model family.
2. Use the generated summary CSV(s) for tables/plots.
3. For GAN studies, compare best ratio per scene against ratio `0.0` baseline.

## Caveats and limitations
- Repository is notebook-first; there is no single CLI experiment runner.
- Dataset CSVs are expected but not included in this repository.
- Many notebooks default to `device = "cuda:1"`; adjust for your hardware.
- TimeGAN notebooks intentionally use a separate contiguous training tree (`TimeGANs_csv`) before writing augmented outputs into the standard scene tree.
- GAN-comparison notebooks include cache-purge logic (including temporary `PURGE_GANS=["timegans"]` in notebook code) tied to prior cache issues; read those cells before resuming partial runs.
- `util.py` contains outlier-removal logic, but active preprocessing currently uses z-score normalization only.
