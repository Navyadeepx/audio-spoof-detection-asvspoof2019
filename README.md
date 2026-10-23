# ASVspoof2019 LA — Spoofed Speech Detection with an Audio Transformer + Pretrained AASIST Hybrid

## 1. Project Title

Audio Spoof Detection on ASVspoof2019 LA using a Hybrid of a Transformer Encoder on 128-band Log-Mel Spectrograms (AudioTransformer) and the Pretrained AASIST Graph-Attention Network on Raw Waveforms

## 2. Student Details

- **Name:** Navyadeep
- **Registration Number:** 2430010370
- **Section:** D

## 3. Dataset

- **Dataset name:** ASVspoof 2019, Logical Access (LA) partition
- **Dataset source:** ASVspoof 2019 Challenge dataset (bonafide vs. spoofed/synthetic speech)
- **Number of samples:** 121,461 total (train + validation + test)
- **Number of classes:** 2 — `bonafide`, `spoof`
- **Train/validation/test split:**

| Split      | Bonafide | Spoof  | Total  | % of all data | % Bonafide | % Spoof | Spoof:Bonafide ratio |
|------------|---------:|-------:|-------:|--------------:|-----------:|--------:|---------------------:|
| Train      | 2,580    | 22,800 | 25,380 | 20.90%        | 10.17%     | 89.83%  | 8.84 : 1             |
| Validation | 2,548    | 22,296 | 24,844 | 20.45%        | 10.26%     | 89.74%  | 8.75 : 1             |
| Test       | 7,355    | 63,882 | 71,237 | 58.65%        | 10.32%     | 89.68%  | 8.69 : 1             |

(see `results/dataset_distribution.csv`). The dataset is strongly imbalanced toward spoof samples (~8.7–8.8 : 1) in every split, with nearly identical class proportions across splits.

- **Preprocessing performed** (see `results/preprocessing_params.csv`):

| Parameter                    | Value                                         | Unit                                                          |
|------------------------------|-----------------------------------------------|---------------------------------------------------------------|
| Sample rate (SR)             | 16000                                         | Hz                                                            |
| Clip duration                | 4                                             | seconds                                                       |
| Fixed waveform length        | 64000                                         | samples                                                       |
| Number of mel bands (N_MELS) | 128                                           | bands                                                         |
| Feature type                 | Log-mel spectrogram (per-clip standardised)   | —                                                             |
| Mono conversion              | Channel-averaged if multi-channel             | —                                                             |
| AASIST raw-waveform length   | 64600                                         | samples (first 64600, repeat-padded if shorter, as in AASIST) |
| Waveform cache dtype         | int16 (memory-mapped)                         | —                                                             |
| Resampling                   | `librosa.resample` to SR if source sr differs | —                                                             |

Log-mel spectrograms (128 x 126) and raw waveforms (64,600 samples) are computed once per split and cached to disk, so no feature extraction happens during training.

## 4. Model

- **Model/architecture name:** Hybrid AudioTransformer + AASIST (`ResidualHybrid`). The transformer branch is written layer by layer with `torch.nn` / `torch.nn.TransformerEncoder` and trained from scratch; the AASIST branch is the official pretrained model, used as a frozen feature extractor.
- **Architecture description:**
  - **Branch A — AudioTransformer** (log-mel input, 1 x 128 x 126): SpecAugment (train only) → patch embedding `Conv2d(1, 128, kernel=(16, 8), stride=(16, 8))` giving 120 tokens → learned `[CLS]` token + learned positional embedding → 4 x `TransformerEncoderLayer` (d_model 128, 4 heads, FF 256, GELU, pre-norm, dropout 0.3) + final LayerNorm → concat(`[CLS]`, mean of patch tokens) = 256-d embedding
  - **Branch B — AASIST** (raw waveform, 64,600 samples): `clovaai/aasist` model + released pretrained weights → 160-d embedding + 2 logits. Downloaded and imported by the notebook, not re-implemented
  - **Fusion head:** `[transformer 256 ‖ AASIST 160]` → LayerNorm → Dropout → Linear(416→128) → GELU → Dropout → Linear(128→2)
  - **Residual fusion:** final logits = AASIST logits + fusion-head output; the last fusion layer is zero-initialised, so training starts exactly at pretrained-AASIST behaviour
  - **Auxiliary head** on the transformer embedding (Linear 256→2, loss weight 0.3, training only)
  - Total / trainable parameters: 915,150 / 617,284 (transformer 562,304; AASIST 297,866, frozen; fusion + aux heads 54,980) — see `results/computational_cost.csv`
  - Saved weights: `models/hybrid_transformer_aasist_stage1.pt` (see `models/model_description.txt` for load instructions)
- **Pretrained model:** AASIST only (official `clovaai/aasist` weights, trained on the ASVspoof2019 LA training partition). The transformer branch and fusion head use random initialisation.
- **Proposed modifications:** A small transformer encoder over log-mel patches (global self-attention across time and frequency) is fused with AASIST through a residual correction on top of AASIST's own logits. Stage 1 (reported here) trains only the transformer + fusion head with AASIST frozen, using cached AASIST outputs. A stage-2 fine-tune (everything unfrozen at LR 3e-5 / 1e-5 for AASIST) is defined in the notebook config but was **not run** in this submission.

## 5. Hyperparameters

| Hyperparameter            | Value                                                                                                                                                  |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Learning rate (stage 1)   | 0.0003 (transformer + fusion head; AASIST frozen)                                                                                                      |
| Learning rate (stage 2)   | 0.00003 transformer/fusion, 0.00001 AASIST — configured, not run                                                                                       |
| Batch size                | 24 (AASIST's official batch size)                                                                                                                      |
| Number of epochs          | 30 max (early stopping; stage 1 stopped at epoch 6, best epoch 1)                                                                                      |
| Optimizer                 | AdamW (no weight decay on biases, norms, positional embedding, `[CLS]`)                                                                                |
| Loss function             | Cross-entropy on the fused head + 0.3 x cross-entropy on each auxiliary head (transformer head, AASIST head), plus an L1 term on the transformer branch |
| Scheduler                 | 1 epoch linear warm-up, then cosine decay (stepped per batch, floor 1% of peak LR)                                                                     |
| Dropout                   | 0.3                                                                                                                                                    |
| Weight decay              | 0.0001                                                                                                                                                 |
| Other parameters          | L1 lambda = 0.00001 (transformer branch only); gradient clipping 5.0; SpecAugment 2 freq + 2 time masks (<=16 bins/frames); early stopping patience = 5 epochs; 5-fold cross-validation (10 epochs, patience 3 per fold); N_MELS = 128; class weights = none; seed = 42 |

## 6. Hyperparameter Tuning

Full log in `results/hyperparameter_tuning_log.csv`. **Most transformer-specific values were not tuned**: they are initial settings chosen from common practice, and the log says so explicitly. Values shared with the earlier CNN experiments were reused, not re-tested.

| Hyperparameter                 | Values tested                      | Final value  | Reason for selection                                                                                              |
|--------------------------------|------------------------------------|--------------|-------------------------------------------------------------------------------------------------------------------|
| Learning rate (stage 1)        | not tuned                          | 3e-4         | Pre-norm transformers train stably around 3e-4 with AdamW + warm-up; 1e-3 (used for the CNN) is usually too high  |
| Batch size                     | not tuned                          | 24           | AASIST's official batch size; the raw-waveform branch dominates GPU memory                                        |
| N_MELS                         | 64, 128 (CNN experiments)          | 128          | 128 bands gave better validation accuracy than 64 for the CNN; reused                                             |
| d_model / heads / layers / FF  | not tuned                          | 128/4/4/256  | Small encoder (~0.56M params) comparable in size to AASIST (~0.30M)                                               |
| Patch size (mel, frames)       | not tuned                          | (16, 8)      | AST-style patches; 120 tokens for a 128x126 spectrogram, cheap for self-attention                                 |
| Dropout probability            | 0.2, 0.3, 0.5 (CNN experiments)    | 0.3          | Best train/validation trade-off for the CNN; reused                                                               |
| Weight decay (AdamW)           | 1e-3, 1e-4, 1e-5 (CNN experiments) | 1e-4         | Best for the CNN; also AASIST's official value                                                                    |
| L1 lambda (transformer branch) | 1e-4, 1e-5, 0 (CNN experiments)    | 1e-5         | Mild sparsity pressure, restricted to the new branch so pretrained AASIST weights are not pulled toward zero      |
| Auxiliary loss weight          | not tuned                          | 0.3          | Keeps each branch individually discriminative                                                                     |
| SpecAugment masks (freq/time)  | not tuned                          | 16/16        | Regularises the transformer against the train-to-evaluation attack mismatch                                       |
| Early stopping patience        | 3, 5, 8 (CNN experiments)          | 5            | Avoided stopping on short-term validation-loss noise                                                              |
| K-fold CV folds                | 3, 5 (CNN experiments)             | 5            | Stable estimate of fold-to-fold variability                                                                       |

## 7. Results

As this is a spoofed-speech detection model, spoof-class metrics are the primary focus. Full per-class and aggregate metrics are in `results/results.csv`; the comparison with published systems is in `results/comparison.csv`; per-epoch training curve values are in `results/hybrid_stage1_history.json`; 5-fold CV results are in `results/cv_results.json`; trained weights are in `models/`.

**Decision threshold note.** Accuracy, precision, recall, F1 and specificity are computed at each model's EER operating point on the test scores, i.e. the threshold is chosen using the test labels. EER and ROC-AUC do not depend on a threshold. Because the threshold sits at the EER point, spoof recall, bonafide recall (specificity) and accuracy are nearly equal. Choosing the threshold on the validation set instead is the stricter protocol.

Corresponding figures are in `figures/`:

- `hybrid_stage1_loss_curve.png`, `hybrid_stage1_accuracy_curve.png` — stage 1 training curves
- `cv_boxplot_fold_metrics.png` — boxplot of the per-fold 5-fold CV scores (accuracy/precision/recall/F1)
- `cv_fold_differences.png` — deviation of each fold from the mean, per metric

| Model                                     | Accuracy | Precision (spoof) | Recall (spoof) | F1 (spoof) | ROC-AUC | EER    |
|-------------------------------------------|---------:|------------------:|---------------:|-----------:|--------:|-------:|
| Hybrid AudioTransformer + AASIST (stage 1) | 99.17%   | 99.91%            | 99.17%         | 99.54%     | 0.9993  | 0.83%  |
| Ablation: transformer head only           | 86.51%   | 98.24%            | 86.51%         | 92.00%     | 0.9300  | 13.48% |
| AASIST pretrained, no training here       | 99.20%   | 99.91%            | 99.20%         | 99.55%     | 0.9993  | 0.80%  |

**Headline result — Hybrid (stage 1):** spoof precision 99.91%, spoof recall 99.17%, accuracy 99.17%, specificity 99.18%, EER 0.83%, ROC-AUC 0.9993. Bonafide precision is 93.21% (bonafide recall 99.18%), which reflects the ~9:1 spoof-heavy class balance.

**The hybrid does not improve on pretrained AASIST.** On this notebook's test split, the pretrained AASIST run through the same pipeline gets EER 0.80% against 0.83% for the hybrid, a difference of 0.03 percentage points that should be read as no measurable gain. The training log agrees: the best validation loss came at epoch 1 (0.0108) and rose afterwards (to 0.0158 by epoch 6) while training loss kept falling, so early stopping restored epoch 1, which is essentially the zero-initialised residual hybrid that starts at AASIST's behaviour. The transformer branch alone is much weaker (EER 13.48%), so the strength of the system comes from the pretrained AASIST branch, not the transformer.

**5-fold cross-validation (training split):** accuracy 1.0000 ± 0.0001, precision 1.0000 ± 0.0001, recall 1.0000 ± 0.0000, F1 1.0000 ± 0.0000 (mean ± std over folds; fold 2 is the only fold below 1.0, at accuracy 0.9998). These numbers are **optimistic**: the released AASIST weights were trained on the entire ASVspoof2019 LA training partition, which includes every CV validation fold, so the CV is a stability check across folds and not an unbiased generalisation estimate. Validation and test splits are not affected by this.

**Validation vs. test.** Validation accuracy in stage 1 is 0.9964–0.9977 and test accuracy is 99.17%, a smaller gap than the earlier CNN had. ASVspoof2019 LA's evaluation partition contains spoofing attacks that are not all present in the training/development data, which is the usual reason test performance sits below validation. Class balance is nearly identical across splits, so imbalance does not explain it.

**Comparison with published systems (CO4)** — see `results/comparison.csv`, same ASVspoof2019 LA benchmark, EER (lower is better). NR = metric not reported in the cited publication; it was not estimated.

| Model                                                                  | EER    |
|------------------------------------------------------------------------|-------:|
| LFCC + GMM (ASVspoof2019 baseline B1) — classical baseline             | 15.80% |
| LFCC + LCNN — CNN-based reference architecture                         | 9.26%  |
| RawNet2 — published deep-learning system                               | 9.50%  |
| AASIST — as published (authors' own pipeline)                          | 0.83%  |
| AASIST pretrained — this notebook's pipeline / test split              | 0.80%  |
| Proposed: AudioTransformer + AASIST (stage 1, AASIST frozen)           | 0.83%  |

The hybrid (0.83% EER) is far below the classical GMM baseline and well below LFCC+LCNN (9.26%) and RawNet2 (9.50%), and matches the published AASIST figure (0.83%). The published AASIST number comes from the authors' own protocol and pipeline, so the like-for-like reference is the pretrained AASIST row measured here (0.80%). With 915,150 parameters the hybrid is about 3x larger than AASIST alone (297,866 parameters; ~0.30M as published) and about 19x smaller than RawNet2 (17.6M in the cited re-implementation).

**Computational cost** (`results/computational_cost.csv`):

| Tag           | Epochs run | Best epoch | Total train time (s) | Avg time/epoch (s) | Device | Total parameters | Trainable parameters |
|---------------|-----------:|-----------:|---------------------:|-------------------:|--------|-----------------:|---------------------:|
| Hybrid-stage1 | 6          | 1          | 172.06               | 28.68              | cuda   | 915,150          | 617,284              |

Stage 1 trains on cached AASIST outputs, so AASIST itself is never run during training; the cost of the raw-waveform branch shows up at test-time scoring and in the waveform cache on disk (about 130 KB per clip).

## 8. Innovation

Not attempted in this submission — this project targets **CO3 (Apply)** and **CO4 (Comparative Analysis)** only. No CO5 innovation/ablation claim is made beyond the branch ablation reported in `results/results.csv`.

## 9. Folder Structure

```
2430010370/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── asvspoof2019_la_transformer_aasist_hybrid.ipynb
│
├── results/
│   ├── results.csv
│   ├── comparison.csv
│   ├── dataset_distribution.csv
│   ├── preprocessing_params.csv
│   ├── hyperparameter_tuning_log.csv
│   ├── computational_cost.csv
│   ├── cv_results.json
│   └── hybrid_stage1_history.json
│
├── models/
│   ├── hybrid_transformer_aasist_stage1.pt
│   └── model_description.txt
│
└── figures/
    ├── cv_boxplot_fold_metrics.png
    ├── cv_fold_differences.png
    ├── hybrid_stage1_loss_curve.png
    └── hybrid_stage1_accuracy_curve.png
```
