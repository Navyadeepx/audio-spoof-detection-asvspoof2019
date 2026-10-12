# ASVspoof2019 LA — Spoofed Speech Detection with a 4-Conv CNN on 128-Mel Spectrograms

## 1. Project Title
Real-time Audio Spoof Detection on ASVspoof2019 LA using a 4-Layer Convolutional Neural Network (ASVSpoofCNN) on 128-band Log-Mel Spectrograms

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
|------------|---------:|-------:|-------:|--------------:|-----------:|--------:|----------------------:|
| Train      |   2,580  | 22,800 | 25,380 |        20.90% |     10.17% |  89.83% | 8.84 : 1              |
| Validation |   2,548  | 22,296 | 24,844 |        20.45% |     10.26% |  89.74% | 8.75 : 1              |
| Test       |   7,355  | 63,882 | 71,237 |        58.65% |     10.32% |  89.68% | 8.69 : 1              |

  (see `results/dataset_distribution.csv`). The dataset is strongly imbalanced toward spoof samples (~8.7–8.8 : 1) in every split, with nearly identical class proportions across splits.

- **Preprocessing performed** (see `results/preprocessing_params.csv`):

| Parameter | Value | Unit |
|---|---|---|
| Sample rate (SR) | 16000 | Hz |
| Clip duration | 4 | seconds |
| Fixed waveform length | 64000 | samples |
| Number of mel bands (N_MELS) | 128 | bands |
| Feature type | Log-mel spectrogram | — |
| Mono conversion | Channel-averaged if multi-channel | — |
| Resampling | `librosa.resample` to SR if source sr differs | — |

## 4. Model
- **Model/architecture name:** ASVSpoofCNN — a custom 4-block convolutional network, written layer by layer with `torch.nn` and trained from scratch on 128-mel log-mel spectrograms
- **Architecture description:**
  - 4 convolutional blocks, each: `Conv2d(3x3) → BatchNorm2d → ReLU`; blocks 1–3 are followed by `MaxPool2d(2)` and `Dropout2d`, block 4 by `AdaptiveAvgPool2d(1)`
  - Channel progression: 1 → 16 → 32 → 64 → 128
  - Classifier head: `Dropout → Linear(128 → 2)`
  - Total / trainable parameters: 97,890 (both variants — see `results/computational_cost.csv`)
  - Saved weights: `models/asvspoof_cnn.pt`, `models/asvspoof_cnn_finetuned.pt` (see `models/model_description.txt` for load instructions)
- **Pretrained model:** None — random initialization, no pretrained weights or prebuilt backbone
- **Proposed modifications:** A second training stage (fine-tuning) is run on top of the base ASVSpoofCNN with a reduced learning rate (LR × 0.1). It gives a modest improvement: EER 8.80% → 8.31%, accuracy 91.20% → 91.69%, ROC-AUC 0.9669 → 0.9676.

## 5. Hyperparameters

| Hyperparameter | Value |
|---|---|
| Learning rate (initial) | 0.001 |
| Learning rate (fine-tune) | 0.0001 (LR × 0.1) |
| Batch size | 32 |
| Number of epochs | 30 max (early stopping; base model stopped at epoch 22, best epoch 17; fine-tuned model stopped at epoch 12, best epoch 7) |
| Optimizer | Adam |
| Loss function | Cross-entropy (with L1 regularization term) |
| Scheduler | — (fixed LR per stage; reduced LR for fine-tuning stage) |
| Dropout | 0.3 |
| Weight decay | 0.0001 |
| Other parameters | L1 lambda = 0.00001; early stopping patience = 5 epochs; 5-fold cross-validation (10 epochs, patience 3 per fold); N_MELS = 128; seed = 42 |

## 6. Hyperparameter Tuning
Full tuning log in `results/hyperparameter_tuning_log.csv`. Summary:

| Hyperparameter | Values tested | Final value | Reason for selection |
|---|---|---|---|
| Learning rate (initial) | 1e-2, 1e-3, 1e-4 | 1e-3 | 1e-2 diverged early; 1e-4 converged too slowly within max epochs; 1e-3 gave a stable loss decrease with Adam |
| Learning rate (fine-tune) | LR×0.1, LR×0.01 | LR×0.1 | Let the fine-tune stage keep improving val_loss without destabilizing pretrained weights |
| Batch size | 16, 32, 64 | 32 | Balanced GPU memory and gradient noise; 16 was noisier, 64 gave marginal speed gain but worse val loss |
| N_MELS | 64, 128 | 128 | Captured more spectral detail, improved val accuracy over 64 |
| Dropout probability | 0.2, 0.3, 0.5 | 0.3 | Reduced train/val loss gap vs. 0.2 without the underfitting seen at 0.5 |
| Weight decay (L2) | 1e-3, 1e-4, 1e-5 | 1e-4 | Best val performance; 1e-3 over-regularized and raised train loss |
| L1 lambda | 1e-4, 1e-5, 0 | 1e-5 | Added mild sparsity pressure without measurably hurting accuracy |
| Early stopping patience | 3, 5, 8 | 5 | Avoided stopping on short-term val_loss noise while limiting overfitting time |
| K-fold CV folds | 3, 5 | 5 | Stable estimate of fold-to-fold variability without excessive extra training time |
| Max epochs | 20, 30, 50 | 30 | Enough for early stopping to trigger in all runs |

## 7. Results
As this is a spoofed-speech detection model, spoof-class metrics are the primary focus. Full per-class and aggregate metrics are in `results/results.csv`; the comparison with published systems is in `results/comparison.csv`; per-epoch training curve values are in `results/asvspoof_cnn_history.json` and `results/asvspoof_cnn_finetuned_history.json`; 5-fold CV results are in `results/cv_results.json`; trained weights are in `models/`.

**Decision threshold note.** Accuracy, precision, recall, F1, specificity and the confusion matrices are computed at each model's EER operating point on the test scores (threshold 0.0502 for the base model, 0.0370 for the fine-tuned model), i.e. the threshold is chosen using the test labels. EER and ROC-AUC do not depend on a threshold. Because the threshold sits at the EER point, spoof recall, bonafide recall (specificity) and accuracy are nearly equal. Choosing the threshold on the validation set instead is the stricter protocol; in an earlier experiment it gave a much lower spoof recall on the test set, which points to a shift between the validation and test score distributions.

Corresponding figures are in `figures/`:
- `asvspoof_cnn_loss_curve.png`, `asvspoof_cnn_accuracy_curve.png` — ASVSpoofCNN training curves
- `asvspoof_cnn_finetuned_loss_curve.png`, `asvspoof_cnn_finetuned_accuracy_curve.png` — fine-tuned ASVSpoofCNN training curves
- `confusion_matrix_asvspoof_cnn.png`, `confusion_matrix_asvspoof_cnn_finetuned.png` — test-set confusion matrices
- `roc_curve.png`, `precision_recall_curve.png` — test-set ROC and precision-recall curves for both models
- `cv_boxplot_fold_metrics.png` — boxplot of the per-fold 5-fold CV scores (accuracy/precision/recall/F1)
- `cv_fold_differences.png` — deviation of each fold from the mean, per metric

| Model | Accuracy | Precision (spoof) | Recall (spoof) | F1 (spoof) | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|
| ASVSpoofCNN | 91.20% | 98.90% | 91.20% | 94.90% | 0.9669 | 8.80% |
| ASVSpoofCNN (fine-tuned) | 91.69% | 98.97% | 91.69% | 95.19% | 0.9676 | 8.31% |

**Headline result — ASVSpoofCNN (fine-tuned):** spoof precision 98.97%, spoof recall 91.69%, accuracy 91.69%, specificity 91.69%, EER 8.31%. Confusion matrix (test set, rows = true, columns = predicted): bonafide 6,744 / 611, spoof 5,306 / 58,576.

**5-fold cross-validation (training split):** accuracy 0.9918 ± 0.0024, precision 0.9993 ± 0.0007, recall 0.9915 ± 0.0031, F1 0.9954 ± 0.0013. The small standard deviations show stable performance across folds. However, held-out test performance (spoof F1 94.90% base / 95.19% fine-tuned) is clearly lower than the CV and validation scores (validation accuracy ≈ 99.9%). This is a generalization gap between the training/validation distribution and the evaluation set, not training instability. Class balance is almost identical across splits, so imbalance does not explain it; a likely contributor is that the ASVspoof2019 LA evaluation partition contains spoofing attacks that are not all present in the training/development data.

**Comparison with published systems (CO4)** — see `results/comparison.csv`, same ASVspoof2019 LA benchmark, EER (lower is better). NR = metric not reported in the cited publication; it was not estimated.

| Model | EER |
|---|---:|
| LFCC + GMM (ASVspoof2019 baseline B1) — classical baseline | 15.80% |
| LFCC + LCNN — CNN-based reference architecture | 9.26% |
| RawNet2 — published deep-learning system | 9.50% |
| AASIST — published deep-learning system (current SOTA on this benchmark) | 0.83% |
| Proposed: ASVSpoofCNN | 8.80% |
| Proposed: ASVSpoofCNN, fine-tuned | 8.31% |

The fine-tuned ASVSpoofCNN (8.31% EER, 97.9K parameters) is well below the classical GMM baseline and slightly below LFCC+LCNN (9.26%) and RawNet2 (9.50%), while being roughly 180x smaller than RawNet2 (about 17.6M parameters in the cited re-implementation). The margin over LCNN and RawNet2 is about 1 percentage point and the systems come from different training setups, so this should be read as "in the same range" rather than a decisive win. It does not approach AASIST (0.83% EER, ~0.30M parameters), which uses a graph-attention architecture purpose-built for anti-spoofing rather than a generic CNN.

**Computational cost** (`results/computational_cost.csv`):

| Tag | Epochs run | Best epoch | Total train time (s) | Avg time/epoch (s) | Device | Parameters |
|---|---:|---:|---:|---:|---|---:|
| ASVSpoofCNN | 22 | 17 | 318.07 | 14.46 | cuda | 97,890 |
| ASVSpoofCNN-FT | 12 | 7 | 172.96 | 14.41 | cuda | 97,890 |

## 8. Innovation
Not attempted in this submission — this project targets **CO3 (Apply)** and **CO4 (Comparative Analysis)** only. No CO5 innovation/ablation claim is made.

## 9. Folder Structure

```text
2430010370/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── asvspoof2019_la_cnn_128mel_4conv_ASVSpoofCNN.ipynb
│
├── results/
│   ├── results.csv
│   ├── comparison.csv
│   ├── dataset_distribution.csv
│   ├── preprocessing_params.csv
│   ├── hyperparameter_tuning_log.csv
│   ├── computational_cost.csv
│   ├── cv_results.json
│   ├── asvspoof_cnn_history.json
│   └── asvspoof_cnn_finetuned_history.json
│
├── models/
│   ├── asvspoof_cnn.pt
│   ├── asvspoof_cnn_finetuned.pt
│   └── model_description.txt
│
└── figures/
    ├── cv_boxplot_fold_metrics.png
    ├── cv_fold_differences.png
    ├── asvspoof_cnn_loss_curve.png
    ├── asvspoof_cnn_accuracy_curve.png
    ├── asvspoof_cnn_finetuned_loss_curve.png
    ├── asvspoof_cnn_finetuned_accuracy_curve.png
    ├── confusion_matrix_asvspoof_cnn.png
    ├── confusion_matrix_asvspoof_cnn_finetuned.png
    ├── roc_curve.png
    └── precision_recall_curve.png
```
