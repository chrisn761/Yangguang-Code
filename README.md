# Raw-Data-Driven Hand-Function Classification

This repository contains the training and evaluation code for the raw-data-driven hand-motion classification experiments described in the manuscript **“A Raw Data-Driven Markerless Smart Sensing System for Digital Screening of Hand Function: A Pilot Study.”**

The code implements the GEL model, two internal GEL ablation models, and five external comparison models for three-class classification of Brunnstrom recovery stages IV, V, and VI.

## Data availability and ethics

The repository does **not** include raw clinical motion recordings, identifiable participant information, or fold-specific model checkpoints. The clinical dataset was collected under medical-ethics and data-management requirements and can only be accessed through an appropriate data request and approval process. Researchers who need access should contact the corresponding authors and provide a scientifically justified request subject to the applicable institutional and ethical requirements.

## Repository structure

```text
.
├── code81/
│   ├── train81.py
│   ├── models.py
│   ├── light_models.py
│   ├── revised_models.py
│   ├── run_all_models_10fold.py
│   ├── train_GEL_10fold.py
│   ├── train_GEL_no_attention_10fold.py
│   ├── train_GEL_no_residual_10fold.py
│   ├── train_GRU_10fold.py
│   ├── train_LSTM_10fold.py
│   ├── train_MSCNN_LSTM_10fold.py
│   ├── train_CNN_LSTM_10fold.py
│   └── train_Transformer_10fold.py
├── public_results/
│   └── *_pooled_10fold_metrics.csv
├── requirements.txt
└── README.md
```

`code81/` contains the complete model and cross-validation implementation. `public_results/` contains one pooled out-of-fold metric file for each evaluated model. The pooled files summarize predictions from all ten outer folds; they are not individual-fold checkpoints.

## Models and code entry points

| Model | Role | Entry point | Main implementation |
|---|---|---|---|
| GEL | Complete proposed model | `train_GEL_10fold.py` | `models.py` |
| GEL without attention | Internal ablation; GAM removed and masked mean pooling used | `train_GEL_no_attention_10fold.py` | `models.py` / `revised_models.py` |
| GEL without residual | Internal ablation; residual additions removed while recurrent blocks and attention are retained | `train_GEL_no_residual_10fold.py` | `models.py` / `revised_models.py` |
| GRU | External recurrent comparison | `train_GRU_10fold.py` | `revised_models.py` |
| LSTM | External recurrent comparison | `train_LSTM_10fold.py` | `revised_models.py` |
| MSCNN-LSTM | External hybrid comparison | `train_MSCNN_LSTM_10fold.py` | `revised_models.py`, `light_models.py` |
| CNN-LSTM | External hybrid comparison | `train_CNN_LSTM_10fold.py` | `light_models.py` |
| Transformer | External attention-based comparison | `train_Transformer_10fold.py` | `revised_models.py`, `light_models.py` |

`run_all_models_10fold.py` launches all configured model entry points sequentially. The scripts use variable-length sequences with padding and length-aware processing; sequences are not divided into independent windows for cross-validation.

## Network configurations

| Model | Principal architecture |
|---|---|
| GEL | 81-to-64 input projection; two-layer bidirectional LSTM with 64 hidden units per direction; two 128-dimensional Residual LSTM blocks; 128-to-1 temporal attention projection; masked weighted pooling; 128-to-64-to-32-to-3 classification head |
| GEL without attention | GEL recurrent backbone without the attention module; masked mean pooling; classification head without attention aggregation |
| GEL without residual | GEL recurrent and attention pathway with residual additions removed from the Residual LSTM blocks |
| GRU | 81-to-64 projection; two-layer bidirectional GRU with 64 hidden units per direction; masked mean pooling; 128-to-64-to-3 head |
| LSTM | 81-to-64 projection; two-layer bidirectional LSTM with 64 hidden units per direction; length-aware temporal pooling; 128-to-64-to-3 head |
| MSCNN-LSTM | Three convolution branches with kernel sizes 3, 5, and 7; 8 channels per branch; single LSTM with 32 hidden units; 32-to-16-to-3 head |
| CNN-LSTM | One 16-channel convolution branch with kernel size 5; single LSTM with 32 hidden units; 32-to-16-to-3 head |
| Transformer | 81-to-64 projection; two Transformer encoder layers; four attention heads; feed-forward dimension 256; local 128-frame processing; mean pooling; 64-to-32-to-3 head |

## Common training and evaluation settings

| Setting | Value |
|---|---|
| Classification task | Three classes: BRS IV, V, and VI |
| Cross-validation | Stratified ten-fold cross-validation at the complete assessment-file/instance level |
| Inner validation | 20% of each outer-development set, stratified by class, for checkpoint selection |
| Outer evaluation | Each complete assessment instance is used as outer validation once; pooled out-of-fold predictions are reported |
| Input | 81-channel preprocessed raw coordinate sequence |
| Sequence handling | Variable-length sequences, padding and length-aware recurrent processing; no truncation |
| Maximum epochs | 100 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Loss | Cross-entropy with label smoothing of 0.1 during model fitting |
| Checkpoint selection | Lowest inner-validation cross-entropy |
| Learning-rate schedule | Cosine annealing with warm restarts, `T_0=10`, `T_mult=2` |
| Dropout | 0.3 |
| Gradient clipping | Maximum norm 1.0 |
| Random seed | 42 for the reported split configuration |

Accuracy, Balanced Accuracy, macro-averaged metrics, weighted metrics, class-wise metrics, AUROC, and confusion matrices are calculated from the out-of-fold predictions. Balanced Accuracy is the unweighted mean recall across the three classes. Macro metrics give equal importance to each class, whereas weighted metrics use the observed class frequencies.

## Running the code

1. Create a Python environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

2. Obtain the approved dataset through the data-access procedure and place the approved data directory and its metadata beside `code81/`, following the path configuration in `train81.py`.

3. Run one model:

```bash
python code81/train_GEL_10fold.py --epochs 100
```

4. Run all configured models:

```bash
python code81/run_all_models_10fold.py --epochs 100
```

Results are written to the configured results directory. Each run records the protocol, architecture, fold assignments, selected checkpoints, out-of-fold predictions, pooled metrics, and confusion matrices. Only the compact pooled metric files are included in this public release.

## Interpretation of the released results

The released pooled files are provided for transparency and reference. They summarize the corresponding archived ten-fold run and should be interpreted together with the assessment-instance-level validation protocol and the manuscript limitations. Repeated assessments from the same participant are algorithmic assessment instances and should not be interpreted as biologically independent patients. Participant-level generalization requires future grouped validation with a larger clinical cohort.

## Citation

If you use this code, please cite the associated manuscript and acknowledge the code release. The repository URL will be inserted here after the GitHub repository is created:

`https://github.com/REPLACE_WITH_REPOSITORY`
