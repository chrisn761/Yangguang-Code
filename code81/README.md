# 81 files / 27 simulated instances

## Current revision (supersedes older architecture notes below)
Input remains unenhanced data81. Outer file-stratified tenfold + inner20% holdout, minimum inner CE selects checkpoint. All optimization settings unchanged, including the current CosineAnnealingLR (not warm restarts).
GEL, CNN-LSTM Compact and GEL-CNN-attention are unchanged. Completed existing runs are reused only after config, file hashes, exact inner/outer membership, architecture and checkpoint compatibility checks. Revised models run into new timestamp folders; no prior results are deleted.
GRU: 2-layer bidirectional64. LSTM and BOTH GEL module-removal variants: identical 2-layer bidirectional LSTM64 + mean pooling. These are plain-LSTM comparison variants, NOT pure attention/residual removal. Legacy '-simple' identifiers are retained for launcher compatibility; architecture.txt records the current actual structure.
Transformer: 2 encoder layers, dim64, 4 heads, FF256, sinusoidal positions; full sequence covered by local128-frame windows. Not global full-record attention.
MSCNN-LSTM: original compact 3/5/7 branches (8 channels each), LSTM reduced32 to16, single unidirectional layer, mean pooling.
GEL-MSCNN-attention: same revised MSCNN backbone with linear16-to1 GEL attention instead of mean pooling. No residual blocks. No guaranteed performance ranking.
Copy the ENTIRE code81 directory. New launcher: train_GEL_MSCNN_attention_10fold.py. run_all_models_10fold.py automatically includes all ten models.
Current output remains results81_file10fold/model/run_timestamp. Prior and revised versions coexist: use architecture.txt/protocol.json to distinguish them.

Run: python run_all_models_10fold.py --epochs 100
Single: python train_GEL_10fold.py --epochs 100
Copy code81 and data81 (including metadata) as sibling folders to another computer. Output: sibling results81_file10fold/model/run_timestamp. Old results81 and results81plus1.0 are untouched.
All model entrypoints now use the 81 unenhanced files. StratifiedKFold is applied directly to files. The three files carrying one SimXX label may cross folds by design. No extra training noise, augmentation or truncation. Same training defaults for all models.
GEL is the existing full original architecture; GEL-CNN-attention is separately included, not substituted.
Both simplified ablations and LSTM use the same single unidirectional LSTM32 architecture. They are structurally identical, not independent single-factor ablations. Same seed can produce identical results.
GRU: single unidirectional GRU32. CNN and MSCNN: compact 16 / 3x8 channels plus single LSTM32. Transformer: one layer, width32, two heads, local128-frame attention.

Ten outer folds are stratified and preassigned by file. Each outer validation fold contains 8 or 9 files, and every file is in outer validation exactly once. Within each outer-development set, 20% is split into inner validation. The selected epoch is the minimum inner cross-entropy epoch; only that checkpoint is evaluated on the outer validation files. Training history logs train and inner metrics each epoch.
IV has only 3 instances, V 9, VI 15. Most folds lack IV. Strict three-class macro metrics are NaN in incomplete folds. Mean/std tables include count: these are AVAILABLE-fold means, not necessarily ten-fold means. Do not label a mean over three valid folds as a ten-fold three-class BA.
pooled_file81 contains all 81 OOF predictions; pooled_instance27 averages three file probabilities per instance before classification. Both have complete three-class metrics and 3x3/three 2x2 confusion matrices (CSV/PNG).
Per-class accuracy=(TP+TN)/N is one-vs-rest accuracy. class_accuracy_recall=TP/(TP+FN) is the true-class correct rate. Per-class balanced_accuracy is binary one-vs-rest BA. Macro metrics use three classes.
These are 81 repeated preprocessed observations arranged under 27 SimXX labels, not 81 independent patients and not confirmed patient-independent validation. Because repeated files may cross folds, this is file-level validation and can be optimistic. Original-source overlap is saved per fold. Never relabel these results as patient-independent or clinical performance.
No model ranking is guaranteed. Source data and older results are untouched. Completed identical code/config runs are skipped. Incomplete runs start a new directory.
