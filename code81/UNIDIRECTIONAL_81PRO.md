# Revision3: seven-model rerun

Run python run_selected_unidirectional_81pro.py --epochs 100.
Input: data81pro. Output: results81_file10fold81pro/model/run_timestamp.
This runner deliberately forces new training for seven models, including already-unidirectional CNN/MSCNN. Existing results remain untouched. Each invocation starts fresh runs.
Single model: python train_GRU_10fold.py --epochs 100 --force-retrain. The other existing per-model entrypoints work identically.
GEL-no-attention-simple, GEL-no-residual-simple and LSTM: two-layer unidirectional LSTM64 with mean pooling, identical architectures. GRU: two-layer unidirectional GRU64. Transformer: one layer width32, two heads, FF64, local128 windows. CNN-LSTM remains one-layer unidirectional LSTM32; MSCNN remains one-layer unidirectional LSTM16.
GEL, GEL-CNN-attention and GEL-MSCNN-attention are NOT included in the runner and their architectures are unchanged.
This supersedes older bidirectional architecture descriptions. No noise or additional data manipulation; no guaranteed ranking. Artificial label-conditioned data and file-level splits are not clinical or patient-independent evidence. These replacement-LSTM variants are not strict single-factor ablations.
