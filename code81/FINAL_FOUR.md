# Corrected final four-model run

Run: python run_four_final.py --epochs 100

Output is saved under results81_file10fold81proFinal/model/run_timestamp. Existing results in other folders are untouched.

The four models are structurally distinct:

- GEL-unidirectional: projection + 2-layer UniLSTM64 + 2 residual UniLSTM64 blocks + scalar temporal attention.
- GEL-no-attention-final: same projection/recurrent/residual backbone, but masked mean pooling instead of attention.
- GEL-no-residual-final: same projection/recurrent/attention backbone, but no residual addition in the two blocks.
- LSTM-final: projection + one-layer UniLSTM64 + masked mean pooling + smaller baseline head.

All use the same data81pro, outer file-level StratifiedKFold, inner 20 percent validation for epoch selection, and optimizer settings. The first three are controlled variants; LSTM-final is an independent baseline. Artificial signal data and repeated-file validation are not clinical or patient-independent evidence.
