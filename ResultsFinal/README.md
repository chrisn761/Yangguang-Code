# Final model results

This folder contains the selected results for the eight models reported in the manuscript. The results were obtained from 27 stroke patients who completed three standardized assessments each, yielding 81 complete assessment instances. The classification task includes BRS IV, V, and VI.

## Model folders

Each model folder contains:

- `fold_metrics.csv`: original ten-fold metrics supplied with the selected run.
- `fold_metrics_macro_weighted.csv`: fold-level Accuracy, Balanced Accuracy, macro metrics, weighted metrics, macro AUROC, and weighted AUROC.
- `pooled_metrics_macro_weighted.csv`: pooled out-of-fold results across all 81 assessment instances.
- `pooled_metrics.csv`: original pooled metric file.
- `pooled_class_metrics.csv`: pooled class-wise performance for BRS IV, V, and VI.
- `pooled_confusion_3class.csv`: pooled three-class confusion matrix.

The weighted metrics use the observed class frequencies. Weighted AUROC was calculated from the pooled out-of-fold class probabilities and the corresponding class supports.

## Pooled results

The file `summary_pooled_macro_weighted.csv` provides the pooled results for all eight models. The file `summary_fold_macro_weighted.csv` provides the corresponding results for each outer fold.

These results are assessment-instance-level results and should not be interpreted as participant-independent validation. Repeated assessments from the same patient may occur in different folds.

## Data access

The raw clinical motion data are not included because their release is restricted by medical-ethics and data-management requirements. Access may be requested from the corresponding author subject to the applicable institutional and ethical approval procedures.

Code repository: https://github.com/chrisn761/Yangguang-Code
