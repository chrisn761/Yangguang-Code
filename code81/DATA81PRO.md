# Current training input: data81pro

This update supersedes old input/output references in README.md.
Copy data81pro (including metadata) and code81 as sibling directories.
Run python run_all_models_10fold.py --epochs 100.
All ten model entrypoints share train81.py. Architectures, splits and optimization are unchanged.
Output: results81_file10fold81pro/model/run_timestamp.
Only completed matching enhanced-data runs are reusable. Raw-data results cannot substitute for enhanced-data training.
The data is the prior 1.0 label-conditioned simulation from data81, not repeated enhancement. IV shifted down, VI up, V unchanged on selected12 channels. Coefficient1.0 is not a p-value or a validated clinical effect size. Source sequences are unchanged in data81.
File-level CV is not patient-independent; missing IV in one fold still produces undefined strict three-class macro metrics. Pooled OOF metrics cover all81 files.
