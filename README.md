# Anonymous artifact: reentrancy detection with a multi-scale GAT
Contents
- hasil/: per-run outputs. For every configuration and seed: fold metrics, validation predictions (with fold and contract_id), raw benchmark predictions (probability of every fold model and their mean), and the run configuration (JSON). Also SHAP outputs, the near-duplicate (containment) measurement, and a nondeterminism diagnostic.
- data/benchmark/: the 16 benchmark contracts (8 vulnerable, 8 fixed with only the statement order changed).
- data/manifest_kontrak_latih.csv: training graphs after deduplication, with label, group (contract_id), size and SHA-256 of the source file. Training sources are from the public Hugging Face dataset mwritescode/slither-audited-smart-contracts and are not redistributed.
- Model weights are not included and can be provided on request.
Splitting unit: the contract. All graphs from one contract (whole graph and function subgraphs) are in the same fold (StratifiedGroupKFold, 5 folds).
Benchmark rule: a pair is correct if the mean probability of the vulnerable version is strictly greater than that of the fixed version.

Notes
- Four safe contracts of data version v2 have no source file in the original archive and therefore have an empty sha256_berkas field in the manifest.
- Paths in scripts were anonymized (the project folder is named PROJECT).

Notebooks (run in this order, on Google Colab)
- notebook/01_data_v1.ipynb: CFG extraction with Slither, conversion to PyTorch Geometric, pruning (data version v1).
- notebook/02_data_v2.ipynb: size-balanced sampling of safe contracts (data version v2).
- notebook/03_data_v2_fitur_posisi.ipynb: position feature variant of v2 (configuration C4).
- notebook/04_data_v3_subgraf.ipynb: per-function labeling and subgraph extraction (data version v3).
- notebook/05_dedup_pelatihan_evaluasi.ipynb: deduplication, group-wise cross-validation, all training runs, benchmark evaluation, SHAP, near-duplicate measurement, and diagnostics.
Remarks: cells that report results of earlier experiments (before the group split) may remain in notebooks 01 to 04 and are superseded by notebook 05. References to other notebook file names in the text cells are historical and those notebooks are not part of this artifact.

Notebooks (run in this order, on Google Colab)
- notebook/01_data_v1.ipynb: CFG extraction with Slither, conversion to PyTorch Geometric, pruning (data version v1).
- notebook/02_data_v2.ipynb: size-balanced sampling of safe contracts (data version v2).
- notebook/03_data_v2_fitur_posisi.ipynb: position feature variant of v2 (configuration C4).
- notebook/04_data_v3_subgraf.ipynb: per-function labeling and subgraph extraction (data version v3).
- notebook/05_dedup_pelatihan_evaluasi.ipynb: deduplication, group-wise cross-validation, all training runs, benchmark evaluation, SHAP, near-duplicate measurement, and diagnostics.
Remarks: cells that report results of earlier experiments (before the group split) may remain in notebooks 01 to 04 and are superseded by notebook 05. References to other notebook file names in the text cells are historical and those notebooks are not part of this artifact.

- docs/: summary tables (final results table and per-pair validation of the fixed contracts) in spreadsheet form.
