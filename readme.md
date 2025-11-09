CS-31 Federated Multi-Modal Framework – Runner Pack


Overview
This is the root folder, containing a three-step, notebook-driven workflow for training and evaluating a federated learning framework with a global encoder + local head architecture. 
It supports heterogeneous sites (tabular and/or image), non-IID data, and varying label spaces. 
The two JSON files are the output of Step0, providing necessary information to Step1 and Step 2.


Contents
IPYNB
1. Step0_training_preparations.ipynb
   Pre-flight checks and utilities: Generated ecessary information to Step1 and Step 2 by concluding the all the single JSON files in "/Dataset".
   Then record them in 2 JSON files.
2. Step1_federated_learning_main_control.ipynb
   Main control of global training loop: orchestrates per-site training, server-side aggregation, checkpointing, and logging.
3. Step2_overall_evaluation.ipynb
   Aggregates metrics across sites and groups, calculate final evaluation scores.

JSON
1. sites_info.json
   Site registry and training configuration. Records each site’s modality, data paths, label column, number of classes, and per-site training options.
2. global_tabular_stats.json
   Global statistics for z-score normalisation of tabular features (means and standard deviations computed over a reference corpus).


Quick start
1. Prepare data
   • Ensure each site’s dataset exists and in the correct path under "./Dataset"
   • Finish all site-level preprocessing, with "{:sie_name}_info.json" (for all sites) and "{:sie_name}_stats.json" (for tabular sites only) existing in the site folder.

2. Run notebooks in order
   • Open Step0_training_preparations.ipynb and run all cells 
   -> You will see new "site_info.json" and "global_tabular_stats.json" being generated.
   • Open Step1_federated_learning_main_control.ipynb and run all cells to train and aggregate for the desired number of global rounds.
   -> You will have all trained models in "./SavedModels".
   • Open Step2_overall_evaluation.ipynb and run all cells to generate evaluation scores and tables.

Notes
1. Field names may differ slightly from your implementation; keep the intent the same: identify data, label space, and site-level training/head settings.
2. If a site has only train/val and no test, leave test fields empty or omit them, and evaluate later when available.

Execution details
Device and runtime
1. Colab: enable GPU if training image encoders; tabular-only runs can be CPU-only but will be slower.
2. Local: ensure Python, PyTorch, torchvision (for image sites), pandas/pyarrow (for tabular), and matplotlib are available.

Outputs
1. Checkpoints: per-round global encoder weights and per-site head weights, all stored in "./SavedModels".
2. Figures and tables: site accuracy tracks; accuracy, f-1, AUROC, AUPRC scores; confusion matrices; per-site and group-level summary tables.

Typical workflow
1. Generate JSON files with Step0. You could optionally validate the information and recompute global_tabular_stats.
2. Train with Step1 for R global rounds; inside each round, every site trains for E epochs, then the server aggregates encoders (FedAvg-style).
3. Evaluate with Step2 at site level and group level (by modality and label-space size), produce final report artefacts.

Common errors and fixes
1. mat1 and mat2 shapes cannot be multiplied
Input shape mismatch between an encoder and a head. Re-check img_size, tabular feature template, and head input dimensions. For tabular, ensure feature ordering matches the encoder’s expected input size; labels must not be included as features.
2. File/path not found
Paths in sites_info.json must match your runtime (Drive mount points differ between Colab and local).
3. CUDA out of memory
Reduce batch_size, use gradient accumulation if implemented, or switch to mixed precision if available.
4. Very slow convergence
Increase per-site epochs E or global rounds R; ensure data loaders are not bottlenecked; consider stronger encoders for images.

Reproducibility tips
1. Fix the path and folder name in "./Dataset", and also re-check code of scanning folders in Step0.
2. Record environment details: CUDA, PyTorch, torchvision, pandas, and Colab/OS versions.
3. Keep sites_info.json and global_tabular_stats.json under version control to track configuration drift.


Support
If anything fails in Step1 or Step2, first re-run Step0 to validate configuration and environment. 
If you need us to align this readme with the exact field names in your notebooks, tell us to inspect the code and we will adapt the schema and instructions precisely.
