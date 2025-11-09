SiteTrainingFunctions script folder

Purpose
This folder contains the site-level training logic used by the federated runner. 
Each script encapsulates how a single site prepares data, builds its local head, trains against the broadcast global encoders, and returns updates and metrics to the server loop.

Files
1. image_site_training.py
- Training utilities for image sites. 
- Builds image data loaders, applies transforms, runs per-epoch optimisation for the site’s local head while using the shared global image encoder. 
- Exposes callable entry points for “train one site for N epochs” and “evaluate one site” (but normally only called by the "train for N spoch" loop).

2. tabular_site_training.py
- Training utilities for tabular sites. Loads tabular datasets, applies z-score normalization using global_tabular_stats.json, aligns features to the template, and trains the site’s local head against the shared global tabular encoder. 
- Provides the same entry points as the image counterpart.

3. training_config.py
Centralised defaults for optimisation and runtime knobs used by both trainers: batch size, epochs per round, learning rate and scheduler settings.


Typical call flow
All these scripts are called by Step1_federated_learning_main_control.ipynb
1. The federated controller broadcasts best_global_encoders to the site.
2. The site trainer loads data, constructs the matching local head, and trains for E epochs.
3. The trainer tracks the accuracy and saves newest_local_heads/<site>.pth after the last epoch
4. The trainer returns the updated global model, the accuracy track through epochs, best accuracy and sample count to the caller.

Expected interfaces
tabular_site_training.py:
1. Utilities and constants
    set_seed(seed=42)
        Sets Python/NumPy/PyTorch seeds and configures cuDNN for determinism.
    TABULAR_STATS_PATH = "global_tabular_stats.json"
        Path to global z-score statistics used by the tabular pipeline.
    TABULAR_FEATURE_TEMPLATE_PATH = "Datasets/TabularTemplateBuilding/tabular_feature_template.json"
        Loads a fixed feature ordering: features + masks.
2. Data preprocessing:
    apply_masked_znorm(df, stats_path, core_feats=None) -> pd.DataFrame:
        Applies z-score normalisation per feature using global stats, but only on rows where f_mask == 1; fills 0 where mask == 0; ensures missing cols exist.
    reorder_features(df, feature_template=TABULAR_FEATURE_TEMPLATE, label_col=None) -> pd.DataFrame
        Reorders feature columns to the global template while keeping the label column intact (label appended at the end). Missing features are added and filled with 0; extra non-label features are ignored with a warning.
3. Dataset:
    class TabularOnlyDataset(csv_path, label_col, stats_path=TABULAR_STATS_PATH)
        Reads a CSV, reorders features, applies masked z-score normalisation, and exposes samples as {"ehr": FloatTensor[d_tab], "label": LongTensor[]}.
4. Model construction
    build_local_model(global_state, n_classes, head_path) -> torch.nn.Module
        Creates SharedEncoders with dims (D_TABULAR, D_EMBEDDING, D_FUSION), loads encoder weights from global_state (strict=False), wraps with TabularClientModel(n_classes), and optionally loads a saved local head from head_path. Returns the full client model.
5. Metrics
    accuracy_from_logits(logits, labels) -> float
        Mean accuracy for a batch.
6. Training/evaluation loops
    evaluation_in_training(model, val_loader) -> (val_loss, val_acc, num_batches)
        Runs eval mode over val_loader, computes cross entropy and accuracy, and returns averages over batches.
    train_one_epoch(model, loader, opt) -> (avg_loss, avg_acc, num_samples)
        Full logging variant: computes weighted accuracy across the epoch.
    train_one_epoch_simple(model, loader, opt) -> num_samples
        Minimal variant used in your main loop; returns only the number of seen samples.
7. Entry point used by the federated controller
    training(site_name, global_state,freeze_global,train_set_path,val_set_path,labelcol,n_classes,newest_head_path,current_best_head_path) -> (updated_tabular_state_dict, sample_count, ckpt_eva_results, perf_track)
    Behaviour
        - Builds a TabularClientModel with SharedEncoders initialised from global_state and an attached site head.
        - If freeze_global is True, freezes enc.parameters() and optimises only model.head with AdamW(lr=LR_TABULAR, weight_decay=WD_TABULAR). Otherwise optimises the whole model.
        - Constructs train/val DataLoaders from CSVs using TabularOnlyDataset and BATCH_TABULAR.
        - Trains for EPOCHS_TABULAR epochs; after each epoch evaluates on val, tracks val accuracy, and if improved: saves model.head.state_dict() to current_best_head_path
        - updates ckpt_eva_results = {"val_loss": ..., "val_acc": ..., "num_samples": ...}
    After training:
        - returns model.enc.tabular_enc.state_dict() as the encoder update for FedAvg
        - saves the final head to newest_head_path
        - returns total sample_count seen in training, ckpt_eva_results of the best epoch, and perf_track (list of val accuracies per epoch).

Notes linking to configs
- Uses LR_TABULAR, WD_TABULAR, BATCH_TABULAR, EPOCHS_TABULAR from SiteTrainingFunctions.training_config.
- Z-score normalisation relies on global_tabular_stats.json fields "global_feature_mean" and "global_feature_std".
- Feature order is enforced by Datasets/TabularTemplateBuilding/tabular_feature_template.json via keys "features" and "masks".

Minimal usage sketch
``` python
updated_tabular, sample_cnt, best_ckpt_metrics, perf_track = training(
    site_name="tabular_4",
    global_state=server_encoders_state,
    freeze_global=True,
    train_set_path="Datasets/tabular_4/train.csv",
    val_set_path="Datasets/tabular_4/val.csv",
    labelcol="target",
    n_classes=3,
    newest_head_path="SavedModels/newest_local_heads/tabular_4.pth",
    current_best_head_path="SavedModels/current_best_local_heads/tabular_4.pth"
)
```

image_site_training.py:
1. Configs and imports:
    Uses VAL_RATIO_IMAGE, EPOCHS_IMAGE, BATCH_IMAGE, LR_IMAGE, WD_IMAGE from SiteTrainingFunctions.training_config.
    device = "cuda" if available, else "cpu".
    Depends on SharedEncoders and ImageClientModel plus D_TABULAR, D_EMBEDDING, D_FUSION.
2. Model construction
    build_local_model(global_state, n_classes, head_path) -> torch.nn.Module
        - Instantiates SharedEncoders(d_tabular=D_TABULAR, d_embedding=D_EMBEDDING, d_fusion=D_FUSION).
        - Loads encoder weights from global_state (strict=False).
        - Wraps with ImageClientModel(shared_encoders=..., n_classes=n_classes).
        - If head_path exists, loads the local head state dict (strict=False).
        Returns the full client model.
3. Dataset and loaders
    class ImageFolderDict(datasets.ImageFolder)
        **getitem** returns {"img": transformed_image_tensor, "label": LongTensor} for consistency with tabular.
    build_loaders(train_set_path, val_set_path, tfms, batch_size=BATCH_IMAGE, workers=4)
        Creates ImageFolderDict datasets with the provided transforms tfms; returns train_loader and val_loader with pin_memory=True and shuffle=True/False for train/val.
3. Training and evaluation loops
    train_one_epoch(model, loader, optimizer, device, log_every=50) -> (avg_loss, avg_acc, num_samples)
        Full variant computing weighted average loss and accuracy across the epoch.
    train_one_epoch_simple(model, loader, optimizer) -> num_samples
        Minimal variant used by the main training loop; returns only number of samples processed
    evaluation_in_training(model, loader, device) -> (val_loss, val_acc, num_samples)
        No-grad evaluation over the validation set; computes cross-entropy and accuracy, returns weighted averages.
4. Entry point used by the federated controller
    training(site_name,global_state,freeze_global,train_set_path,val_set_path,tsfm,n_classes,newest_head_path, current_best_head_path) -> (updated_image_state_dict, sample_count, ckpt_eva_results, perf_track)
    Behaviour
        1. Logs start message and initialises perf_track.
        2. Builds the client model via build_local_model and moves it to device.
        3. If freeze_global is True:
            - sets enc parameters requires_grad=False
            - optimiser = Adam(model.head.parameters(), lr=LR_IMAGE, weight_decay=WD_IMAGE)
            Else: 
            - optimiser = Adam(model.parameters(), lr=LR_IMAGE, weight_decay=WD_IMAGE)
        4. Builds train/val DataLoaders with build_loaders(train_set_path, val_set_path, tsfm, batch_size=BATCH_IMAGE, workers=4).
        5. For epoch in 1..EPOCHS_IMAGE:
            - trains with train_one_epoch_simple (returns samples only)
            - evaluates with evaluation_in_training
            - appends v_acc to perf_track and accumulates sample count
            - checkpoint selection is loss-based: if val_loss decreased by more than 1e-6, saves model.head to current_best_head_path and updates ckpt_eva_results = {"val_loss": v_loss, "val_acc": v_acc, "num_samples": v_sp}
            - prints a short “[best updated] acc: …” line when improved
        6. After epochs:
            - returns model.enc.image_enc.state_dict() as the site’s encoder update for aggregation
            - saves the final head to newest_head_path
            - prints done message
    Returns
        1. updated_image: state dict of image_enc after local training
        2. sp_count: total number of training samples processed this round
        3. ckpt_eva_results: metrics at the best validation-loss checkpoint
        4. perf_track: list of per-epoch validation accuracies

Assumptions and requirements
- tsfm is a torchvision.transforms.Compose matching the encoder’s expected input size and normalisation.
- Directory structure under train_set_path and val_set_path must follow ImageFolder conventions (one subfolder per class).
- Head is site-specific and must match n_classes; encoder dimensionality must remain consistent across rounds.

Minimal usage sketch
``` python
updated_image, sample_cnt, best_ckpt, perf_track = training(
    site_name="image_3",
    global_state=server_encoders_state,
    freeze_global=True,
    train_set_path="Datasets/image_3/train",
    val_set_path="Datasets/image_3/val",
    tsfm=my_transforms,          # e.g., Resize/Normalize matching the encoder
    n_classes=5,
    newest_head_path="SavedModels/newest_local_heads/image_3.pth",
    current_best_head_path="SavedModels/current_best_local_heads/image_3.pth"
)
```
Notes
- Checkpoint criterion differs from the tabular trainer (loss vs accuracy); keep this intentional or align both if you want uniform selection logic.
- If you later switch to mixed precision, add autocast/GradScaler around the forward/backward where applicable.



Common pitfalls
• Shape mismatch between encoder output and head input; confirm representation size (rep_dim) is consistent.
• Non-deterministic results when seeds are not set; set seeds in Step0 and in training_config.py.
• Dataloader bottlenecks on Colab; tune num_workers and persistent_workers if available.
• Metric drift when class imbalance is severe; choose a selection metric aligned with the task (macro-F1 or AUROC).


