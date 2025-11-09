readme.md

Title
FinalEvaluations module

Purpose
Run final site-level inference and compute metrics from a standard prediction table. Works for binary and multi-class tasks. The three scripts here are designed to be drop-in for any trained checkpoint set produced by the federated runner.

Files and what they do

1. evaluation_metrics.py
   Input format
   A pandas DataFrame with columns:
   y_true  -> integer ground-truth labels
   y_pred  -> integer predicted labels
   y_prob  -> predicted probabilities
   • binary: a single float per row (probability of the positive class)
   • multi-class: a list/array per row with length equal to the number of classes

   Functions
   site_evaluation_cm(df)     -> numpy.ndarray confusion matrix
   site_evaluation_auroc(df)  -> float AUROC
   • binary: uses y_prob
   • multi-class: roc_auc_score(..., multi_class="ovr", average="macro")
   site_evaluation_auprc(df)  -> float AUPRC (average precision)
   • binary: uses y_prob
   • multi-class: average="macro"
   site_evaluation_acc(df)    -> float accuracy
   site_evaluation_f1(df)     -> float F1
   • binary: default f1_score
   • multi-class: macro F1

2. image_site_prediction.py
   Purpose
   Runs inference for an image site using the shared encoders plus the site’s local head and returns a prediction DataFrame compatible with evaluation_metrics.py.

   Expected inputs
   global_state      state_dict for shared encoders already loaded by the caller
   best_head_path    path to the site’s head checkpoint
   test_set_path     root directory of the test split in ImageFolder layout
   tsfm              torchvision transforms matching the encoder’s input size and normalisation
   n_classes         number of classes
   BATCH_IMAGE       batch size taken from training_config

   Key components
   build_local_model(global_state, n_classes, head_path)
   • builds SharedEncoders(D_TABULAR, D_EMBEDDING, D_FUSION)
   • loads global_state (strict=False)
   • wraps with ImageClientModel and loads the site head if present
   ImageFolderDict
   • datasets.ImageFolder variant that returns {"img": tensor, "label": LongTensor}
   make_predictions(site_name, global_state, test_set_path, tsfm, n_classes, best_head_path) -> DataFrame
   • creates DataLoader over ImageFolderDict(test_set_path, transform=tsfm) with batch_size=BATCH_IMAGE
   • forward pass → logits → softmax probs and argmax preds
   • for binary: y_prob is the positive-class probability; for multi-class: full probability vector
   • concatenates all batches and returns a DataFrame via store_predictions_in_df

   Output format
   DataFrame with columns y_true, y_pred, y_prob (float for binary; list/array for multi-class)

3. tabular_site_prediction.py
   Purpose
   Runs inference for a tabular site, reusing the same preprocessing as training, and returns a prediction DataFrame compatible with evaluation_metrics.py.

   Expected inputs
   global_state       state_dict for shared encoders already loaded by the caller
   best_head_path     path to the site’s head checkpoint
   test_set_path      CSV file for the test split
   labelcol           name of the label column in the CSV
   n_classes          number of classes
   BATCH_TABULAR      batch size taken from training_config

   Key components
   build_local_model(global_state, n_classes, head_path)
   • builds SharedEncoders(D_TABULAR, D_EMBEDDING, D_FUSION)
   • loads global_state (strict=False)
   • wraps with TabularClientModel and loads the site head if present
   TabularOnlyDataset (imported from SiteTrainingFunctions.tabular_site_training)
   • applies the same feature re-ordering and masked z-score normalisation used in training
   • returns {"ehr": FloatTensor, "label": LongTensor}
   make_predictions(site_name, global_state, test_set_path, labelcol, n_classes, best_head_path) -> DataFrame
   • DataLoader over TabularOnlyDataset with batch_size=BATCH_TABULAR
   • forward pass → logits → softmax probs and argmax preds
   • for binary: y_prob is the positive-class probability; for multi-class: full probability vector
   • concatenates all batches and returns a DataFrame via store_predictions_in_df

   Output format
   DataFrame with columns y_true, y_pred, y_prob (float for binary; list/array for multi-class)



Typical usage

Image site

``` python
from FinalEvaluations.image_site_prediction import make_predictions
from FinalEvaluations.evaluation_metrics import (
    site_evaluation_cm, site_evaluation_auroc, site_evaluation_auprc,
    site_evaluation_acc, site_evaluation_f1
)

df_pred = make_predictions(
    site_name="image_3",
    global_state=global_state,                      # loaded best_global_encoders
    test_set_path="Datasets/image_3/test",
    tsfm=my_eval_transforms,                        # resize + normalize
    n_classes=5,
    best_head_path="SavedModels/overall_best_local_heads/image_3.pth"
)

cm    = site_evaluation_cm(df_pred)
auroc = site_evaluation_auroc(df_pred)
auprc = site_evaluation_auprc(df_pred)
acc   = site_evaluation_acc(df_pred)
f1    = site_evaluation_f1(df_pred)
```

Tabular site

``` python
from FinalEvaluations.tabular_site_prediction import make_predictions
from FinalEvaluations.evaluation_metrics import (
    site_evaluation_cm, site_evaluation_auroc, site_evaluation_auprc,
    site_evaluation_acc, site_evaluation_f1
)

df_pred = make_predictions(
    site_name="tabular_4",
    global_state=global_state,                      # loaded best_global_encoders
    test_set_path="Datasets/tabular_4/test.csv",
    labelcol="target",
    n_classes=3,
    best_head_path="SavedModels/overall_best_local_heads/tabular_4.pth"
)

cm    = site_evaluation_cm(df_pred)
auroc = site_evaluation_auroc(df_pred)
auprc = site_evaluation_auprc(df_pred)
acc   = site_evaluation_acc(df_pred)
f1    = site_evaluation_f1(df_pred)
```

Caveats and tips
- For AUROC/AUPRC in binary tasks, y_prob must be the probability of the positive class and both classes must appear in y_true.
- For multi-class AUROC/AUPRC, y_prob must be a same-length vector for every sample; the functions use macro averaging.
- Ensure the evaluation transforms for images match the encoder’s expected input size and normalisation; disable heavy augmentations.
- For tabular data, the evaluation dataset must use the same feature ordering and masked z-score normalisation as training; labels must not be included as features.
- Heads are site-specific and must match n_classes and the encoder representation size used during training.
