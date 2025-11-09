Datasets

purpose
This folder holds all site datasets (image and tabular) plus small helper scripts and notebooks that prepare metadata and global templates. It is organised by site name (for example, image_1, tabular_1) so each site can be trained and evaluated independently in the federated workflow.

high-level layout

* image_1/
  * train/ val/ test/    image splits in torchvision ImageFolder format (one subfolder per class)
  * image_1_info.json    site metadata (class names/order, counts, paths; optional notes)
  * image_1_tsfm.py      the evaluation/training transforms used for this site (torchvision)
  * step_3_store_info_in_json.ipynb  utility notebook that writes/updates the info json

* image_3/
  mirrors the same structure as image_1 for another image site

* tabular_1/
  * diabetes_012_train.csv / val.csv / test.csv   split datasets ready for loaders
  * diabetes_012_ready_to_model.csv               preprocessed full table (before split)
  * diabetes_012_health_indicators_BRFS…csv       original/raw source
  * feature_template_v1.json                      early per-site feature schema (historical)
  * tabular_1_info.json                           site metadata (paths, label column, n_classes, notes)
  * tabular_1_stats.json                          per-site stats if needed (counts, basic moments)
  * step_1_basic_preprocessing.ipynb              cleaning/typing/feature engineering
  * step_2_template_reshape_&_split.ipynb         align to global template, split to train/val/test
  * step_3_store_info_in_json.ipynb               writes/updates the info json

* tabular_3/ , tabular_4/ , tabular_rex/
  additional tabular sites; each mirrors the tabular_1 pattern (csv splits, site info json, and any prep notebooks/files specific to that site)
  note the prep notebooks are coded by different members, therefore might be different from each other.

* TabularTemplateBuilding/
  * tabular_feature_template_creation.ipynb       builds the global feature + mask order used by all tabular sites
  * tabular_feature_template.json                 the canonical global feature template consumed by loaders
  * feature_mean_var_and_samplecount.ipynb        computes global means/stds and counts (supporting z-score normalisation across sites)

shared utilities
* image_resize_with_padding.py                    helper script to letterbox images to a square size without distortion
* sample_template_specification.txt               short note describing the sample I/O contracts used by the project
* tabular_feature_template.json                   stores the template. can be directly used in preprocessing if suitable


conventions
* image sites follow ImageFolder layout under train/ val/ test/ with class subfolders. Class index order follows the subfolder sorting unless overridden by the site’s image_*_info.json.
* tabular sites expose split CSVs and declare their label column and class count in tabular_*_info.json. All tabular features are later reordered to match TabularTemplateBuilding/tabular_feature_template.json; mask columns (feature_mask) are included for missingness.
* global statistics for z-score normalisation are stored at project root in global_tabular_stats.json (computed from the template notebooks) and used by loaders; individual tabular_*_stats.json may also exist for per-site reference.


adding a new site
* image: create image_X/ with train/ val/ test/ in ImageFolder format; add image_X_info.json and, if needed, image_X_tsfm.py.
* tabular: create tabular_X/ with train/ val/ test CSVs and tabular_X_info.json (must include label column name and n_classes). Ensure your features can be reordered to the global template; if a feature is missing, the loader will fill with zeros and use the mask.



notes
* [IMPORTANT] all site folders are well prepared, which means: 1. cleaned and split 2. with prepared "<site_name>_info.json" and "<site_name>_stats.json", which enables you to directly run the main step (Step0, Step1 and Step2) in the root folder.

* [IMPORTANT] Each time you add or check a site, to make sure they are: 
    - cleaned
    - strictly reshaped as the format 
    - split 
    - record info and stats JSON correctly.
If something in the Step0, 1, 2 behaves abmormally, always check whether the above 4 things were correct.

* keep file names stable once referenced in sites_info.json so training notebooks can discover data reliably.

* avoid heavy data augmentation in evaluation transforms; keep resize and normalisation consistent with training.

* if you regenerate the global template or global stats, re-run Step0 in the main workflow so all downstream code picks up the changes.
