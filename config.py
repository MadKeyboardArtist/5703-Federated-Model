# some overall settings
D_TABULAR = 21
# the total number of features in the input data 
# should be read from the file after preprocessing
D_EMBEDDING = 128 
# should be changed based on the result of encoders
# always keep the same for both MLP and CNN
D_FUSION = 128 # TBA

# training configs, for all modalities
VAL_RATIO = 0.2
EPOCHS    = 5
BATCH     = 64
LR        = 3e-4
WD        = 1e-4
