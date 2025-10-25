# overall model shape settings
D_TABULAR = 21 # should be 16
# the total number of features in the input data 
# manually set, based on the preprocessing outcome tamplate, for 8 features + 8 masks

D_IMG = 224
# all images are resized to by tsfm be square
# the square of D_IMG * D_IMG
# (seemd not being used yet...)

D_EMBEDDING = 128 
# should be changed based on the result of encoders
# always keep the same for both MLP and CNN

D_FUSION = D_EMBEDDING
# the dim of fusion outcome (output)
