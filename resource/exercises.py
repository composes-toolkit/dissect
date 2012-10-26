All the input data can be found in:
DATA_PATH = /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial

PART 1 - build_core_space.py
1. Load the space saved in DATA_PATH/CORE_SS.verbnoun.core.pkl
2. Apply ppmi weighting on it
3. Select the top 2000 features
4. Apply SVD-100 on it
5. Output the space in pickle format (.pkl) as well as in .dm format
6. Look at the output and at the log file

PART 2 - build_peripheral_space.py
1. Create a peripheral space using the counts in DATA_PATH/per.raw.SV.sm,
DATA_PATH/per.raw.SV.rows, DATA_PATH/per.raw.SV.cols and the core space from 1
2. Output it as pickle

PART 3 - train_composition.py
1. Load training data from file DATA_PATH/ML08_SV_train.txt
2. Load argument space from part 1 and phrase space from part 2
3. Train a lexical function model on it using Ridge Regression with lambda=2
4. Save this trained model to a file

PART 4 - apply_composition.py
1. Load the saved model from PART 3
2. Load data from DATA_PATH/ML08nvs_test.txt (list of elements to be composed)
3. Load an argument space, the space from PART 1
4. Apply compose on this and save the result space

PART 5 - compute_similarities.py
1. Load the composed space you have just saved in part 4
2. Use this space to compute the similarities of the pairs in 
DATA_PATH/ML08data_new.txt. (Columns 1 and 2 contain the pairs of words.)

PART 6 - evaluate_similarities.py
1. Evaluate the similarities. DATA_PATH/ML08data_new.txt contains the gold 
standard scores in column 3.


