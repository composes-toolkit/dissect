LARGER DATA

PART 1 - build_core_space.py
1. Load the space saved in /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/CORE_SS.verbnoun.core.pkl
2. Apply ppmi weighting on it
3. Select the top 2000 features
4. Apply SVD-100 on it
5. Output the space in pickle format (.pkl) as well as in .dm format
6. Look at the output and at the log file

PART 2 - build_peripheral_space.py
1. Create a peripheral space using the counts in /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/per.raw.SV.sm
and /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/per.raw.SV.cols, and the core space from 1
2. Output it as pickle

PART 3 - train_composition.py
1. load training data from file /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08_SV_train.txt
2. load argument space from part 1 and phrase space from part 2
3. train lexical function model on it
4. save this trained model to a file

PART 4 - apply_composition.py
1. load the saved model from 3
2. load data from /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08nvs_test.txt (list of elements to be composed)
3. load an argument space,the space from PART 1
4. apply compose on this and save the result space,(also in .dm format)

PART 5 - compute_similarities.py
1. load the composed space you have just saved
2. use this space to compute the similarities of the pairs in /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08data_new.txt

PART 6 - evaluate_similarities.py
1. evaluate the similarities, the file /mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08data_new.txt contains the gold in column 3



- document how to set a logger path
- document io functions for reading data
- document space (get rows... and so on)
- link the documentation fo matrix
- create some examples.... show it from a script


