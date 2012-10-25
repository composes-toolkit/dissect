LARGER DATA

PART 1 - creating core spaces
0. Turn the logger on and set an output file
1. Load the space saved in ..
2. Apply ppmi weighting on it
3. Select the top 1500 features
4. Apply SVD-100 on it
5. Output the space in pickle format (.pkl) as well as in .dm format
6. look at the output and at the log file

PART 2 - similarity computations
0. Load the space you have just saved (from either of the two formats)
1. Compute the similarity between the words in file... 
2. Look at the output
3. Compute the neighbours of the words in file...
4. look at the output

PART 3 - peripheral spaces
0. Load the peripheral input data in .. (contains only counts!)
1. Create a peripheral space from it, using the core space from 1
2. Output it as pickle and as dm

PART 4 - train a weighted additive compositon model
0. load training data from file 
1. load argument space from part 1 and phrase space from part 3
2. load test data from ...

2. train weighted additive model on this data
3. save it to a file and export its parameters
4. look at its paramters
5. apply it to the test data...
6. look at the result

PART 4 - train a weighted additive compositon model
0. load training data from file 
1. load argument space from part 1 and phrase space from part 3
2. load test data from ...

7. train lexical function model on it
8. save it to a file
9. get the similarity between "" and "" as lexical funtions
10. apply it to the test data
11. look at the result

================
SCORE THE RESULT
================











USEFUL EXAMPLES:
- how to activate the logger
- print the shape of a matrix ... link the Matrix class doc
- get rows in a space... link the space documentation
- io_utils class...document it, use the things to read in data

  




    