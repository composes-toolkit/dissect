'''
Created on Sep 14, 2012

@author: georgianadinu
Exemplifies:
 1) Reading a co-occurence file
 2) Applying a weighting scheme
 3) Reducing dimensionality
 4) Saving a space
 5) Computing similarity within one space
'''


def main(): 
    
    #input formats: .mat .tbl .pkldump
    space = Space()    
    space.setCoreMatrix("../data/counts1.mat", "../data/counts1.row", "../data/count1.col", "mat")
    
    # Apply Positive pmi weigthing
    space = space.applyWeighting    
    
    # Apply SVD reduction
    space = space.Reduce
    
    # Save resulted matrix
    space.dumpMatrix()

    # Compute similarity within one space
    calculator = SimilarityCalculator(space1)
   
    print calculator.computeSimilarity("man", "car")

    calculator.computeSimilarity("../data/pairs_of_words.txt", "../data/pairs_of_words.sim.txt")


if __name__ == '__main__':
    main()
    