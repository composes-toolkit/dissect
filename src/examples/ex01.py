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
    
    #1 builds space
    space = Space.build("../data/counts1.mat", "../data/counts1.row", "../data/count1.col", "mat")
    
    #2 builds space
    space = Space.build("../data/counts1.mat", "mat")
    
    #3 loads space object
    space = Space.load(".pickle")
    
    space.printInfo()
        
    #4
    space.export(format = "mat")
        
    # Apply Positive pmi weigthing
    w = PpmiWeigthing()
    space = space.applyWeighting(PpmiWeigthing()).applyWeighting(PlmiWeighting())    
    
    space = space.applyWeighting("ppmi")
    
    PpmiWeighting
    # Apply SVD reduction
    space = space.applyDimReduction("svd", 300)
    
    space.printInfo()
    
    space.save("..")
    

    # Compute similarity within one space
    calculator = SimilarityCalculator(space, similarity = "cos")
   
    print calculator.computeSimilarity("man", "car")

    calculator.computeSimilarity("../data/pairs_of_words.txt", "../data/pairs_of_words.sim.txt")


if __name__ == '__main__':
    main()
    