'''
Created on Sep 14, 2012

@author: georgianadinu

Exemplifies:
 1) loading a saved space
 2) computing similarities and finding neighbours using two spaces 
'''


def main(): 
    
    #load saved space 1
    
    #load saved space 2
   
    # Compute similarity with two spaces
    calculator = SimilarityCalculator(space1, space2, similarity = "cos")
    
    print calculator.computeSimilarity("man", "car")
    
    calculator.computeSimilarity("../data/pairs_of_words.txt", "../data/pairs_of_words.sim.txt")
 
    # Find neighbors with two spaces (neighbors in space 2 of word from space 1)
    print calculator.computeNeighbors("man", 10)
    
    
    #TO DO space3 = space1.Concat(space2)
    # 4 types of spaces:
    # core space:
    # composite space: core + peripheral
    # mixed space (obtained through concatenation, only similarity comp is valid on it)
    # composed space.. ??(core space?) - 
    
    calculator.computeNeighbors("../data/list_of_words", "../data/list_of_words.neigh.txt", 10)

    
    
    
if __name__ == '__main__':
    pass