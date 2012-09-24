'''
Created on Sep 20, 2012

@author: georgianadinu
'''

class Weighting(object):
    '''
    classdocs
    '''
    _name = "we are NOT stupid"

        
    def get_name(self):
        return self._name
    
    name = property(get_name)