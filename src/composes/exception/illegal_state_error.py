'''
Created on Jun 15, 2012

@author: thenghia.pham
'''

class IllegalStateError(Exception):
    '''
    '''
    def __init__(self, msg):
        self.__msg = msg


class IllegalOperationError(Exception):
    '''
    '''
    def __init__(self, msg):
        self.__msg = msg