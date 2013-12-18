#FIXME: remove!!!
class External(object):
    '''
    Base class for transformations based on external commands
    '''
    def create_operation(self):
        raise NotImplementedError()
    
    def apply(self, matrix):
        raise NotImplementedError()