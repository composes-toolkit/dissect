
class InvalidArgumentError(Exception):
    '''
    '''
    def __init__(self, msg):
        self.__msg = msg