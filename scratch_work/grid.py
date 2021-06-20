import tensorflow as tf
    
class DGS:
    '''
    Class that handles a scalar grid of doubles.

    '''
    
    def __init__(self, field):
        '''
        Read in the field.
        
        PARAMETERS
        ----------
        field: a 3D tensor
        
        '''
        
        self.field = field