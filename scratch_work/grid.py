import tensorflow as tf
    
class Grid:
    '''
    Class that handles a grid/scalar field of floats. 

    '''
    
    def __init__(self, field):
        '''
        Read in the field.
        
        PARAMETERS
        ----------
        field: a 3D tensor
        
        '''
        
        self.field = field