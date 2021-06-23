import tensorflow as tf
    
class Grid:
    '''
    Class that handles a grid/scalar field of floats. 

    '''
    
    def __init__(self, field, shape, size):
        '''
        Read in the field.
        
        PARAMETERS
        ----------
        field: a 3D tensor
        shape: a vector containing the tensor's dimensions
        size: a vector containing the field's physical dimensions
        
        '''
        
        self.field = field
        self.shape = shape
        self.size = size