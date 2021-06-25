import tensorflow as tf

# Indexing

def gmlt_index_wrap(i, n):
    '''
    Wrap an index into an array's bounds.
    
    PARAMETERS
    ----------
    i: index
    n: length of array
    
    '''
    r = i - n * int( float(i) / float(n) )
    
    if (r < 0):
        r += n

    return r