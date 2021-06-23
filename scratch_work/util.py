import tensorflow as tf

# Indexing

def gmlt_index_wrap(const int64_t i, const int64_t n):
    '''
    Wrap an index into an array's bounds.
    
    PARAMETERS
    ----------
    i: index
    n: length of array
    
    '''
    r = i - n * (int) ( (float) i / (float) n )
    
    if (r < 0):
        r += n

    return r