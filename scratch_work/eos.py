import tensorflow as tf
import numpy as np

# local modules
import eos_t
    
class EOS_at_z:
    '''
    Class that handles the equation of state at a given redshift z.

    '''
    
    def __init__(self, z):
        '''
        Read in the redshift.
        
        PARAMETERS
        ----------
        z: a float
        
        '''
        
        self.z = z
        
    def nyx_eos(rhob, temp):
        '''
        Calculate n_HI for a given density and temperature.

        PARAMETERS
        ----------
        rhob: a float
        temp: a float
        
        '''
        
        return eos_t.eos.nyx_eos(self.z, rhob, temp)
        
def main():
    # order of magnitude values
#     z = 3.
#     rhob_cgs = 1e-29
#     temp = 6e3

    # even if the inputs are arrays, nyx_eos (seemingly) only takes
    # the first entry in each array and returns one float
    z = [[3.,3.1], [3.,3.]]
    rhob_cgs = [[1e-29,1.3e-29], [1e-29,1.5e-29]]
    temp = [[6e3,6.2e3], [6.2e3,6e3]]
    
    z = np.asarray(z)
    rhob_cgs = np.asarray(rhob_cgs)
    temp = np.asarray(temp)
    print(z)
    
    nhi = eos_t.eos.nyx_eos(z, rhob_cgs, temp)
    print('nhi:', nhi)

if __name__ == '__main__':
    main()