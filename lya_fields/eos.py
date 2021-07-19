import tensorflow as tf
import numpy as np
import scipy.interpolate as interp

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
        
        self.z = float(z)
        
    def nyx_eos(self, rhob, temp):
        '''
        Calculate n_HI for a given density and temperature.

        PARAMETERS
        ----------
        rhob, temp: floats or 1-element tensors
        
        '''

        return eos_t.eos.nyx_eos(self.z, rhob, temp)
    
    def nyx_eos_vec(self, arr):
        '''
        A version of nyx_eos that supports vectorization.

        PARAMETERS
        ----------
        arr: tuple containing (rhob, temp)
        
        '''
        rhob = arr[0]
        temp = arr[1]

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

# prevent this script from being run accidentally
if __name__ == '__main__':
    main()