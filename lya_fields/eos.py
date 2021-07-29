import tensorflow as tf
import numpy as np
import scipy.interpolate as interp
import sys

# local modules
import eos_t
    
class EOS_at_z:
    '''
    Class that handles the equation of state at a given redshift z.

    '''
    
    def __init__(self, z, rhob_cgs_conversion):
        '''
        Read in the redshift and perform interpolations.
        
        PARAMETERS
        ----------
        z: redshift (float)
        rhob_cgs_conversion: conversion from rhob to rhob_cgs (float)
        
        INSTANCE VARIABLES
        ------------------
        z: redshift
        n: the function n(log10(rho), log10(T)), interpolated with RectBivariateSpline
        n_logr, n_logt: the partial derivatives of n w.r.t. log10(rho) and log10(T),
        respectively; both interpolated with interp2d
        
        '''

        self.z = float(z)
        
        ## interpolate the function n_HI and its two 1st-order partial derivatives
        
        # set the ranges for interpolation
        length = 200 

        log10_rho_range = np.linspace(-2, 3, num=length)
        rho_range = 10**log10_rho_range
        rhob_cgs_range = rhob_cgs_conversion * rho_range

        log10_t_range = np.linspace(3, 6, num=length)
        temp_range = 10**log10_t_range

        nhi_grid = np.ndarray((length, length)) # this is a 2D grid, but NOT a Grid object
        for i in range(length):
            for j in range(length):
                nhi_grid[i, j] = self.nyx_eos(rhob_cgs_range[i], temp_range[j])
                
        # interpolate n
        deg = 3 # degree of spline; default is 3
        self.n = interp.RectBivariateSpline(log10_rho_range, log10_t_range, nhi_grid, kx=deg, ky=deg)

        # interpolate the 1st-order partial derivatives: n_[log10(rho)] and n_[log10(T)]
        n_logr_grid = self.n(log10_rho_range, log10_t_range, dx=1, dy=0)
        self.n_logr = interp.interp2d(log10_rho_range, log10_t_range, n_logr_grid)

        n_logt_grid = self.n(log10_rho_range, log10_t_range, dx=0, dy=1)
        self.n_logt = interp.interp2d(log10_rho_range, log10_t_range, n_logt_grid)
        
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
        A version of nyx_eos that supports vectorization. (Not used.)

        PARAMETERS
        ----------
        arr: tuple containing (rhob, temp)
        
        '''
        rhob = arr[0]
        temp = arr[1]

        return eos_t.eos.nyx_eos(self.z, rhob, temp)

    def flatten(self, t):
        '''
        Flatten an array or tensor t. Used in compute_nhi.

        '''
        
        size = tf.size(t)
        return tf.reshape(t, [size])

    def unflatten(self, t, shape):
        '''
        Unflatten an array or tensor t to have a specified shape. Used in compute_nhi.

        '''

        return tf.reshape(t, shape)

    
    def compute_dn_dlogx(self, log10_rhob, log10_temp, x):
        '''
        Compute n_logx at specific coordinates for (log10_rhob, log10_temp), where x 
        is either rhob or temp.
        
        PARAMETERS
        ----------
        log10(rhob), log10(temp): floats
        x: a string, either 'rhob' or 'temp' 
        
        '''
                
        if x == 'rhob':
            return interp.dfitpack.bispeu(self.n_logr.tck[0], self.n_logr.tck[1], self.n_logr.tck[2], \
                                        self.n_logr.tck[3], self.n_logr.tck[4], \
                                        log10_rhob, log10_temp)[0]
        elif x == 'temp':
            return interp.dfitpack.bispeu(self.n_logt.tck[0], self.n_logt.tck[1], self.n_logt.tck[2], \
                                        self.n_logt.tck[3], self.n_logt.tck[4], \
                                        log10_rhob, log10_temp)[0]
        else:
            raise NameError("The argument x must be either 'rhob' or 'temp'")
    
    @tf.custom_gradient
    def compute_nhi(self, log10_rhob, log10_temp):
        '''
        Compute the neutral hydrogen number density (n_HI) field, returning a 3D tensor. 
        This method includes a custom gradient function (grad).
        
        PARAMETERS
        ----------
        log10_rhob, log10_temp: 3D tensors containing log10(rho) and log10(T) 
        
        '''
        
        def grad(upstream):
            '''
            Compute the gradients dn/drho and dn/dt via chain rule:
            
            dn/dx = dn/dlogx * dlogx/dx = dn/dlogx / x / ln(10)
            
            (The values of dn/dlogx are computed via n_logr and n_logt.)

            '''
            
            # test interp.dfitpack.bispeu
            try:
                fake_rho, fake_t = [[-20,-19]], [[[4,3]]]
                
                filler = interp.dfitpack.bispeu(self.n_logr.tck[0], self.n_logr.tck[1], self.n_logr.tck[2], \
                                              self.n_logr.tck[3], self.n_logr.tck[4], \
                                            self.flatten(fake_rho), self.flatten(fake_t))[0]
                print('interp.dfitpack.bispeu works!')
            except Exception as err:
                print('Error: interp.dfitpack.bispeu doesn\'t work')
                print(sys.exc_info()[0])
                
            # test compute_dn_dlogx
            try:
                fake_rho = tf.constant([-20, -19])
                fake_t = tf.constant([3, 4])
                
                print('fake_rho:', fake_rho)
                print('fake_rho type:', type(fake_rho))
                print('fake_rho shape:', tf.shape(fake_rho))
                
                filler = self.compute_dn_dlogx(self.flatten(fake_rho), self.flatten(fake_t), 'rhob')
                print('compute_dn_dlogx works on 1D tensors!')
            except Exception as err:
                print('Error: compute_dn_dlogx doesn\'t work on 1D tensors')
                print(sys.exc_info()[0])
        
#             try:
#                 fake_rho = tf.constant([[-20, -19], [-20, -19]])
#                 fake_t = tf.constant([[3, 4], [3, 4]])
#                 filler = self.compute_dn_dlogx(self.flatten(fake_rho), self.flatten(fake_t), 'rhob')
#                 print('compute_dn_dlogx works on 2D tensors!')
#             except Exception as err:
#                 print('Error: compute_dn_dlogx doesn\'t work on 2D tensors')
#                 print(sys.exc_info()[0])

            # compute the tensors containing the dn/dlogx values 
#             dn_dlogr = interp.dfitpack.bispeu(self.n_logr.tck[0], self.n_logr.tck[1], self.n_logr.tck[2], \
#                                               self.n_logr.tck[3], self.n_logr.tck[4], \
#                                             self.flatten(log10_rhob), self.flatten(log10_temp))[0]
#             dn_dlogr = self.unflatten(dn_dlogr, log10_rhob.shape)

#             dn_dlogt = interp.dfitpack.bispeu(self.n_logt.tck[0], self.n_logt.tck[1], self.n_logt.tck[2], \
#                                               self.n_logt.tck[3], self.n_logt.tck[4], \
#                                             self.flatten(log10_rhob), self.flatten(log10_temp))[0]
#             dn_dlogt = self.unflatten(dn_dlogt, log10_rhob.shape)

            dn_dlogr = self.compute_dn_dlogx(self.flatten(log10_rhob), self.flatten(log10_temp), 'rhob')
            dn_dlogt = self.compute_dn_dlogx(self.flatten(log10_rhob), self.flatten(log10_temp), 'temp')

            dn_dlogr = self.unflatten(dn_dlogr, log10_rhob.shape)
            dn_dlogt = self.unflatten(dn_dlogt, log10_rhob.shape)
            
            # compute the tensors containing the dn/dx values 
            dn_drho = tf.divide(dn_dlogr, 10**log10_rhob) / np.log(10)
            dn_dt = tf.divide(dn_dlogt, 10**log10_temp) / np.log(10)
        
            return upstream * dn_drho, upstream * dn_dt
        
        nhi = self.n(log10_rhob, log10_temp, grid=False) # this is a 3D ndarray
        return tf.convert_to_tensor(nhi), grad
        
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