import tensorflow as tf

from constants import *

odf_const = pi * e_cgs * e_cgs / (m_e_cgs * c_cgs)

hilya_f = 0.41619671797998276
hilya_lambda = 1.2150227340678867e-05 # in cm
odf_hilya = hilya_f * hilya_lambda

def gmlt_spec_odf_cosmo(h, e_z):
    '''
    Both parameters should be given as tensors.
    
    PARAMETERS
    ----------
    h: reduced Hubble constant
    e_z: the value of E(z) at a given z; E(z) is implemented in universe.py
    
    '''
    H_z = H0_100_cgs * h * e_z
    return tf.divide(1.0, H_z)

def gmlt_spec_odf_hilya(h, e_z):
    '''
    Both parameters should be given as tensors.
    
    PARAMETERS
    ----------
    h: reduced Hubble constant
    e_z: the value of E(z) at a given z; E(z) is implemented in universe.py
    
    '''
    return odf_const * odf_hilya * gmlt_spec_odf_cosmo(h, e_z)

# this method isn't called elsewhere in Gimlet, so I'll ignore it for now
def gmlt_spec_od_midpoint(prefactor, m_x,
    v_domain,
    num_elements, n_array,
    vpara_array, t_array,
    num_pixels, tau_array):
    '''
    Midpoint evaluated at the center of elements.
    
    PARAMETERS
    ----------
    prefactor, m_x, v_domain: doubles
    num_elements: an int
    n_array, vpara_array, t_array: arrays of doubles
    num_pixels: an int
    tau_array: an array of doubles
    '''
    
    pass

# TODO: implement the 2 methods below
def gmlt_spec_od_pwc_exact(od_factor, m_x,
    v_domain,
    num_elements, n_array,
    vpara_array, t_array,
    num_pixels, tau_array):
    '''
    Analytic form for piecewise constant data.
    
    PARAMETERS
    ----------
    od_factor, m_x, v_domain: doubles
    num_elements: an int
    n_array, vpara_array, t_array: arrays of doubles
    num_pixels: an int
    tau_array: an array of doubles
    
    '''
    
    pass

def gmlt_spec_od_grid(universe, redshift, dist, n_hi,
    temp, v_para,
    num_pixels, tau):
    '''
    Assuming that the sim and spectral shapes are the same here.
    
    PARAMETERS
    ----------
    universe: a Universe object
    redshift: a double
    dist: a GridDist object
    n_hi, temp, v_para: arrays of doubles
    num_pixels: an int
    tau: an empty array of doubles
    '''
    
    # TODO: initialize the tau array?
    
    # return tau
    pass


