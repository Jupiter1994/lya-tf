import numpy as np
import tensorflow as tf
import time

# local modules
from constants import *
import grid
import util 

odf_const = pi * e_cgs * e_cgs / (m_e_cgs * c_cgs)

hilya_f = 0.41619671797998276
hilya_lambda = 1.2150227340678867e-05 # in cm
odf_hilya = hilya_f * hilya_lambda

def gmlt_spec_odf_cosmo(h, e_z):
    '''
    Returns the cosmological optical depth factor as a float tensor.
    
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
    Returns the H I Lyman-alpha optical depth factor as a float tensor.
    
    Both parameters should be given as tensors.
    
    PARAMETERS
    ----------
    h: reduced Hubble constant
    e_z: the value of E(z) at a given z; E(z) is implemented in universe.py
    
    '''
    return odf_const * odf_hilya * gmlt_spec_odf_cosmo(h, e_z)

def roll_left(matrix, shifts):
    '''
    For each row in matrix, roll/shift the elements to the left by a certain offset (given by shifts).
    Assumes that matrix is a tensor with shape (m, n) and shifts is a tensor with shape (m,). Used in
    the gmlt_spec_od_pwc_exact method.
    
    This method is taken from this Stack Overflow answer: https://stackoverflow.com/a/58176568/16360935
    
    '''

    # get shape of the input matrix
    shape = tf.shape(matrix)

    # compute and stack the meshgrid to get the index matrix of shape (2,m,n)
    ind = tf.stack(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij'))

    # reshape it to (m,n,2)
    ind = tf.transpose(ind, [1,2,0])

    # add the value from shifts to the corresponding row and apply modulo shape[1];
    # this will effectively introduce the desired shift, but at the level of indices
    shifted_ind = tf.math.floormod(tf.transpose(tf.transpose(ind[:,:,1]) + shifts), shape[1])

    # convert the shifted indices to the right shape
    new_ind = tf.transpose(tf.stack([ind[:,:,0], shifted_ind]) , [1,2,0]) 

    # return the resliced tensor
    return tf.gather_nd(matrix, new_ind)

def gmlt_spec_od_pwc_exact(od_factor, m_x, v_domain, num_elements, 
                           n_array, vpara_array, t_array, num_pixels):
    '''
    Analytic form for piecewise constant data. Returns a 1D tensor
    that contains the optical depth of each cell along a line of sight.
    
    Note: gmlt_spec_od_grid uses the same value for num_elements and num_pixels.
    This value is referred to as N in the comments.
    
    PARAMETERS
    ----------
    od_factor, m_x: floats
    v_domain: a float tensor
    num_elements: number of elements in the line-of-sight; an int
    n_array, vpara_array, t_array: 1D tensors of floats
    num_pixels: pixel width of the simulated spectrum; an int
    
    '''
    # numerical floors
    t_min = 1.0

    element_dv = v_domain / num_elements
    pixel_dv = v_domain / num_pixels

    # The thermal velocity prefactor.
    # v_th = sqrt( 2 k T / m ), so compute 2 k / m_x once.
    vth_factor = 2.0 * k_cgs / m_x
    # thermal velocity tensor
    v_doppler = tf.math.sqrt(vth_factor * (t_array + t_min))

    # Initialize tau array
    tau_array = tf.zeros([num_pixels], dtype='float64')
    
    # line-center velocity tensors
    all_inds = np.arange(num_elements)
    vlc_l = element_dv * all_inds + vpara_array
    vlc_m = element_dv * (all_inds + 0.5) + vpara_array
    vlc_h = element_dv * (all_inds + 1.0) + vpara_array
    
    # only look at cells with positive density
    # inds = all_inds[(n_array > 0).numpy()]
    
    # the line center and thermal broadening in pixel index space; both have shape (N,)
    pix_lc = tf.cast(vlc_m / pixel_dv, dtype=tf.int32)
    pix_doppler = v_doppler / pixel_dv
    
    # Figure out which pixels we should restrict to.
    #num_integ_pixels = tf.cast(5 * pix_doppler + 0.5, dtype=tf.int32) + 1
    num_integ_pixels = 29
    window_width = (int) (2 * (num_integ_pixels) + 1) # 60
    
    # the Doppler profile bounds; both have shape (N,)
    ipix_lo = pix_lc - num_integ_pixels - 1
    ipix_hi = pix_lc + num_integ_pixels + 1
    ipixes = tf.linspace(ipix_lo, ipix_hi, window_width + 1, axis=-1) # (N, 61)
    
    # calculate the bin velocity with the original indices to (eventually) match v_lc
    v_pixel = tf.math.multiply(pixel_dv, tf.math.add(ipixes,0.5))
    
    # create tau profiles
    vlc_l = tf.reshape(vlc_l, [num_pixels, 1])
    vlc_h = tf.reshape(vlc_h, [num_pixels, 1])
    v_dopp_diag = tf.linalg.diag(v_doppler)
    v_dopp_diag_inv = tf.linalg.inv(v_dopp_diag)
    
    dxl = tf.linalg.matmul(v_dopp_diag_inv, v_pixel - vlc_l)
    dxh = tf.linalg.matmul(v_dopp_diag_inv, v_pixel - vlc_h)
    
    n_diag = tf.linalg.diag(n_array)
    tau_prof = tf.linalg.matmul(n_diag, tf.math.erf(dxl) - tf.math.erf(dxh))
    
    # "insert" tau_prof onto a N^2 grid
    #extra_cols = num_pixels - tf.shape(tau_prof).numpy()[1] # this only works with map_fn
    extra_cols = num_pixels - window_width - 1 # this works with vectorized_map
    paddings = tf.constant([[0, 0], [0, extra_cols]])
    tau_prof = tf.pad(tau_prof, paddings, "CONSTANT", constant_values=0)
    
    # roll each profile to its correct line center;
    # the offsets should be pix_lc - 30 to the right
    offsets_left = num_pixels - (pix_lc - 30)
    tau_prof_rolled = roll_left(tau_prof, offsets_left)
    
    # add up all profiles
    tau_array = tf.reduce_sum(tau_prof_rolled, axis=0)

    # Don't forget prefactor.
    f = 0.5 * od_factor
    tau_array *= f
    
    return tau_array

def reshape_2d(arr):
    '''
    Reshape a tensor of shape (x, y, z) to (x*y, z). This is applied to the fields in
    gmlt_spec_od_grid in order to support vectorization.

    PARAMETERS
    ----------
    arr: a 3D tensor

    '''

    x,y,z = arr.shape

    return tf.reshape(arr, (x*y, z))
    
def gmlt_spec_od_grid(universe, redshift, size, 
                      n_hi, temp, v_para, num_pixels):
    '''
    Returns a Grid object with the same shape and size as the input fields.
    (This method assumes that the sim and spectral shapes are the same.)
    
    PARAMETERS
    ----------
    universe: a Universe object
    redshift: a float
    size: physical dimensions of the field (a 3-term vector)
    n_hi, temp, v_para: the fields of the corresponding Grid objects (3D tensors)
    num_pixels: the output's length along the z-axis (an int)
    
    '''
    
    # load in the field's dimensions
    dist_nx = n_hi.shape[0]
    dist_ny = n_hi.shape[1]
    num_elements = n_hi.shape[2]

    # Get the domain size in velocity.
    l = size[2]
    domain_v_size = universe.chi_to_velocity_cgs(l, redshift)

    # The optical depth prefactor (a float tensor)
    odf = gmlt_spec_odf_hilya(universe.h, universe.E(redshift))

    #tau_field = tf.zeros([dist_nx, dist_ny, num_elements], dtype='float64')

    # Iterate over sim grid skewers.
    start2 = time.time()
    
    # @tf.function(); including this throws a "Tensor can't be used as Python bool" error
    @tf.autograph.experimental.do_not_convert()
    def od_pwc_exact2(skewers):
        '''
        Call gmlt_spec_od_pwc_exact, except all arguments (besides the 3 skewers)
        are pre-specified in gmlt_spec_od_grid, outside the scope of this method.
        
        PARAMETERS
        ----------
        skewers: (n_hi, temp, v_para), a tuple of skewers (1D tensors)
        
        '''
        
        n = skewers[0]
        t = skewers[1]
        vp = skewers[2]

        return gmlt_spec_od_pwc_exact(odf, m_H_cgs, domain_v_size, num_elements,
                                     n, vp, t, num_pixels)
    
    # apply od_pwc_exact2 to the x*y skewers
    skewers = (reshape_2d(n_hi), reshape_2d(temp), reshape_2d(v_para))
    #tau_field = tf.map_fn(od_pwc_exact2, skewers, fn_output_signature=tf.float64)
    tau_field = tf.vectorized_map(od_pwc_exact2, skewers)
        
    # reshape tau from (x*y,z) to (x,y,z)
    tau_field = tf.reshape(tau_field, n_hi.shape) # reshape (x*y,z) to (x,y,z)
    
    print('tau duration:', time.time() - start2)
        
    return grid.Grid(tau_field, n_hi.shape, size)

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
    prefactor, m_x, v_domain: floats
    num_elements: an int
    n_array, vpara_array, t_array: arrays of floats
    num_pixels: an int
    tau_array: an array of floats
    '''
    
    pass
