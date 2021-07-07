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

def gmlt_spec_od_pwc_exact(od_factor, m_x, v_domain, num_elements, 
                           n_array, vpara_array, t_array, num_pixels):
    '''
    Analytic form for piecewise constant data. Returns a 1D tensor
    that contains the optical depth of each cell along a line of sight.
    
    PARAMETERS
    ----------
    od_factor, m_x: floats
    v_domain: a float tensor
    num_elements: number of elements in the line-of-sight; an int
    n_array, vpara_array, t_array: 1D tensors of floats
    num_pixels: pixel width of the simulated spectrum; an int
    
    '''
    
    def ind_wrap(i):
        '''
        Performs util.gmlt_index_wrap on an index, where r (in this case, num_pixels)
        is pre-specified outside the scope of this method.

        PARAMETERS
        ----------
        i: index (an int tensor)

        '''

        return util.gmlt_index_wrap(i, num_pixels)
    
    # numerical floors
    t_min = 1.0

    element_dv = v_domain / num_elements
    pixel_dv = v_domain / num_pixels

    # The thermal velocity prefactor.
    # v_th = sqrt( 2 k T / m ), so compute 2 k / m_x once.
    vth_factor = 2.0 * k_cgs / m_x

    # Initialize tau array
    tau_array = tf.zeros([num_pixels], dtype='float64')

    # Loop over elements, adding optical depth to nearby pixels
    for i in range(num_elements):
        # make sure this cell has positive density.
        # with vectorized_map, neither of the below pieces of code work:
        if (n_array[i] <= 0.0):
            continue
        # assert tf.math.greater(n_array[i], 0.0)
            
        # thermal velocity.
        v_doppler = tf.math.sqrt(vth_factor * (t_array[i] + t_min))
        # line-center velocity
        vlc_l = element_dv * i + vpara_array[i]
        vlc_m = element_dv * (i + 0.5) + vpara_array[i]
        vlc_h = element_dv * (i + 1.0) + vpara_array[i]

        # The line center and thermal broadening in pixel index space.
        pix_lc = vlc_m / pixel_dv
        pix_doppler = v_doppler / pixel_dv

        # Figure out which pixels we should restrict to.
        num_integ_pixels = tf.cast(5 * pix_doppler + 0.5, dtype=tf.int32) + 1
        num_integ_pixels = tf.cast(num_integ_pixels, dtype=tf.float64)
        ipix_lo = pix_lc - num_integ_pixels - 1
        ipix_hi = pix_lc + num_integ_pixels + 1
        
        # convert the Doppler profile bounds from float to int
        ipix_lo = tf.cast(ipix_lo, dtype=tf.int32)
        ipix_hi = tf.cast(ipix_hi, dtype=tf.int32)
        
        ### Create the Doppler profile and add it to tau_array
        
        # Include the hi bin.
        ipixes = tf.constant(range(ipix_lo, ipix_hi + 1), dtype=tf.int32)
        # wrap indices into the spectrum's bounds
        jw = tf.map_fn(ind_wrap, ipixes)
        # necessary for tensor_scatter_nd_add
        jw = tf.reshape(jw, [tf.shape(jw).numpy()[0], 1]) 
        
        # Calculate the bin velocity with the original indices to match v_lc
        ipixes = tf.cast(ipixes, dtype=tf.float64)
        v_pixel = tf.math.multiply(pixel_dv, tf.math.add(ipixes,0.5))
        
        # Add tau contribution to the pixel
        dxl = (v_pixel - vlc_l) / v_doppler
        dxh = (v_pixel - vlc_h) / v_doppler
        tau_i = tf.math.multiply(n_array[i], (tf.math.erf(dxl) - tf.math.erf(dxh)))
        tau_array = tf.tensor_scatter_nd_add(tau_array, jw, tau_i)

        # testing tf.vectorized_map
#         print(ipix_lo)
#         print(ipix_lo.eval())

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
    # vectorized_map throws errors
    # tau_field = tf.vectorized_map(od_pwc_exact2, skewers)
    tau_field = tf.map_fn(od_pwc_exact2, skewers, fn_output_signature=tf.float64)
        
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
