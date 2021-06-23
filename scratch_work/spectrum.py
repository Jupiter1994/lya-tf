import tensorflow as tf

# local modules
from constants import *
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
    od_factor, m_x, v_domain: floats
    num_elements: an int
    n_array, vpara_array, t_array: arrays of floats
    num_pixels: an int
    tau_array: an array of floats
    
    '''
    
    '''
    Gimlet code:
    -------------
    
    // numerical floors
    const double t_min = 1.0;

    const double element_dv = v_domain / num_elements;
    const double pixel_dv = v_domain / num_pixels;

    // The thermal velocity prefactor.
    // v_th = sqrt( 2 k T / m ), so compute 2 k / m_x once.
    const double vth_factor = 2.0 * k_cgs / m_x;

    // Reset tau array to 0.0.
    for (int i = 0; i < num_pixels; ++i) { tau_array[i] = 0.0; }

    // Loop over elements, adding optical depth to nearby pixels
    for (int i = 0; i < num_elements; ++i) {
        // make sure this cell has positive density.
        if (n_array[i] > 0.0) {
            // thermal velocity.
            double v_doppler = sqrt( vth_factor * (t_array[i] + t_min) );
            // line-center velocity
            double vlc_l = element_dv * i + vpara_array[i];
            double vlc_m = element_dv * (i + 0.5) + vpara_array[i];
            double vlc_h = element_dv * (i + 1.0) + vpara_array[i];

            // The line center and thermal broadening in pixel index space.
            double pix_lc = vlc_m / pixel_dv;
            double pix_doppler = v_doppler / pixel_dv;

            // Figure out which pixels we should restrict to.
            int num_integ_pixels = (int)(5 * pix_doppler + 0.5) + 1;
            int ipix_lo = pix_lc - num_integ_pixels - 1;
            int ipix_hi = pix_lc + num_integ_pixels + 1;

            // Note the <= to include the hi bin.
            for (int j = ipix_lo; j <= ipix_hi; ++j) {
                // Calculate the bin velocity with the original index to match v_lc.
                double v_pixel = pixel_dv * (j + 0.5);
                // Wrap j into bounds.
                int jw = util.gmlt_index_wrap(j, num_pixels);

                // Add tau contribution to the pixel.
                double dxl = (v_pixel - vlc_l) / v_doppler;
                double dxh = (v_pixel - vlc_h) / v_doppler;
                tau_array[jw] += n_array[i] * ( erf(dxl) - erf(dxh) );
            }
        }
    }

    // Don't forget prefactor.
    double f = 0.5 * od_factor;
    for (int i = 0; i < num_pixels; ++i) {
        tau_array[i] *= f;
    }
    
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
    redshift: a float
    dist: a GridDist object
    n_hi, temp, v_para: arrays of floats
    num_pixels: an int
    tau: an empty array of floats
    '''
    
    '''
    Gimlet code:
    ------------
    
    # load in the grid's dimensions
    const int64_t dist_nx = dist->n(0);
    const int64_t dist_ny = dist->n(1);
    const int num_elements = dist->n(2);

    # Get the domain size in velocity.
    double l = dist->grid->l(2);
    double domain_v_size = universe.chi_to_velocity_cgs(l, redshift);

    # The optical depth prefactor (a float tensor)
    odf = gmlt_spec_odf_hilya(universe.h(), universe.E(redshift))

    # Iterate over axis1 and axis2. Copy skewer values, compute tau, and
    # save back to field.

    # Iterate over sim grid skewers.
    for (int64_t ix = 0; ix < dist_nx; ++ix) {
        for (int64_t iy = 0; iy < dist_ny; ++iy) {
            # Figure out index of (x, y) skewer start in (x, y, z) arrays.
            const int64_t skewer_i0 = (ix * dist_ny + iy) * num_elements;
            const int64_t spec_i0 = (ix * dist_ny + iy) * num_pixels;
            # Pass the skewer off
            gmlt_spec_od_pwc_exact(odf, m_H_cgs, domain_v_size,
                num_elements, n_hi + skewer_i0, v_para + skewer_i0,
                temp + skewer_i0,
                num_pixels, tau + spec_i0);
        }
    }
    
    '''
    # TODO: initialize the tau array?
    
    # return tau
    pass

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
