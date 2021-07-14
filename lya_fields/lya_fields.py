'''
Write the number density of neutral hydrogen (nHI), real- and redshift-space optical depths
      into an HDF5 file.
      
'''

import h5py
import numpy as np
import scipy.interpolate as interp
import tensorflow as tf
import time

# local modules
import eos
import grid
import snapshot
from spectrum import gmlt_spec_od_grid
from constants import *

# define dataset paths
ds_path_rhob = "/native_fields/baryon_density"
ds_path_temp = "/native_fields/temperature"
ds_path_vx   = "/native_fields/velocity_x"
ds_path_vy   = "/native_fields/velocity_y"
ds_path_vz   = "/native_fields/velocity_z"

ds_path_nhi      = "/derived_fields/HI_number_density"
ds_path_tau_real = "/derived_fields/tau_real"
ds_path_tau_red  = "/derived_fields/tau_red"

def write_field(field, dset_path, file_path):
    '''
    Save a field into an HDF5 file.
    
    PARAMETERS
    ----------
    field: a 3D tensor
    dset_path: the path of the dataset/field, e.g. 'tau_real'
    file_path: the path of the HDF5 file to save to
    '''
    f = h5py.File(file_path,'a') # read/write if file exists, create otherwise

    if dset_path in f: # replace an existing dataset
        data = f[dset_path]
        data[...] = field.numpy()
    else: # create a new dataset
        f.create_dataset(dset_path, data=field.numpy())

    f.close()
        
def set_nhi(snap, rhob, temp):
    '''
    Interpolate n(rho, T) and its gradients, calls compute_nhi, and returns the n_HI grid.
    
    PARAMETERS
    ----------
    snap: a Snapshot object
    rhob: the baryon density field (a Grid object)
    temp: the temperature field (a Grid object)
    
    '''
    
    ## load in snapshot properties
    a = snap.scale_factor
    z = snap.z
    u = snap.universe
    h = u.h
    omega_b = u.omega_b
    
    mean_rhob_cgs = omega_b * h*h * rho_crit_100_cgs
    a3_inv = 1.0 / (a * a * a)
    #rhob_cgs = mean_rhob_cgs * a3_inv * rhob.field

    # initialize EOS object
    #nhi_field = tf.zeros(rhob.shape, dtype='float64')
    eos_obj = eos.EOS_at_z(z)
    
    start2 = time.time()
    
    ## interpolate the function f, which represents n(log(rho), log(T))

    # set the ranges for interpolation
    length = 100 
    
    log10_rho_range = np.linspace(-1, 3, num=length)
    rho_range = 10**log10_rho_range
    rhob_cgs_range = mean_rhob_cgs * a3_inv * rho_range

    log10_t_range = np.linspace(3, 6, num=length)
    temp_range = 10**log10_t_range
    
    # note: this is just a 2D grid, NOT a Grid object
    nhi_grid = np.ndarray((length, length))
    for i in range(length):
        for j in range(length):
            nhi_grid[i, j] = eos_obj.nyx_eos(rhob_cgs_range[i], temp_range[j])
    
    deg = 3 # degree of spline; default is 3
    f = interp.RectBivariateSpline(log10_rho_range, log10_t_range, nhi_grid, kx=deg, ky=deg)
    
    ## interpolate the 1st-order partial derivatives: n_[log(rho)] and n_[log(T)]
    f_logr_grid = f(log10_rho_range, log10_t_range, dx=1, dy=0)
    f_logr = interp.interp2d(log10_rho_range, log10_t_range, f_logr_grid)

    f_logt_grid = f(log10_rho_range, log10_t_range, dx=0, dy=1)
    f_logt = interp.interp2d(log10_rho_range, log10_t_range, f_logt_grid)
    
    # get log10(x) via change of base formula
    log10_rhob = tf.math.log(rhob.field) / np.log(10)
    log10_temp = tf.math.log(temp.field) / np.log(10)
    
    # pass the 2 fields and 3 functions to compute_nhi
    nhi_field = compute_nhi(log10_rhob, log10_temp, f, f_logr, f_logt)
    
    print('EOS duration:', time.time() - start2)
    
    return grid.Grid(nhi_field, rhob.shape, rhob.size)

def flatten(t):
    '''
    Flatten an array or tensor t. Used in compute_nhi.
    
    '''
    size = tf.size(t)
    return tf.reshape(t, [size])

def unflatten(t, shape):
    '''
    Unflatten an array or tensor t to have a specified shape. Used in compute_nhi.
    
    '''

    return tf.reshape(t, shape)

@tf.custom_gradient
def compute_nhi(log10_rhob, log10_temp, n, n_logr, n_logt):
    '''
    Compute the neutral hydrogen number density (n_HI) field, returning a tensor. 
    Includes a custom gradient function.
    
    PARAMETERS
    ----------
    log10_rhob, log10_temp: 3D tensors containing log10(rho) and log10(T)
    n: n(log(rho), log(T)) (function interpolated with RectBivariateSpline)
    n_logr, n_logt: the partial derivatives of n_log10 w.r.t. log10(rho) and log10(T), 
    respectively (functions interpolated with interp2d)
    
    '''    
    
    def grad(upstream):
        '''
        Compute the gradients dn/drho and dn/dt via chain rule:
        dn/dx = dn/dlogx * dlogx/dx = dn/dlogx / x / ln(10)
        
        '''
        
        # compute the dn/dlogx tensors 
        dn_dlogr = interp.dfitpack.bispeu(n_logr.tck[0], n_logr.tck[1], n_logr.tck[2], \
                                          n_logr.tck[3], n_logr.tck[4], \
                                        flatten(log10_rhob), flatten(log10_temp))[0]
        dn_dlogr = unflatten(dn_dlogr, log10_rhob.shape)

        dn_dlogt = interp.dfitpack.bispeu(n_logt.tck[0], n_logt.tck[1], n_logt.tck[2], \
                                          n_logt.tck[3], n_logt.tck[4], \
                                        flatten(log10_rhob), flatten(log10_temp))[0]
        dn_dlogt = unflatten(dn_dlogt, log10_rhob.shape)
        
        # compute the dn/dx tensors 
        dn_drho = tf.divide(dn_dlogr, 10**log10_rhob) / np.log(10)
        dn_dt = tf.divide(dn_dlogt, 10**log10_temp) / np.log(10)
        
        return upstream * dn_drho, upstream * dn_dt
    
    nhi = n(log10_rhob, log10_temp, grid=False) # this is an ndarray
    return tf.convert_to_tensor(nhi), grad

def main():
    start1 = time.time() # track the total duration
    
    ### load in the density and temperature fields as grids ###

    filename = "../../../../../cscratch1/sd/jupiter/sim2_z3_FGPA_cgs.h5"
    snap = snapshot.Snapshot(filename)
    
    # subsection shape
    shape = [1, 10, 10]
    
    # string representing the subsection's dimensions, e.g. '4x4x4'
    dims_str = str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2])
    
    # file paths to routine durations and results
    t_path = 'times/times' + dims_str + '.txt'
    results_path = 'results/tf_fields' + dims_str + '.h5'
    
    rhob = snap.read_subfield(ds_path_rhob, shape)
    temp = snap.read_subfield(ds_path_temp, shape)
    
    # track routine durations
    times = open(t_path, 'w')
    
    ### compute nhi grid ###
    start = time.time()
    nhi = set_nhi(snap, rhob, temp)
    times.write('EOS duration = ' + str(time.time() - start) + '\n')
    
    write_field(nhi.field, 'nhi', results_path)
    
    ### calculate optical depth fields ###
    # real-space tau
    vpara = grid.Grid(tf.zeros(rhob.shape, dtype='float64'), rhob.shape, rhob.size)
    
    start = time.time()
    tau_real = gmlt_spec_od_grid(snap.universe, snap.z, nhi.size,
            nhi.field, temp.field, vpara.field, nhi.field.shape[2])
    times.write('tau_real duration = ' + str(time.time() - start) + '\n')
    
    write_field(tau_real.field, 'tau_real', results_path)

    # redshift-space tau
    #vpara = snap.read_field(ds_path_vz)
    vpara = snap.read_subfield(ds_path_vz, shape)
    
    start = time.time()
    tau_red = gmlt_spec_od_grid(snap.universe, snap.z, nhi.size,
            nhi.field, temp.field, vpara.field, nhi.field.shape[2])
    times.write('tau_red duration = ' + str(time.time() - start) + '\n')

    write_field(tau_red.field, 'tau_red', results_path)
    
    times.write('Total duration = ' + str(time.time() - start1) + '\n')
    times.close()
    
    print('Total duration:', time.time() - start1)
    
# prevent this script from being run accidentally
if __name__ == '__main__':
    main()