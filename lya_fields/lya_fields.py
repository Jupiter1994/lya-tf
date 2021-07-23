'''
Write the number density of neutral hydrogen (nHI), real- and redshift-space optical depths
      into an HDF5 file.
      
'''

import h5py
import numpy as np
import scipy.interpolate as interp
import sys
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

def set_nhi(snap, rhob, temp):
    '''
    Calls compute_nhi, and returns the n_HI grid.
    
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
    rhob_cgs_conversion = mean_rhob_cgs * a3_inv
    
    start2 = time.time()
        
    # get log10(x) via change of base formula
    log10_rhob = tf.math.log(rhob.field) / np.log(10)
    log10_temp = tf.math.log(temp.field) / np.log(10)
    
    # compute the field
    nhi_field = compute_nhi(log10_rhob, log10_temp, rhob_cgs_conversion, z)
    
    print('EOS duration:', time.time() - start2)
    
    return grid.Grid(nhi_field, rhob.shape, rhob.size)

@tf.custom_gradient
def compute_nhi(log10_rhob, log10_temp, rhob_cgs_conversion, z):
    '''
    Compute the neutral hydrogen number density (n_HI) field, returning a 3D tensor. 
    
    n_HI and its two 1st-order partial derivatives are interpolated as functions of 
    log10(rho) and log10(T). This method includes a custom gradient function, grad.
    
    PARAMETERS
    ----------
    log10_rhob, log10_temp: 3D tensors containing log10(rho) and log10(T) 
    rhob_cgs_conversion: conversion from rhob to rhob_cgs (float)
    z: redshift (float)
    
    '''
    
#     Interpolate the function n(log10(rho), log10(T)) and its two 1st-order partial 
#     derivatives

    # set the ranges for interpolation
    length = 200 
    
    log10_rho_range = np.linspace(-2, 3, num=length)
    rho_range = 10**log10_rho_range
    rhob_cgs_range = rhob_cgs_conversion * rho_range

    log10_t_range = np.linspace(3, 6, num=length)
    temp_range = 10**log10_t_range
    
    eos_obj = eos.EOS_at_z(z)
    nhi_grid = np.ndarray((length, length)) # this is a 2D grid, but NOT a Grid object
    for i in range(length):
        for j in range(length):
            nhi_grid[i, j] = eos_obj.nyx_eos(rhob_cgs_range[i], temp_range[j])
            
    # interpolate n
    deg = 3 # degree of spline; default is 3
    n = interp.RectBivariateSpline(log10_rho_range, log10_t_range, nhi_grid, kx=deg, ky=deg)
    
    # interpolate the 1st-order partial derivatives: n_[log10(rho)] and n_[log10(T)]
    n_logr_grid = n(log10_rho_range, log10_t_range, dx=1, dy=0)
    n_logr = interp.interp2d(log10_rho_range, log10_t_range, n_logr_grid)

    n_logt_grid = n(log10_rho_range, log10_t_range, dx=0, dy=1)
    n_logt = interp.interp2d(log10_rho_range, log10_t_range, n_logt_grid)
    
    def grad(upstream):
        '''
        Compute the gradients dn/drho and dn/dt via chain rule:
        dn/dx = dn/dlogx * dlogx/dx = dn/dlogx / x / ln(10)
        
        '''
        
        # test flatten()
#         try:
#             filler = flatten(log10_rhob)
#             print('flatten works!')
#         except:
#             print('Error: flatten method doesn\'t work')
#             print('Traceback:')
#             print(sys.exc_info()[2])
            
        # test interp.dfitpack.bispeu
        try:
            fake_rho, fake_t = [[-20,-19]], [[[4,3]]]
            filler = interp.dfitpack.bispeu(n_logr.tck[0], n_logr.tck[1], n_logr.tck[2], \
                                          n_logr.tck[3], n_logr.tck[4], \
                                        flatten(fake_rho), flatten(fake_t))[0]
            print('interp.dfitpack.bispeu works!')
        except Exception as err:
            print('Error: interp.dfitpack.bispeu doesn\'t work')
            print(sys.exc_info()[0])
            
        # compute the tensors containing the dn/dlogx values 
        dn_dlogr = interp.dfitpack.bispeu(n_logr.tck[0], n_logr.tck[1], n_logr.tck[2], \
                                          n_logr.tck[3], n_logr.tck[4], \
                                        flatten(log10_rhob), flatten(log10_temp))[0]
        dn_dlogr = unflatten(dn_dlogr, log10_rhob.shape)

        dn_dlogt = interp.dfitpack.bispeu(n_logt.tck[0], n_logt.tck[1], n_logt.tck[2], \
                                          n_logt.tck[3], n_logt.tck[4], \
                                        flatten(log10_rhob), flatten(log10_temp))[0]
        dn_dlogt = unflatten(dn_dlogt, log10_rhob.shape)
        
        # compute the tensors containing the dn/dx values 
        dn_drho = tf.divide(dn_dlogr, 10**log10_rhob) / np.log(10)
        dn_dt = tf.divide(dn_dlogt, 10**log10_temp) / np.log(10)
        
        # don't calculate grads for rhob_cgs_conversion and z
        return upstream * dn_drho, upstream * dn_dt, 0, 0
    
    nhi = n(log10_rhob, log10_temp, grid=False) # this is a 3D ndarray
    return tf.convert_to_tensor(nhi), grad

def main():
    start1 = time.time() # track the total duration
    
    ### load in the density and temperature fields as grids ###

    filename = "../../../../../cscratch1/sd/jupiter/sim2_z3_FGPA_cgs.h5"
    snap = snapshot.Snapshot(filename)
    
    # subsection shape
    shape = [2, 2, 1024]
    
    # string representing the subsection's dimensions, e.g. '4x4x4'
    dims_str = str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2])
    
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        dims_str += 'gpu'
    
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