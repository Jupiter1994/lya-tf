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

<<<<<<< HEAD
# subsection shape
shape = [1, 1024, 1024]
=======
save_grads = False

## subsection shape
shape = [1, 1, 1024]
>>>>>>> gradient
    
## define dataset paths
ds_path_rhob = "/native_fields/baryon_density"
ds_path_temp = "/native_fields/temperature"
ds_path_vx   = "/native_fields/velocity_x"
ds_path_vy   = "/native_fields/velocity_y"
ds_path_vz   = "/native_fields/velocity_z"

ds_path_nhi      = "/derived_fields/HI_number_density"
ds_path_tau_real = "/derived_fields/tau_real"
ds_path_tau_red  = "/derived_fields/tau_red"

## define paths to store the gradients 

# should have the same shape as the input fields
path_dn_dr = "/n_gradients/dn_dr"
path_dn_dt = "/n_gradients/dn_dt"

# should have the shape [shape, shape],
# e.g. [1, 1, 1024, 1, 1, 1024] for one skewer
path_dtreal_dr = "/tau_jacobians/dtreal_drho"
path_dtred_dr = "/tau_jacobians/dtred_drho"
path_dtreal_dt = "/tau_jacobians/dtreal_dt"
path_dtred_dt = "/tau_jacobians/dtred_dt"

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
    Returns the n_HI grid.
    
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
    eos_obj = eos.EOS_at_z(z, rhob_cgs_conversion)
    nhi_field = eos_obj.compute_nhi(log10_rhob, log10_temp)
    #nhi_field = compute_nhi(log10_rhob, log10_temp, rhob_cgs_conversion, z)
    
    print('EOS duration:', time.time() - start2)
    
    return grid.Grid(nhi_field, rhob.shape, rhob.size)

def main():
    
    # track routine durations
    times = open(t_path, 'w')
    
    start1 = time.time() # track the total duration
    
    ### load in the density and temperature fields as grids ###

    filename = "../../../../../cscratch1/sd/jupiter/sim2_z3_FGPA_cgs.h5"
    snap = snapshot.Snapshot(filename)
    
    # string representing the subsection's dimensions, e.g. '4x4x4'
    dims_str = str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2])
    
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        dims_str += 'gpu'
    
    # file paths to routine durations and results
    t_path = 'times/times' + dims_str + '.txt'
    results_path = 'results/tf_fields' + dims_str + '.h5'
    
    with tf.GradientTape(persistent=True) as tape:
        rhob = snap.read_subfield(ds_path_rhob, shape)
        temp = snap.read_subfield(ds_path_temp, shape)

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
        vpara = snap.read_subfield(ds_path_vz, shape)

        start = time.time()
        tau_red = gmlt_spec_od_grid(snap.universe, snap.z, nhi.size,
                nhi.field, temp.field, vpara.field, nhi.field.shape[2])
        times.write('tau_red duration = ' + str(time.time() - start) + '\n')

        write_field(tau_red.field, 'tau_red', results_path)
    
    if save_grads:
        dn_dr = tape.gradient(nhi.field, rhob.field)
        dn_dt = tape.gradient(nhi.field, temp.field)
        write_field(dn_dr, path_dn_dr, results_path)
        write_field(dn_dt, path_dn_dt, results_path)
        
        dtreal_dr = tape.jacobian(tau_real.field, rhob.field)
        dtred_dr = tape.jacobian(tau_red.field, rhob.field)
        
        # note: for one skewer, the below lines may throw an out-of-memory error
        dtreal_dt = tape.jacobian(tau_real.field, temp.field)
        dtred_dt = tape.jacobian(tau_red.field, temp.field)
        
        write_field(dtreal_dr, path_dtreal_dr, results_path)
        write_field(dtred_dr, path_dtred_dr, results_path)
        write_field(dtreal_dt, path_dtreal_dt, results_path)
        write_field(dtred_dt, path_dtred_dt, results_path)
    
    times.write('Total duration = ' + str(time.time() - start1) + '\n')
    times.close()
    
    print('Total duration:', time.time() - start1)
    
# prevent this script from being run accidentally
if __name__ == '__main__':
    main()