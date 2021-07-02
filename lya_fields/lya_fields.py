'''
Write the number density of neutral hydrogen (nHI), real- and redshift-space optical depths
      into an HDF5 file.
      
'''

import h5py
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
    Compute the neutral hydrogen number density (n_HI) field.
    Returns a Grid object.
    
    PARAMETERS
    ----------
    snap: a Snapshot object
    rhob: the baryon density field (a Grid object)
    temp: the temperature field (a Grid object)
    
    '''
    
    a = snap.scale_factor
    z = snap.z
    u = snap.universe
    h = u.h
    omega_b = u.omega_b
    
    # initialize nhi array and EOS object
    nhi_field = tf.zeros(rhob.shape, dtype='float64')
    eos_obj = eos.EOS_at_z(z)
    
    mean_rhob_cgs = omega_b * h*h * rho_crit_100_cgs
    a3_inv = 1.0 / (a * a * a)
    rhob_cgs = mean_rhob_cgs * a3_inv * rhob.field
    
    # run through EOS for each cell
    start2 = time.time()
    
    size = tf.size(rhob_cgs).numpy()
    # flatten the arrays
    elems = (tf.reshape(rhob_cgs, [size,]), tf.reshape(temp.field, [size]))
    nhi_field = tf.map_fn(eos_obj.nyx_eos_vec, elems, fn_output_signature=tf.float64)
    
    # un-flatten the result
    nhi_field = tf.reshape(nhi_field, rhob.shape)
    
#     elems = (rhob_cgs, temp.field)
#     nhi_field = tf.vectorized_map(eos_obj.nyx_eos, elems)
    
#     for i in range(rhob.shape[0]):
#         for j in range(rhob.shape[1]):
#             for k in range(rhob.shape[2]):
#                 rho = eos_obj.nyx_eos(rhob_cgs[i,j,k], temp.field[i,j,k])
#                 # assign rho to nhi_field[i,j,k]
#                 nhi_field = tf.tensor_scatter_nd_add(nhi_field, [[i,j,k]], [rho])
     
    print('EOS duration:', time.time() - start2)
    
    return grid.Grid(nhi_field, rhob.shape, rhob.size)

def main():
    start1 = time.time() # track the total duration
    
    ### load in the density and temperature fields as grids ###

    filename = "../../../../../cscratch1/sd/jupiter/sim2_z3_FGPA_cgs.h5"
    snap = snapshot.Snapshot(filename)
    
    # subsection shape
    shape = [1, 100, 100]
    
    # string representing the subsection's dimensions, e.g. '4x4x4'
    dims_str = str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2])
    
    # file paths to routine durations and results
    t_path = 'times/times' + dims_str + '.txt'
    results_path = 'results/tf_fields' + dims_str + '.h5'
    
    rhob = snap.read_field2(ds_path_rhob, shape)
    temp = snap.read_field2(ds_path_temp, shape)
    
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
    vpara = snap.read_field2(ds_path_vz, shape)
    
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