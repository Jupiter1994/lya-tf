'''
Write the number density of neutral hydrogen (nHI), real- and redshift-space optical depths
      into an HDF5 file.
      
'''

import h5py
import tensorflow as tf

# local modules
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
    f = h5py.File(file_path,'a') # 'a': read/write if file exists, create otherwise

    if name in f: # replace an existing dataset
        data = f[dset_path]
        data[...] = field.numpy()
    else: # create a new dataset
        f.create_dataset(dset_path, data=field.numpy())

    f.close()
        
def set_nhi(snap, rhob, temp):
    '''
    Compute the neutral hydrogen number density (n_HI) field.
    
    PARAMETERS
    ----------
    snap: a Snapshot object
    rhob: the baryon density field
    temp: the temperature field
    
    '''
    
    a = snap.scale_factor
    z = snap.z
    u = snap.universe
    h = u.h()
    omega_b = u.omega_b()
    
    # TODO: initialize nhi array/grid
    nhi = None
       
    mean_rhob_cgs = omega_b * h*h * rho_crit_100_cgs
    a3_inv = 1.0 / (a * a * a)
    
    # TODO: run through EOS for each cell
#     for (size_t i = 0; i < rhob->size(); ++i) {
#         double rhob_cgs = mean_rhob_cgs * a3_inv * rhob_array[i];
#         nyx_eos(&z, &rhob_cgs, temp_array + i, nhi_array + i);
#     }
    
    return nhi

def main():
    filename = "../../../../cscratch1/sd/jupiter/sim2_z3_FGPA_cgs.h5"
    snap = snapshot.Snapshot(filename)
    
    # TODO: initialize arrays/grids after loading in the tensors?
    rhob = snap.read_field(ds_path_rhob)
    temp = snap.read_field(ds_path_temp)
    
    ## TODO: implement set_nhi
    nhi = set_nhi(snap, rhob, temp)
    
    # read nhi into an HDF5 file
    results_path = 'derived_fields_test.h5'
    write_field(nhi, 'nhi', results_path)
    
    ## calculate optical depth fields
    
    # real-space tau
    vpara = tf.zeros(rhob.shape)
    # TODO: implement gmlt_spec_od_grid
#   tau = gmlt_spec_od_grid(snap.universe, snap.z, nhi->dist, nhi->array,
#         temp->array, vpara->array, dist->n(2), tau->array)
    write_field(tau, 'tau_real', results_path)
    
    # redshift-space tau
    vpara = snap.read_field(ds_path_vz)
    # TODO: implement gmlt_spec_od_grid
#   tau = gmlt_spec_od_grid(snap.universe, snap.z, nhi->dist, nhi->array,
#         temp->array, vpara->array, dist->n(2), tau->array)
    write_field(tau, 'tau_red', results_path)

    
    snap.close()
    
    