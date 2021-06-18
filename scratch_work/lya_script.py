import h5py
import tensorflow as tf

# local modules
import snapshot

## constants
rho_crit_100_cgs = 1.8788200386793017e-29

# given a field and its name (e.g. 'tau_real'), save it into the HDF5 file at path
def write_field(field, name, path):
    f = h5py.File(path,'a') # 'a': read/write if file exists, create otherwise

    if name in f: # replace an existing dataset
        data = f[name]
        data[...] = field
    else: # create a new dataset
        f.create_dataset(name, data=field)

    f.close()
        
# given a snapshot, baryon density field, and temperature field, 
# return the n_HI field
def set_nhi(snap, rhob, temp):
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
    
    # TODO: initialize array/grids
    rhob = None
    temp = None
    
    nhi = set_nhi(snap, rhob, temp)
    # read nhi into an HDF5 file
    results_path = 'derived_fields.h5'
    write_field(nhi, 'nhi', results_path)
    
    snap.close()
    
    