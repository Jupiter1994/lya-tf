import h5py
import tensorflow as tf

# local modules
import universe

# basic class that handles a simulation snapshot
class Snapshot:
    
    # create a snapshot from an HDF5 file's path
    def __init__(self, path):
        self.file_path = path 
        snap = h5py.File(self.file_path,'r') # HDF file
        
        # Grid/domain properties
        self.shape = snap['domain'].attrs['shape'] # 1024^3
        self.size = snap['domain'].attrs['size']
        
        # Universe properties
        self.z = tf.Variable(snap['universe'].attrs['redshift'])
        self.scale_factor = tf.divide(1, self.z+1)
        omega_b = tf.Variable(snap['universe'].attrs['omega_b'])
        omega_m = tf.Variable(snap['universe'].attrs['omega_m'])
        omega_l = tf.Variable(snap['universe'].attrs['omega_l'])
        h = tf.Variable(snap['universe'].attrs['hubble'])
        sigma_8, n_s = tf.Variable(0.), tf.Variable(0.)

        self.univ = universe.Universe(omega_b, omega_m, omega_l, h, sigma_8, n_s)
        
        snap.close()
        
    # read in a field with a given path, e.g. '/native_fields/baryon_density'
    def read_field(self, path):
        snap = h5py.File(self.file_path,'r')
        field = snap[path]

        snap.close()
        
        return field
    
    # write a field into a given path, e.g. '/derived_fields/tau_real'
    def write_field(self, field, path):
        snap = h5py.File(self.file_path,'w')
        data = snap[path]
        data[...] = field

        snap.close()
    
    # print info about the snapshot
    def print_metadata(self):
        print('Snapshot info:\n')
        print('File path:', self.file_path)
        print()
        print('Domain shape:', self.shape)
        print('Domain size:', self.size)
        print()
        print('z =', self.z)