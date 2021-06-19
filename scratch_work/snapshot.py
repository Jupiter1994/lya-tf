import h5py
import tensorflow as tf

# local modules
import universe

class Snapshot:
    '''
    Class that handles a simulation snapshot. 

    '''

    def __init__(self, path):
        '''
        Create a snapshot from an HDF5 file.
        
        PARAMETERS
        ----------
        path: path to the HDF5 file
        
        '''
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
        
    def read_field(self, path):
        '''
        Read in a field from the snapshot's source file.
        
        PARAMETERS
        ----------
        path: the field's path within the file, e.g. '/native_fields/baryon_density'
        
        '''
        snap = h5py.File(self.file_path,'r')
        field = tf.Variable(snap[path][()]) # convert ndarray to variable

        snap.close()
        
        return field
    
    def write_field(self, field, path):
        '''
        Write a field into the snapshot's source file.
        
        PARAMETERS
        ----------
        path: the path to write the field into, e.g. '/derived_fields/tau_real'
        
        '''
        snap = h5py.File(self.file_path,'w')
        data = snap[path]
        data[...] = field

        snap.close()
    
    def print_metadata(self):
        '''
        Print info about the snapshot.
        
        '''
        print('Snapshot info:\n')
        print('File path:', self.file_path)
        print()
        print('Domain shape:', self.shape)
        print('Domain size:', self.size)
        print()
        print('z =', self.z)