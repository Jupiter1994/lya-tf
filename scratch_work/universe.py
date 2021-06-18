import tensorflow as tf

## constants
Mpc_cgs = 3.08568e24
H0_100_cgs = 3.2407767493712897e-18
    
# basic class that handles the flat LCDM Universe model
class Universe:
    
    # set the universe properties
    # note: the six given parameters are tf variables
    def __init__(self, omega_b, omega_m, omega_l, h, sigma_8, n_s):
        # load in properties
        self.omega_b = omega_b
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h
        self.sigma_8 = sigma_8
        self.n_s = n_s

        # derived properties
        self.omega_dm = omega_m - omega_b
        self.H_0 = 100.0 * h
        self.H_0_cgs = H0_100_cgs * h
        
    # accessor methods
    def omega_b(self): return self.omega_b
    def omega_dm(self): return self.omega_dm
    def omega_m(self): return self.omega_m
    def omega_l(self): return self.omega_l
    def h(self): return self.h
    def sigma_8(self): return self.sigma_8
    def n_s(self): return self.n_s
    def H_0(self): return self.H_0
    
    # z is a number
    def E(self, z):
        z = tf.constant(z, dtype='float64')
        zz = 1.0 + z
        return tf.math.sqrt(self.omega_m * tf.math.pow(zz,3) + self.omega_l)


    def H(self, z):
        return self.H_0 * self.E(z)
    

    def da_dt(self, z):
        return self.H(z) / (1.0 + z)
    

    def chi_to_proper_cgs(self, chi, z):
        # chi is in comoving Mpc/h.
        return tf.divide(chi * Mpc_cgs, self.h) / (1.0 + z)
    

    def chi_to_velocity_cgs(self, chi, z):
        # chi is in comoving Mpc/h.
        # v = da/dt L
        #   = H_0 L E(z) / (1 + z)
        #   = 100 h km/s/Mpc chi Mpc/h E(z) / (1 + z)
        #   = 100 km/s chi E(z) / (1 + z)
        # since we want v in cm/s, return 1e5 times the above
        return 1.0e7 * chi * self.E(z) / (1.0 + z)
    
    # TODO: implement universe.cc methods
    def comoving_distance(z):
        pass
    
    def z_of_chi(chi):
        pass
    