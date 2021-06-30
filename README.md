## Lya-TF
Mapping baryon properties to Lyman-alpha flux with TensorFlow. Based on the C++ toolkit Gimlet.

## Tutorial
1. Activate the Conda environment specified by lya-tf.yml.
2. In `lya_fields.py`, specify the dimensions of the fields you want to compute by specifying `n`. (For example, if you want to compute fields with size 20^3, set `n` to 20.) 
3. Run `python lya_fields.py` in the command line.
4. Use `plot_fields.ipynb` (located in `lya_fields`) to visualize the computed fields.

## Contents
- `lya_fields` replicates the `/apps/lya_fields` application in Gimlet. It contains the main script, `lya_fields.py`, as well as dependencies, results files, and Jupyter notebooks.
- `lya-tf.yml` is the file specifying this project's Conda environment, which includes TensorFlow 2.4. (NOTE: Python 3.7 and NumPy 1.18 are needed in order to wrap eos-t.f90 in Python.)
- `data_exploration.ipynb` explores the HDF5 file containing the hydrodynamical simulation results.