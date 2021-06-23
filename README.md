## Summary
Mapping baryon properties to Lyman-alpha flux with TensorFlow. Based on the C++ toolkit Gimlet.

## File Descriptions
- `lya-tf.yml` is the file specifying this project's Conda environment, which includes TensorFlow 2.4. (NOTE: Python 3.7 and NumPy 1.18 are needed in order to wrap eos-t.f90 in Python.)
- `data_exploration.ipynb` explores the HDF5 file containing the hydrodynamical simulation results.
- `scratch_work` contains Python scripts, classes, etc. that replicate Gimlet.