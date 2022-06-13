## THALAS/Lya-TF
TensorFlow Hydrodynamics Analysis for Lyman Alpha Simulations (THALAS), originally named Lya-TF, is a fully differentiable tool for mapping fields of baryonic properties (baryon density, temperature, and velocity) to Lyman-alpha optical depth. Based on the C++ toolkit Gimlet.

## Dark Matter Reconstruction Demo
[Here](https://colab.research.google.com/drive/1hc8YgpVpT5k57tXn5Il8BCGYrtDrAEX6?usp=sharing) is a simple demonstration of using THALAS to reconstruct a dark matter (DM) density field from a given Lyman-alpha optical depth field. It uses the Fluctuating Gunn-Peterson Approximation (FGPA) to map DM density to baryonic properties, while THALAS maps the baryon fields to optical depth.

## Tutorial
1. Activate the Conda environment specified by lya-tf.yml.
2. Run the command `python3 -m numpy.f2py -c eos-t.f90 -m eos_t`. This creates the eos_t Python module, which is used in `eos.py`.
3. In `lya_fields.py`, specify the dimensions of the fields you want to compute by specifying `shape` in the main method. (For example, if you want to compute fields with shape 10x10x1024, set `shape` to `[10, 10, 1024]`.) 
4. Run `python lya_fields.py` in the command line.
5. Use `plot_fields.ipynb` (located in `lya_fields`) to visualize the computed fields.

## Contents
- `lya_fields` replicates the `/apps/lya_fields` application in Gimlet. It contains the main script, `lya_fields.py`, as well as dependencies, results files, and Jupyter notebooks.
- `lya-tf.yml` is the file specifying this project's Conda environment, which includes TensorFlow 2.4. (NOTE: Python 3.7 and NumPy 1.18 are necessary for step 2, which wraps eos-t.f90 in Python.)
- `data_exploration.ipynb` explores the HDF5 file containing the hydrodynamical simulation results.
- `test_gpus.ipynb` and `test_gpus.py` explore TensorFlow usage with GPUs.
