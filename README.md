# SupplementaryCodes-for-Self-balanced-BO
# Python Code for Self-Balanced Bloch Oscillations

This repository contains the Python scripts used to generate the numerical results for the paper **“Demonstration of self-balance mechanism with Bloch oscillations in momentum bandgap engineering.”** The code reproduces the main dynamical features discussed in the manuscript, including k-gap amplified Bloch oscillations, self-balanced Bloch oscillations, channel-polarization diagnostics, and robustness tests under parameter detuning and disorder.

The simulations are performed in dimensionless units, following the parameter conventions described in the Supplemental Material. The time evolution is integrated using `solve_ivp` from SciPy. For clarity, the scripts are named according to the corresponding main-text or supplementary figures, such as Fig1d.py and coupling_disorder_in_FigS4.py.

This code is provided for research and reproducibility purposes. If you use this repository in your work, please cite our work:
DOI: https://doi.org/10.1103/snm3-mb3y
