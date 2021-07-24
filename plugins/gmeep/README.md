# gmeep 0.0.2

GDS Meep and MPB interface based on [meep example](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/)

# Port naming convention

f, s_1_1_1_1_a, s_1_1_1_1_m

where the first column is the freqency, a: angle in rad, m: module

S11 = S11m * exp(1j*angle)
S11m = abs(S11)
S11a = angle(S11)

# Objectives

- Compute Transmission for a simulation
- Compute full Sparameters
- Compute modes with MPB
