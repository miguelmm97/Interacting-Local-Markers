# Generate parameter data to submit to the cluster
import numpy as np

Nsamples = 10                  # Number of disorder smaples
L = np.array([10])  #np.array([12, 14, 16, 18])  # System sizes
Lsim = np.repeat(L, Nsamples)


with open('params_XYZ_try.txt', 'w') as f:
    for l in Lsim:
        f.write('{}\n'.format(l))