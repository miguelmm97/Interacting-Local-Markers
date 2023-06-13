# Generate parameter data to submit to the cluster
import numpy as np

Nsamples = 300                  # Number of disorder smaples
L = np.array([18])              # System sizes
Lsim = np.repeat(L, Nsamples)


with open('params_XYZ_18.txt', 'w') as f:
    for l in Lsim:
        f.write('{}\n'.format(l))