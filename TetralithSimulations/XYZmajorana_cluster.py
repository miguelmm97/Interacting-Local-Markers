import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from XYZmajorana_class import XYZmajorana, local_marker, spectrum, band_flattening
from numpy import pi
import argparse
import h5py
import os
from datetime import date

start_time = time.time()

# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Argument parser for the XYZ model simulation')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params_XYZ.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='outXYZ')
args = parser.parse_args()

# Variables that we iterate with the cluster
L = None          # System size

# Input data
if args.line is not None:
    print("Line number:", args.line)
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == args.line:
                params = line.split()
                L = int(params[0])
else:
    raise IOError("No line number was given")

if L is None:
    raise ValueError("Size is none")

#%% Parameters

# Parameter space
gamma_vec = np.linspace(-8, 8, 80)    # Exponent of the maximum value for the XY couplings
X         = 1                         # X coupling
Y         = 1                         # Y coupling
Z         = 10                        # Z coupling

# Definitions
av_marker = np.zeros(len(gamma_vec))
gap = np.zeros(av_marker.shape)
Id = np.eye(int(L))
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
S = np.kron(sigma_x, Id)              # Chiral symmetry of the chain (BdG x position space)
rho_gamma = np.zeros((gamma_vec.shape[0], int(2 * L), int(2 * L)))
#%% Transition between +1 and -1

for i, gamma in enumerate(gamma_vec):
    print('Realisation: {}/{}'.format(i, len(gamma_vec)))

    # Hamiltonian
    chain = XYZmajorana(X, Y, Z, gamma, L)
    Heven, disX, disY = chain.calc_sparse_Hamiltonian(parity='even', bc='periodic')
    Eeven, psi_even = eigsh(Heven, k=5, which='SA')
    Hodd = chain.calc_sparse_Hamiltonian(parity='odd', bc='periodic', dis_X=disX, dis_Y=disY)[0]
    Eodd, psi_odd = eigsh(Hodd, k=5, which='SA')

    # Select the many-body ground state
    E_full = np.sort(np.concatenate((Eeven, Eodd)))
    gap[i] = E_full[1] - E_full[0]
    if Eeven[0] < Eodd[0]: GS = psi_even[:, 0]; parity_gs = 'even'
    else: GS = psi_odd[:, 0]; parity_gs = 'odd'
    GS = GS / np.linalg.norm(GS)

    # Single-particle density matrix
    rho = chain.calc_opdm_from_psi(GS, parity=parity_gs)
    rho_gamma[i, :, :] = rho
    values = spectrum(rho)[0]
    rho_flat = band_flattening(rho)

    # Local chiral marker
    marker = np.zeros((L,))
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho_flat, S, j)
    av_marker[i] = np.mean(marker)

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))


#%% Saving the data

# Output data
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)
with h5py.File(filepath, 'w') as f:
    f.create_dataset("marker_data",                 data=av_marker)
    f.create_dataset("opdm_data",                   data=rho_gamma)
    f.create_dataset("gap_data",                    data=gap)
    f["marker_data"].attrs.create("L",              data=L)
    f["marker_data"].attrs.create("gamma",          data=gamma_vec)
    f["marker_data"].attrs.create("X",              data=X)
    f["marker_data"].attrs.create("Y",              data=Y)
    f["marker_data"].attrs.create("Z",              data=Z)
    f["marker_data"].attrs.create("Date",           data=str(date.today()))

