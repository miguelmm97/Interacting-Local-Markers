import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from XYZmajorana_class import model, local_marker, spectrum
from numpy import pi
import argparse
import h5py
import os
from datetime import date

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

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameter space
theta1 = np.linspace(-pi, 0, 10)
theta2 = np.linspace(0, pi, 50)
theta_vec = np.concatenate((theta1, theta2))        # General angle parameter
t_vec = (1 / np.sqrt(6)) * np.sin(theta_vec)        # Nearest-neighbour hopping
Delta_vec = (1 / np.sqrt(2)) * np.cos(theta_vec)    # Nearest-neighbour pairing
Vint_vec = (2 / np.sqrt(6)) * np.sin(theta_vec)     # Nearest-neighbour density-density interactions
t2 = 0                                              # Next-to-nearest neighbour hopping
Delta2 = 0                                          # Next-to-nearest neighbour pairing
mu = 0                                              # Chemical potential
lamb = 0.1                                          # Onsite disorder

# Definitions
av_marker = np.zeros(len(t_vec))
gap = np.zeros(av_marker.shape)
Id = np.eye(int(L))                                # Identity in position space
S = np.kron(sigma_x, Id)                           # Chiral symmetry of the chain (BdG x position space)

#%% Transition between +1 and -1

for i, (t, Delta, V) in enumerate(zip(t_vec, Delta_vec, Vint_vec)):
    print('Realisation: {}/{}'.format(i, len(t_vec)))

    # Hamiltonian
    chain = model(t, t2, V, mu, Delta, Delta2, lamb, L)
    Heven = chain.calc_sparse_Hamiltonian(parity='even', bc='periodic')
    Eeven, psi_even = eigsh(Heven, k=5, which='SA')
    Hodd = chain.calc_sparse_Hamiltonian(parity='odd', bc='periodic')
    Eodd, psi_odd = eigsh(Hodd, k=5, which='SA')

    # Select the many-body ground state
    E_full = np.sort(np.concatenate((Eeven, Eodd)))
    gap[i] = E_full[1] - E_full[0]
    if Eeven[0] < Eodd[0]: GS = psi_even[:, 0]; parity_gs = 'even'
    else: GS = psi_odd[:, 0]; parity_gs = 'odd'
    GS = GS / np.linalg.norm(GS)

    # Single-particle density matrix
    rho = chain.calc_opdm_from_psi(GS, parity=parity_gs)

    # Local chiral marker
    marker = np.zeros((L,))
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho, S, j)
    av_marker[i] = np.mean(marker)




#%% Saving the data

# Output data
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)
with h5py.File(filepath, 'w') as f:
    f.create_dataset("data", data=av_marker)
    f["data"].attrs.create("L",              data=L)
    f["data"].attrs.create("theta",          data=theta_vec)
    f["data"].attrs.create("t_vec",          data=t_vec)
    f["data"].attrs.create("Delta_vec",      data=Delta_vec)
    f["data"].attrs.create("Vint_vec",       data=Vint_vec)
    f["data"].attrs.create("t2",             data=t2)
    f["data"].attrs.create("Delta2",         data=Delta2)
    f["data"].attrs.create("mu",             data=mu)
    f["data"].attrs.create("lamb",           data=lamb)
    f["data"].attrs.create("Date",           data=str(date.today()))

