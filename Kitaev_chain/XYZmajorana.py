import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from XYZmajorana_class import XYZmajorana, local_marker, spectrum
from numpy import pi

start_time = time.time()

#%% Parameters

# Parameter space
gamma_vec = np.linspace(-8, 8, 80)    # Exponent of the maximum value for the XY couplings
X         = 1                         # X coupling
Y         = 1                         # Y coupling
Z         = 10                        # Z coupling
L         = 12                        # Length of the chain

# Definitions
av_marker = np.zeros(len(gamma_vec))
gap = np.zeros(av_marker.shape)
Id = np.eye(int(L))
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
S = np.kron(sigma_x, Id)              # Chiral symmetry of the chain (BdG x position space)

#%% Transition between +1 and -1



for i, gamma in enumerate(gamma_vec):
    print('Realisation: {}/{}'.format(i, len(gamma_vec)))

    # Hamiltonian
    chain = XYZmajorana(X, Y, Z, gamma, L)
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
    values = spectrum(rho)[0]

    # Local chiral marker
    marker = np.zeros((L,))
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho, S, j)
    av_marker[i] = np.mean(marker)

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))


#%% Figures

# Local marker as a function of theta
plt.figure()
plt.plot(gamma_vec, av_marker, ".b")
plt.xlim([gamma_vec[0], gamma_vec[-1]])
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$\delta$")
plt.title('Average Marker')
plt.show()

# # Energy gap for the ground state as a function of theta
plt.figure()
plt.plot(gamma_vec, gap, ".b")
plt.xlim([gamma_vec[0], gamma_vec[-1]])
# plt.ylim([-2, 2])
plt.ylabel('$gap$')
plt.xlabel("$\delta$")
plt.title('Gap')
plt.show()
