import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from Jens2 import model, local_marker, spectrum
from numpy import pi

start_time = time.time()

#%% Parameters

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameter space
theta_vec = np.linspace(0, pi/2, 25)
t_vec = (1 / np.sqrt(6)) * np.sin(theta_vec)       # Nearest-neighbour hopping
Delta_vec = (1 / np.sqrt(2)) * np.cos(theta_vec)   # Nearest-neighbour pairing
Vint_vec = (2 / np.sqrt(6)) * np.sin(theta_vec)    # Nearest-neighbour density-density interactions
t2 = 0                                             # Next-to-nearest neighbour hopping
Delta2 = 0                                         # Next-to-nearest neighbour pairing
mu = 0                                             # Chemical potential
lamb = 0                                           # Onsite disorder
L = 8                                              # Length of the chain
parity = 'even'

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
    H = chain.calc_sparse_Hamiltonian(parity=parity, bc='periodic')
    E0, psi0 = eigsh(H, k=2, which='SA')
    gap[i] = E0[1] - E0[0]
    # H2 = chain.calc_Hamiltonian(parity=parity, bc='periodic')
    # E02, psi02 = spectrum(H2)

    # Single-particle density matrix
    # n_eig = 0
    # psi = V[:, n_eig]
    # psi0 = psi0[:, 0] / np.linalg.norm(psi0)
    psi0 = psi0[:, 0] / np.linalg.norm(psi0[:, 0])
    rho = chain.calc_opdm_from_psi(psi0, parity=parity)
    values = spectrum(rho)[0]

    # Local chiral marker
    marker = np.zeros((L,))
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho, S, j)
    av_marker[i] = np.mean(marker)

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))



#%% Figures

# Local marker
plt.figure()
plt.plot(theta_vec, av_marker, ".b")
plt.xlim([0, pi/2])
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$\\theta$")
plt.title('Average Marker')
plt.show()

# Energy gap for the ground state
plt.figure()
plt.plot(theta_vec, gap, ".b")
plt.xlim([0, pi/2])
# plt.ylim([-2, 2])
plt.ylabel('$gap$')
plt.xlabel("$\\theta$")
plt.title('Gap')
plt.show()

# Spectrum of the density matrix
plt.figure()
plt.plot(range(0, 2*L), values, '.b', markersize=6)
plt.ylim(0, 1)
plt.xlim(0, 2 * L)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.title('$\\rho$')
plt.show()
