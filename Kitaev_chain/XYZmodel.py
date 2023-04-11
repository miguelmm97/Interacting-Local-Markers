import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from Jens2 import model, local_marker, Ham_bdg, spectrum, ManyBodySpectrum

start_time = time.time()
#%% Parameters


aux = np.linspace(-1 / np.sqrt(3), 1 / np.sqrt(3), 10)
Vint_vec = np.concatenate((aux[::-1], aux))
t_vec = - Vint_vec / 2
Delta_vec = np.concatenate((-np.sqrt(0.5 - 0.75 * aux ** 2), np.sqrt(0.5 - 0.75 * aux[::-1] ** 2)))
x_plot = np.linspace(0, 2 * np.pi, t_vec.shape[0])

# t_vec = np.linspace(-1, 1, 1)            # Nearest-neighbour hopping
t2 = 0                                   # Next-to-nearest neighbour hopping
# Delta = 1                                # Nearest-neighbour pairing
Delta2 = 0                               # Next-to-nearest neighbour pairing
# Vint = 0.0                               # Density-density interactions
mu = 0                                   # Chemical potential
lamb = 0.5                                 # Onsite disorder
L = 16                                     # Length of the chain
parity = 'even'

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
av_marker = np.zeros(len(t_vec))
Id = np.eye(int(L))                       # Identity in position space
S = np.kron(sigma_x, Id)                  # Chiral symmetry of the chain (BdG x position space)

#%% Transition between +1 and -1

for i, t in enumerate(t_vec):
    print('Realisation: {}/{}'.format(i, len(t_vec)))

    # XYZ model for the specified parameters
    chain = model(t, t2, Vint_vec[i], mu, Delta_vec[i], Delta2, lamb, L)
    H = chain.calc_sparse_Hamiltonian(parity=parity, bc='periodic')  # Hamiltonian on the odd parity sector
    # H2 = chain.calc_Hamiltonian(parity='odd', bc='periodic')  # Hamiltonian on the odd parity sector
    E0, psi0 = eigsh(H, k=1, which='SA')
    # print(np.allclose(H.todense(), H2))
    # E02, psi0 = spectrum(H2)
    # print(E0, psi0.shape)
    # print(E02[0], psi02[:, 0].shape)

    # Single particle density matrix for the middle of the spectrum
    # n_eig = 0
    # psi = V[:, n_eig]
    psi0 = psi0[:, 0] / np.linalg.norm(psi0)
    rho = chain.calc_opdm_from_psi(psi0, parity=parity)  # Single particle density matrix (BdG x position space)
    values = spectrum(rho)[0]

    # Local chiral marker
    marker = np.zeros((L,))  # Definition of the marker vector
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho, S, j)  # Local marker at each site of the chain
    av_marker[i] = np.mean(marker)

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))

# Local marker
plt.figure()
plt.plot(x_plot, av_marker, ".b")
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$t$")
plt.title('Average Marker')
plt.show()

# Spectrum of the density matrix
# plt.figure()
# plt.plot(range(0, 2*L), values, '.b', markersize=6)
# plt.ylim(0, 1)
# plt.xlim(0, 2 * L)
# plt.xlabel("Eigenstate number")
# plt.ylabel("Eigenvalue")
# plt.title('$\\rho$')
# plt.show()

