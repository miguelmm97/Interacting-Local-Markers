import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from XYZmajorana_class import model, local_marker, spectrum

start_time = time.time()
#%% Parameters
# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameter space
t = -1                                     # Nearest-neighbour hopping
t2 = 0                                     # Next-to-nearest neighbour hopping
Delta = 1                                  # Nearest-neighbour pairing
Delta2 = 0                                 # Next-to-nearest neighbour pairing
Vint = 0.5                                 # Density-density interactions
mu = 0                                     # Chemical potential
lamb_vec = np.linspace(0, 10, 10)          # Onsite disorder
L = 9                                      # Length of the chain
parity = 'even'

# Definitions
av_marker = np.zeros(len(lamb_vec))
Id = np.eye(int(L))                       # Identity in position space
S = np.kron(sigma_x, Id)                  # Chiral symmetry of the chain (BdG x position space)

#%% Transition between +1 and -1

for i, lamb in enumerate(lamb_vec):
    print('Realisation: {}/{}'.format(i, len(lamb_vec)))

    # Hamiltonian
    chain = model(t, t2, Vint, mu, Delta, Delta2, lamb, L)
    H = chain.calc_sparse_Hamiltonian(parity=parity, bc='periodic')
    E0, psi0 = eigsh(H, k=1, which='SA')
    # H2 = chain.calc_Hamiltonian(parity='odd', bc='periodic')
    # E02, psi0 = spectrum(H2)

    # Single particle density matrix
    # n_eig = 0
    # psi = V[:, n_eig]
    psi0 = psi0[:, 0] / np.linalg.norm(psi0)
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
plt.plot(lamb_vec, av_marker, ".b")
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

