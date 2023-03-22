import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from Jens2 import model, local_marker, Ham_bdg, spectrum, ManyBodySpectrum


#%% Parameters
t = 3                                     # Nearest-neighbour hopping
t2 = 0.2                                    # Next-to-nearest neighbour hopping
Delta = t                                 # Nearest-neighbour pairing
Delta2 = t2                               # Next-to-nearest neighbour pairing
V = t2                                    # Density-density interactions
mu = 0                                    # Chemical potential
lamb = 0.5                                # Onsite disorder
L = 12                                    # Length of the chain

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


#%% Hamiltonian

# Many body calculation
chain = model(t, t2, V, mu, Delta, Delta2, lamb, L)       # Kitaev chain with the specified parameters
H = chain.calc_Hamiltonian(parity='odd', bc='periodic')   # Hamiltonian on the odd parity sector
E, V = np.linalg.eigh(H)                                  # Spectrum in the even parity sector

n_eig = 10
psi = V[:, n_eig]
psi = psi / np.linalg.norm(psi)
rho = chain.calc_opdm_from_psi(psi, parity='odd')        # Single particle density matrix (BdG x position space)
values = spectrum(rho)[0]
# chain.check_rho_eigenstates(V)

# Local chiral marker
Id = np.eye(int(L))                                          # Identity in position space
S = np.kron(sigma_x, Id)                                     # Chiral symmetry of the chain (BdG x position space)
marker = np.zeros((L, ))                                     # Definition of the marker vector
for i in range(L):
    marker[i] = local_marker(L, np.arange(L), rho, S, i)     # Local marker at each site of the chain
print(np.mean(marker))



#%% Figures

# Spectrum of the Hamiltonian
plt.figure()
plt.plot(range(len(E)), E, '.y', markersize=15)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.title('$H$')
plt.show()


# # Spectrum of the density matrix
plt.figure()
plt.plot(range(0, 2*L), values, '.b', markersize=6)
plt.ylim(0, 1)
plt.xlim(0, 2 * L)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.title('$\\rho$')
plt.show()
#
# # Local marker
plt.figure()
plt.plot(np.arange(L), marker, ".b")
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$x$")
plt.title('Marker')
plt.show()


