import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from Jens2 import model, local_marker, Ham_bdg, spectrum


#%% Parameters
t = 1                                    # Nearest-neighbour hopping
t2 = 0                                    # Next-to-nearest neighbour hopping
Delta = 0                                # Nearest-neighbour pairing
Delta2 = 0                               # Next-to-nearest neighbour pairing
V = 0                                     # Density-density interactions
mu = -0.0                               # Chemical potential
lamb = 0                                # Onsite disorder
L = 10                                  # Length of the chain

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


#%% Hamiltonian
chain = model(t, t2, V, mu, Delta, Delta2, lamb, L)          # Kitaev chain with the specified parameters
H = chain.calc_Hamiltonian(parity='even', bc='periodic')     # Hamiltonian on the even parity sector
Energy, Evecs = np.linalg.eigh(H)                            # Spectrum in the even parity sector

# Single particle density matrix
#
psi = Evecs[:, 1]                                         # Particula eigenstate in the parity sector
psi = psi / np.linalg.norm(psi)
rho = chain.calc_opdm_from_psi(psi, parity='even')           # Single particle density matrix (BdG x position space)
values = np.linalg.eigvalsh(rho)                             # Occupation numbers (spectrum of rho)
#
# # Local chiral marker
# Id = np.eye(int(L))                                          # Identity in position space
# S = np.kron(sigma_x, Id)                                     # Chiral symmetry of the chain (BdG x position space)
# marker = np.zeros((L, ))                                     # Definition of the marker vector
# for i in range(L):
#     marker[i] = local_marker(L, np.arange(L), rho, S, i)     # Local marker at each site of the chain
# print(np.mean(marker))
H2 = Ham_bdg(L, t, 0, mu)
E2 = spectrum(H2)[0]

#%% Figures

# Spectrum of the density matrix
plt.figure()
plt.plot(range(len(Energy)), Energy, '.b', markersize=6)
# plt.plot(r, Energy[r], '.r', markersize=6)
# plt.ylim(-0.1, 0.1)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.title('$H$')

plt.figure()
plt.plot(range(len(E2)), E2, '.b', markersize=6)
# plt.plot(r, Energy[r], '.r', markersize=6)
# plt.ylim(-0.1, 0.1)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.title('$Hsp$')


# Spectrum of the density matrix
plt.figure()
plt.plot(range(0, 2*L), values, '.b', markersize=6)
# plt.ylim(-0.1, 0.1)
plt.xlim(0, 2 * L)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.title('$\\rho$')
#
#
# # Local marker
# plt.figure()
# plt.plot(np.arange(L), marker, ".b")
# plt.ylim([-2, 2])
# plt.ylabel('$\\nu(r)$')
# plt.xlabel("$x$")
# plt.title('Marker')
# plt.show()


# Plots
# plt.plot(range(0, 2 * L), val, '.b', markersize=6)  # Plot of the energy
# # plt.ylim(-0.1, 0.1)
# plt.xlim(0, 2 * L)
# plt.xlabel("Eigenstate number")
# plt.ylabel("Eigenvalue")
# plt.title("$i[P,R]$, $S= \sigma_x$")
# plt.show()



