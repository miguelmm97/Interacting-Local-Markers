import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from Jens2 import model, local_marker, Ham_bdg, spectrum, ManyBodySpectrum


#%% Parameters
t = 1                                     # Nearest-neighbour hopping
t2 = 0                                    # Next-to-nearest neighbour hopping
Delta = 1                                 # Nearest-neighbour pairing
Delta2 = 0                                # Next-to-nearest neighbour pairing
V = 0                                     # Density-density interactions
mu = 0.01                                 # Chemical potential
lamb = 0.01                                # Onsite disorder
L = 4                                     # Length of the chain

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


#%% Hamiltonian
chain = model(t, t2, V, mu, Delta, Delta2, lamb, L)          # Kitaev chain with the specified parameters
Heven = chain.calc_Hamiltonian(parity='even', bc='periodic')     # Hamiltonian on the even parity sector
Eeven, Veven = np.linalg.eigh(Heven)


# Hodd = chain.calc_Hamiltonian(parity='odd', bc='periodic')     # Hamiltonian on the even parity sector
# Eeven, Veven = np.linalg.eigh(Heven)                            # Spectrum in the even parity sector
# Eodd, Vodd = np.linalg.eigh(Hodd)                            # Spectrum in the even parity sector
# E = np.concatenate((Eeven, Eodd))
# idx = E.argsort()
# E = E[idx]
# H2 = Ham_bdg(L, t, Delta, mu)
# E2 = spectrum(H2)[0]
# Ep = E2[E2 >= 0]
# Esp = ManyBodySpectrum(Ep)
#

# Single particle density matrix
psi = Veven[:, 1]                                         # Particula eigenstate in the parity sector
psi = psi / np.linalg.norm(psi)
rho = chain.calc_opdm_from_psi(psi, parity='even')           # Single particle density matrix (BdG x position space)
values = np.linalg.eigvalsh(rho)
P = np.zeros((2 * L,))
P[L:] = 1
# if np.allclose(spectrum(rho)[0], P, rtol=1e-15):
#     print('Good job!:)')
# else:
#     raise AssertionError("OPDM not a projector for product states!")
# Occupation numbers (spectrum of rho)
#
# # Local chiral marker
# Id = np.eye(int(L))                                          # Identity in position space
# S = np.kron(sigma_x, Id)                                     # Chiral symmetry of the chain (BdG x position space)
# marker = np.zeros((L, ))                                     # Definition of the marker vector
# for i in range(L):
#     marker[i] = local_marker(L, np.arange(L), rho, S, i)     # Local marker at each site of the chain
# print(np.mean(marker))




#%% Figures

# Spectrum of the density matrix
plt.figure()
plt.plot(range(len(Eeven)), Eeven, '.y', markersize=15)
# plt.plot(range(len(Esp)), Esp, '.b', markersize=5)
# plt.plot(r, Energy[r], '.r', markersize=6)
# plt.ylim(-0.1, 0.1)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.title('$H$')
plt.show()

# plt.figure()
# plt.plot(range(len(E2)), E2, '.b', markersize=6)
# # plt.plot(r, Energy[r], '.r', markersize=6)
# # plt.ylim(-0.1, 0.1)
# plt.xlabel("Eigenstate number")
# plt.ylabel("Energy")
# plt.title('$Hsp$')
#
#
# # Spectrum of the density matrix
plt.figure()
plt.plot(range(0, 2*L), values, '.b', markersize=6)
# plt.ylim(-0.1, 0.1)
plt.xlim(0, 2 * L)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.title('$\\rho$')
plt.show()
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



