import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from Jens2 import model, local_marker, spectrum, Ham_bdg, ManyBodySpectrum
from numpy import pi

start_time = time.time()

#%% Parameters

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameter space
theta_vec = np.linspace(-pi, pi, 100)               # General angle parameter
t_vec = (1 / np.sqrt(6)) * np.sin(theta_vec)       # Nearest-neighbour hopping
Delta_vec = (1 / np.sqrt(2)) * np.cos(theta_vec)   # Nearest-neighbour pairing
Vint_vec = (2 / np.sqrt(6)) * np.sin(theta_vec)    # Nearest-neighbour density-density interactions
# t_vec = np.linspace(-1, 1, 20)
# Delta_vec = np.repeat(-1, t_vec.shape[0])
# Vint_vec = np.repeat(0.1, t_vec.shape[0])
# t_vec = [1]; Delta_vec = [-1]; Vint_vec = [0]
t2 = 0                                              # Next-to-nearest neighbour hopping
Delta2 = 0                                          # Next-to-nearest neighbour pairing
mu = 0                                              # Chemical potential
lamb = 0.1                                            # Onsite disorder
L = 12                                              # Length of the chain
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
    Heven = chain.calc_sparse_Hamiltonian(parity='even', bc='periodic')
    Eeven, psi_even = eigsh(Heven, k=11, which='SA')
    Hodd = chain.calc_sparse_Hamiltonian(parity='odd', bc='periodic')
    Eodd, psi_odd = eigsh(Hodd, k=11, which='SA')

    # Select the many-body ground state
    E_full = np.sort(np.concatenate((Eeven, Eodd)))
    gap[i] = E_full[1] - E_full[0]
    if Eeven[0] < Eodd[0]: GS = psi_even[:, 0]; parity_gs = 'even'
    else: GS = psi_odd[:, 0]; parity_gs = 'odd'
    GS = GS / np.linalg.norm(GS)

    # H2 = chain.calc_Hamiltonian(parity=parity, bc='periodic')
    # E02, psi02 = spectrum(H2)

    # Single-particle density matrix
    # n_eig = 0
    # psi = V[:, n_eig]
    # psi0 = psi0[:, 0] / np.linalg.norm(psi0)
    # psi0 = psi0[:, 0] / np.linalg.norm(psi0[:, 0])
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
plt.plot(theta_vec, av_marker, ".b")
plt.xlim([theta_vec[0], theta_vec[-1]])
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$\\theta$")
plt.title('Average Marker')
plt.show()
#
# # Energy gap for the ground state as a function of theta
plt.figure()
plt.plot(theta_vec, gap, ".b")
plt.xlim([theta_vec[0], theta_vec[-1]])
# plt.ylim([-2, 2])
plt.ylabel('$gap$')
plt.xlabel("$\\theta$")
plt.title('Gap')
plt.show()

# # Local marker as a function of t, Delta or V
# plt.figure()
# plt.plot(t_vec, av_marker, ".b")
# plt.xlim(t_vec[0], t_vec[-1])
# plt.ylim([-2, 2])
# plt.ylabel('$\\nu(r)$')
# plt.xlabel("$t$")
# plt.title('Average Marker')
# plt.text(t_vec[-1]-0.5, 1.5, '$\Delta=$ {}, $V=$ {}'.format(Delta_vec[0], Vint_vec[0]))
# plt.show()
#
# # Energy gap for the ground state as a function of t, Delta or V
# plt.figure()
# plt.plot(t_vec, gap, ".b")
# plt.xlim(t_vec[0], t_vec[-1])
# plt.ylim(0, 3*np.abs(t_vec[0]))
# plt.ylabel('$gap$')
# plt.xlabel("$t$")
# plt.title('Gap')
# plt.show()

# Spectrum of the density matrix
# plt.figure()
# plt.plot(range(0, 2*L), values, '.b', markersize=6)
# plt.ylim(0, 1)
# plt.xlim(0, 2 * L)
# plt.xlabel("Eigenstate number")
# plt.ylabel("Eigenvalue")
# plt.title('$\\rho$')
# plt.show()


# # Many body spectrum
# plt.figure()
# plt.plot(np.arange(0, E02.shape[0]), E02, '.b')