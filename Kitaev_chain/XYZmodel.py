import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from Jens2 import model, local_marker, Ham_bdg, spectrum, ManyBodySpectrum


#%% Parameters
t_vec = np.linspace(-1, 1, 10)            # Nearest-neighbour hopping
t2 = 0                                    # Next-to-nearest neighbour hopping
Delta = 1                                 # Nearest-neighbour pairing
Delta2 = 0                                # Next-to-nearest neighbour pairing
Vint = 0.5                                # Density-density interactions
mu = 0                                    # Chemical potential
lamb = 0.5                                # Onsite disorder
L = 12                                    # Length of the chain

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
    chain = model(t, t2, Vint, mu, Delta, Delta2, lamb, L)
    H = chain.calc_Hamiltonian(parity='odd', bc='periodic')  # Hamiltonian on the odd parity sector
    E, V = np.linalg.eigh(H)                                 # Spectrum in the even parity sector

    # Single particle density matrix for the middle of the spectrum
    n_eig = 1
    psi = V[:, n_eig]
    psi = psi / np.linalg.norm(psi)
    rho = chain.calc_opdm_from_psi(psi, parity='odd')  # Single particle density matrix (BdG x position space)
    values = spectrum(rho)[0]

    # Local chiral marker
    marker = np.zeros((L,))  # Definition of the marker vector
    for j in range(L):
        marker[j] = local_marker(L, np.arange(L), rho, S, i)  # Local marker at each site of the chain
    av_marker[i] = np.mean(marker)

plt.figure()
plt.plot(t_vec, av_marker, ".b")
plt.ylim([-2, 2])
plt.ylabel('$\\nu(r)$')
plt.xlabel("$t$")
plt.title('Average Marker')
plt.show()

