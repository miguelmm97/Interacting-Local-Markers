import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import matplotlib.ticker as ticker
from XYZmajorana_class import XYZmajorana, local_marker, spectrum
from numpy import pi

start_time = time.time()

#%% Parameters

# Parameter space
# gamma_vec = np.linspace(-8, 8, 80)    # Exponent of the maximum value for the XY couplings
gamma_vec = [0]
X         = 1                         # X coupling
Y         = 1                         # Y coupling
Z         = 0                        # Z coupling
L         = 12                         # Length of the chain

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
    Heven, disX, disY = chain.calc_sparse_Hamiltonian(parity='even', bc='periodic')
    Eeven, psi_even = eigsh(Heven, k=5, which='SA')
    Hodd = chain.calc_sparse_Hamiltonian(parity='odd', bc='periodic', dis_X=disX, dis_Y=disY)[0]
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

plt.figure()
plt.plot(range(0, 2*L), values, '.b', markersize=6)
plt.ylim(-0.5, 1.05)
plt.xlim(0, 2 * L)
plt.xlabel("$\\alpha$")
plt.ylabel("$n_\\alpha$")
plt.title('$\\rho$')
plt.show()


fig1, ax = plt.subplots(figsize=(6, 5))
ax.plot(range(0, 2*L), values, '.b', markersize=6)
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$n_\\alpha$')
ax.tick_params(which='major', width=0.75, labelsize=10)
ax.tick_params(which='major', length=7,  labelsize=10)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=3.5)
majorsy = [0, 0.5, 1]
minorsy = [0.25, 0.75]
majorsx = [-5]
minorsx = [-5]
majorsx_str = [""]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax.legend(bbox_to_anchor=(0.5, 0.7), frameon=False, fontsize=10, handletextpad=0.01)
# ax.text(28, 0.7, '$\delta$', fontsize=10)
# ax.text(2, 0.6, '$L=18$', fontsize=10)




# plt.tight_layout()
plt.savefig("OPDM.pdf", bbox_inches="tight")