#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:48:54 2022

@author: jensba
"""
import random

import numpy as np
from dataclasses import dataclass
from scipy import sparse
from scipy.stats import unitary_group
import itertools
import matplotlib.pyplot as plt


def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates


def local_marker(L_x, x, P, S, site):
    """
    Calculates the local CS marker for the specified site on a 1d chain in the BdG basis

    ----------
    Parameters:
    L_x, L_y, L_z: Number of lattice sites in each direction
    n_orb : Number of orbitals
    n_sites: Number of lattice sites
    x, y, z: Vectors with the position of the lattice sites
    P : Valence band projector
    S: Chiral symmetry operator of the model
    site: Number of the site we calculate the marker on

    ----------
    Returns:
    Marker vector on each site
    """

    # Position operators  (take the particular site to be as far from the branch cut as possible)
    half_Lx = np.floor(L_x / 2)
    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + np.heaviside(half_Lx - x[site],
                                                                                                 0) * abs(
        half_Lx - x[site])
    x = (x + deltaLx) % L_x  # Relabel of the operators
    X = np.concatenate((x, x))  # X vector in BdG basis
    X = np.reshape(X, (len(X), 1))  # Column vector x

    # Marker calculation
    M = P @ S @ (X * P)  # Marker operator (BdG x position space)
    marker = -2 * (M[site, site] + M[L_x + site, L_x + site])  # Trace over each site

    return marker


def bin_to_n(s, L):
    """
    Given an integer s representing a basis state in an L site system,
    return the occupation number representation as an array.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    s : integer
    L : number of sites

    Returns
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1

    Example
    ----------
    >>> bin_to_n(10,4)
    array([0, 1, 0, 1])

    """

    return np.array([int(x) for x in reversed(bin(s)[2:].zfill(L))])


def n_to_bin(n):
    """
    Given an occupation number representation of a state n,
    return the integer corresponding to its binary representation.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1

    Returns
    ----------
    s : integer

    Example
    ----------
    >>> n_to_bin([0,1,0,1])
    10

    """

    return int(''.join(str(x) for x in reversed(n)), 2)


def Ham_bdg(L, t, delta, mu):
    sigma = [np.array([[1, 0], [0, 1]]),  # Pauli 0
             np.array([[0, 1], [1, 0]]),  # Pauli x
             np.array([[0, -1j], [1j, 0]]),  # Pauli y
             np.array([[1, 0], [0, -1]])]  # Pauli z

    # Real space basis
    vector = np.zeros(L);
    basis = []
    for index in range(L):
        vector[index] = 1
        basis.append(vector)
        vector = np.zeros(L)

    # Real space projectors
    n_offdiag = []  # |n><n+1|
    n_diag = []  # |n><n|
    for i in range(L):
        j = np.mod(i + 1, L)
        n_offdiag.append(np.outer(basis[i], basis[j]))
        n_diag.append(np.outer(basis[i], basis[i]))

    n_offdiag_tot = sum(n_offdiag)  # sums all n_ matrices as the tensor product is linear
    n_diag_tot = sum(n_diag)

    tau = -(t * sigma[3] + 1j * delta * sigma[2])
    H = np.kron(tau, n_offdiag_tot)
    H = H + np.conj(H.T) - mu * np.kron(sigma[3], n_diag_tot)

    return 0.5 * H


def ManyBodySpectrum(E):
    spectrum = []
    comb_list = []
    combinations = itertools.product([-1, 1], repeat=len(E))
    for i in combinations:
        # print(i)
        spectrum.append(np.dot(E, np.array(i)))

    energy = np.array(spectrum)
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    return energy


@dataclass
class model:
    """ spinless 1D fermions """
    t: float  # nearest neighbor hopping t(c^dagger_i c_{i+1} + h.c)
    t2: float
    V: float  # nearest neighbor density-density interaction Vc^dagger_ic_i
    mu: float  # chemical potential -mu (c^\dagger_i c_i -1/2)
    Delta: float  # Pairing potential Delta c_i c_{i+1}
    Delta2: float
    lamb: float
    L: int  # Number of sites

    def __post_init__(self):
        self.calc_basis()

    def get_ar(self, parity='even'):

        if parity == 'even':
            r_to_a = self.rp2a
            a_to_r = self.a2rp
        elif parity == 'odd':
            r_to_a = self.rm2a
            a_to_r = self.a2rm
        else:
            raise ValueError('parity must be "even" or "odd"')

        return r_to_a, a_to_r

    def calc_basis(self):
        """
        Separates the full Hilbert space into even and odd sectors.
        The "a" integer runs over all integers, while the rp = 0,..., 2^{L-1}-1
        indices the basis states of the even sector, and rm similarily the odd sector
        The mapping between the two set if integers is given by a_to_rp and rp_to_a
        and similar for a_to_rm and rm_to_a.

        Returns
        -------
        None.

        """
        L = self.L
        a_to_rp = {}  # Map from the binary basis to the parity basis
        a_to_rm = {}  # Map from the binary basis to the parity basis
        rp_to_a = {}  # Map from parity basis to the binary basis
        rm_to_a = {}  # Map from parity basis to the binary basis
        rp = 0  # Dimension counting for the parity sector
        rm = 0  # Dimension counting for the parity sector

        for a in range(2 ** L):

            n_a = bin_to_n(a, L)  # Occupation number representation of state a

            if np.sum(n_a) % 2 == 0:  # Even parity sector
                a_to_rp[a] = rp
                rp_to_a[rp] = a
                rp += 1
            else:  # Odd parity sector
                a_to_rm[a] = rm
                rm_to_a[rm] = a
                rm += 1

        self.a2rp = a_to_rp  # Update maps between the different bases
        self.rp2a = rp_to_a  # Update maps between the different bases
        self.a2rm = a_to_rm  # Update maps between the different bases
        self.rm2a = rm_to_a  # Update maps between the different bases

    def calc_Hamiltonian(self, parity='even', bc='periodic'):

        """
        Calculates the Hamiltonian in the even or odd parity sector.

        Parameters
        ----------
        parity : string, optional
            Chooses the parity sector that is to be calculated.
            It can be either 'even' or 'odd'
            The default is 'even'.
        bc : string, optional
            Sets the boundary condition.
            Can be either 'periodic' or 'open'
            The default is 'periodic'.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        H : np.array
            The Hamiltonian in the given sector with given boundary conditions.

        """
        L = self.L
        dim = 2 ** (L - 1)
        H = np.zeros((dim, dim))

        # Parity sector and boundary condition
        if parity == 'even':
            r_to_a = self.rp2a
            a_to_r = self.a2rp
            self.Hbasis = np.zeros((len(self.a2rp), self.L + 1))
        elif parity == 'odd':
            r_to_a = self.rm2a
            a_to_r = self.a2rm
            self.Hbasis = np.zeros((len(self.a2rm), self.L + 1))
        else:
            raise ValueError('parity must be "even" or "odd"')

        if bc == 'periodic':
            Lhop = L
            Lhop2 = L
        elif bc == 'open':
            Lhop = L - 1
            Lhop2 = L - 2
        else:
            raise ValueError('boundary condition must be "periodic" or "open"')

        # Disorder realisation
        disorder_pot = self.lamb * np.random.uniform(0, 1, size=L)
        disorder_hop1 = np.random.uniform(0, 1, size=L)
        disorder_pair1 = np.random.uniform(0, 1, size=L)

        # Hamiltonian
        for r in range(dim):
            a = r_to_a[r]
            n = bin_to_n(a, L)
            self.Hbasis[r, :-1] = bin_to_n(a, self.L)
            self.Hbasis[r, -1] = r

            # Diagonal terms (mu and V terms)
            H[r, r] += self.V * np.dot(n[:Lhop], np.roll(n, -1)[:Lhop])
            H[r, r] += -np.dot(n - 0.5, self.mu * np.ones(L))
            H[r, r] += -np.dot(n, disorder_pot)

            # Nearest-neighbour terms
            for i in range(Lhop):

                j = np.mod(i + 1, Lhop)  # j is either i+1 or 0

                # Hopping term
                try:
                    b, h = self.hopping(n, i, j)
                    s = a_to_r[b]
                    ht = -self.t * h  # * disorder_hop1[i]
                    H[s, r] += ht
                    H[r, s] += np.conjugate(ht)
                except TypeError:
                    pass

                # Pairing term
                try:
                    b, h = self.pairing(n, i, j)
                    s = a_to_r[b]
                    hp = -self.Delta * h  # * disorder_pair1[i]
                    H[s, r] += hp
                    H[r, s] += np.conjugate(hp)
                except TypeError:
                    pass

            # Next-to-nearest neighbour
            for i in range(Lhop2):

                j = np.mod(i + 2, L)  # j is either i+1 or 0

                # Hopping term
                try:
                    b, h = self.hopping(n, i, j)
                    s = a_to_r[b]
                    ht = -self.t2 * h * (2 * n[np.mod(i + 1, L)] - 1)
                    H[s, r] += ht
                    H[r, s] += np.conjugate(ht)
                except TypeError:
                    pass

                # Pairing term
                try:
                    b, h = self.pairing(n, i, j)
                    s = a_to_r[b]
                    hp = -self.Delta2 * h * (2 * n[np.mod(i + 1, L)] - 1)
                    H[s, r] += hp
                    H[r, s] += np.conjugate(hp)
                except TypeError:
                    pass

        return H

    def hopping(self, n, i, j):

        if i == j and n[i] == 1:
            a = n_to_bin(n)
            return a, 1
        elif n[i] == 0 and n[j] == 1:
            a = n_to_bin(n)
            exponent = np.sum(n[min(i, j) + 1:max(i, j)])
            return a + 2 ** i - 2 ** j, (-1) ** exponent
        else:
            return None

    def pairing(self, n, i, j):
        """
        Calculates the binary r, and phase alpha where
        c_i c_j |a> = \alpha |r>
        with a obtained from the state n = [n0,...,n_{L-1}]

        Parameters
        ----------
        n : list of integers
            the binary representation of the state [n0,n1,...,n_{L-1}].
        i : int
        j : int

        Returns
        -------
        r and alpha if the state |a> is not anihilated,
        otherwise return None.

        """
        if i != j and (n[i] == 1 and n[j] == 1):
            a = n_to_bin(n)
            exponent = np.sum(n[min(i, j):max(i, j)])
            return a - 2 ** i - 2 ** j, np.sign(i - j) * (-1) ** exponent
        else:
            return None

    def calc_opdm_operator(self, parity='even'):
        """
        Calculates the matrix rho_ij = <r|c_i^dagger c_j|s>
        and <r|c_i c_j|s> in subspace of even or odd parity.
        rho['even/odd'][i,j] = rho_ij
        with rho_ij a sparse matrix.

        Parameters
        ----------
        parity : string, optional
            The parity of the basis for rho. The default is 'even'.


        Returns
        -------
        None. But sets self.rho to rho.

        """

        L = self.L
        dim = 2 ** (L - 1)

        rho = {}
        rho['eh'] = {}
        rho['hh'] = {}

        # Initializing rho_eh operator as a lil_matrix
        for i, j in itertools.product(range(L), range(L)):
            rho['eh'][i, j] = sparse.lil_matrix((dim, dim))
            rho['hh'][i, j] = sparse.lil_matrix((dim, dim))

        # Parity sector
        if parity == 'even':
            r_to_a = self.rp2a
            a_to_r = self.a2rp
        elif parity == 'odd':
            r_to_a = self.rm2a
            a_to_r = self.a2rm
        else:
            raise ValueError('parity must be "even" or "odd"')

        # Calculation of the matrix rho in the parity subspace
        for r in range(dim):

            # State r in the different basis
            a = r_to_a[r]  # Binary
            n = bin_to_n(a, L)  # Number representation

            # Iterate over the possible ij combinations
            for i, j in itertools.product(range(L), range(L)):

                # Electron-hole block
                try:
                    b, alpha = self.hopping(n, i, j)  # c_i^\dagger c_j /r>
                    rho['eh'][i, j][a_to_r[b], r] = alpha  # <b/ c^\dagger c/ r>
                    # print("eh :" + str(i) + str(", ") + str(j) + " = " + str(alpha))
                except TypeError:
                    pass

                # Hole-hole block
                try:
                    b, alpha = self.pairing(n, i, j)  # c_i c_j /r>
                    rho['hh'][i, j][a_to_r[b], r] = alpha  # <b/ c^\dagger c/ r>
                    # print("hh :" + str(i) + str(", ") + str(j) + " = " + str(alpha))
                except TypeError:
                    pass

        # Convert sparse lil matrices to csr
        self.rho = {}
        self.rho['eh'] = {}
        self.rho['hh'] = {}
        for key in rho['eh']:
            self.rho['eh'][key] = rho['eh'][key].tocsr()
            self.rho['hh'][key] = rho['hh'][key].tocsr()

        return None

    def calc_opdm_from_psi(self, psi, parity='even'):
        """
        Calculates the one-particle-density matrix from a state psi.

        rho_opdm =  [[ <psi|c^\dagger_i c_j|psi> , <psi|c^\dagger_i c^\dagger_j|psi>],
                     [          <psi|c_i c_j|psi>,   <psi|c_i c^\dagger_j|psi> ]]

        Parameters
        ----------
        psi : numpy arrary
            The state for which we want the opdm.
        parity : TYPE, optional
            the parity of the subspace in which the state lives. The default is 'even'.

        Returns
        -------
        rho_opdm : numpy array 2L x 2L
            the opdm.
        """

        if not hasattr(self, 'rho'):
            self.calc_opdm_operator(parity)

        L = self.L
        rho_opdm = np.zeros((2 * L, 2 * L))

        for i, j in itertools.product(range(L), range(L)):
            rho_opdm[i, j] = np.dot(psi.conjugate(), (self.rho['eh'][i, j] * psi))
            rho_opdm[i + L, j] = np.dot(psi.conjugate(), (self.rho['hh'][i, j] * psi))
        rho_opdm[L: 2 * L, L: 2 * L] = np.eye(L) - rho_opdm[:L, :L]  # .T
        rho_opdm[:L, L:2 * L] = rho_opdm[L: 2 * L, :L].T.conjugate()
        # for i, j in itertools.product(range(L), range(L)):
        #     rho_opdm[i, j] = np.dot(psi.conjugate(), (self.rho['eh'][i, j] * psi))
        #     rho_opdm[i, j+L] = np.dot(psi.conjugate(), (self.rho['hh'][i, j] * psi))
        # rho_opdm[L: 2*L, L: 2*L] = np.eye(L) - rho_opdm[:L, :L].conj()
        # rho_opdm[L:2*L, :L] = -rho_opdm[:L, L:2*L].conj()

        return rho_opdm

    def show_basis(self):
        self.calc_basis()

        auxp = np.zeros((len(self.a2rp), self.L + 1))
        auxm = np.zeros((len(self.a2rm), self.L + 1))

        for j in range(len(self.a2rp)):
            auxp[j, :-1] = bin_to_n(self.rp2a[j], self.L)
            auxp[j, -1] = j
        for j in range(len(self.a2rm)):
            auxm[j, :-1] = bin_to_n(self.rm2a[j], self.L)
            auxm[j, -1] = j

        print("-----------------")
        print("Even parity basis")
        print("-----------------")
        print(auxp)
        print("-----------------")
        print("odd parity basis")
        print("-----------------")
        print(auxm)
        print("-----------------")
        print("Hamiltonian basis")
        print("-----------------")
        print(self.Hbasis)

    def show_rho(self, i, j, parity="even"):

        rho_eh = self.rho['eh'][i, j].todense()
        rho_hh = self.rho['hh'][i, j].todense()

        if parity == "even":
            for i in range(len(self.a2rp)):
                for j in range(len(self.a2rp)):

                    if rho_eh[i, j] != 0:
                        print("State " + str(i) + str(", ") + str(bin_to_n(self.rp2a[i], self.L))
                              + str(" is connected to state ") + str(j) + str(", ")
                              + str(bin_to_n(self.rp2a[j], self.L)) + str(" by eh"))

                    if rho_hh[i, j] != 0:
                        print("State " + str(i) + str(", ") + str(bin_to_n(self.rp2a[i], self.L))
                              + str(" is connected to state ") + str(j) + str(", ")
                              + str(bin_to_n(self.rp2a[j], self.L)) + str(" by hh"))


        else:
            for i in range(len(self.a2rm)):
                for j in range(len(self.a2rm)):

                    if rho_eh[i, j] != 0:
                        print("State " + str(i) + str(", ") + str(bin_to_n(self.rm2a[i], self.L))
                              + str(" is connected to state ") + str(j) + str(", ")
                              + str(bin_to_n(self.rm2a[j], self.L)) + str(" by eh"))

                    if rho_hh[i, j] != 0:
                        print("State " + str(i) + str(", ") + str(bin_to_n(self.rm2a[i], self.L))
                              + str(" is connected to state ") + str(j) + str(", ")
                              + str(bin_to_n(self.rm2a[j], self.L)) + str(" by hh"))
        return rho_eh, rho_hh

    def check_rho_prodStates(self, Vodd, parity="even", u1charge="conserved"):
        """
        Checks that the single particle density matrix becomes a projector for any
        product state
        """

        # Definition OPDM operator, projector and psi
        self.calc_opdm_operator(parity)
        P = np.zeros((2 * self.L,))
        P[self.L:] = 1
        psi = np.zeros((len(self.a2rp),)) if parity == "even" else np.zeros((len(self.a2rm),))

        for i in range(len(psi)):
            psi[i] = 1
            if u1charge == "conserved":
                pass  # Random state conserving U(1) charge
            else:
                psi = unitary_group.rvs(len(psi)) @ psi  # Random state not conserving U(1) charge
            rho = self.calc_opdm_from_psi(psi, parity)  # OPDM

            if not np.allclose(spectrum(rho)[0], P, rtol=1e-15):
                raise AssertionError("OPDM not a projector for product states!")
            psi = np.zeros((len(self.a2rp),)) if parity == "even" else np.zeros((len(self.a2rm),))

        print("check done for product states with U(1) charge " + str(u1charge))

    def check_rho_eigenstates(self, V):

        """
        Checks that the single particle density matrix becomes a projector for any
        eigenstate of the hamiltonian
        """

        sigma_x = np.array([[0, 1], [1, 0]])
        Id = np.eye(int(self.L))
        S = np.kron(sigma_x, Id)
        P = np.zeros((2 * self.L,))
        P[self.L:] = 1
        for i in range(len(V[:, 0])):
            psi2 = V[:, i]
            psi2 = psi2 / np.linalg.norm(psi2)
            rho = self.calc_opdm_from_psi(psi2, parity='odd')  # Single particle density matrix (BdG x position space)
            values = spectrum(rho)[0]

            print("--------")
            print("State :", i)
            print("Projector: ", np.allclose(P, values))
            print("PH Symmetry: ", np.allclose(- S @ rho @ S + np.eye(2 * int(self.L)), np.conj(rho)))
            print("Trace: ", np.allclose(np.trace(rho), self.L))
            print("Hermiticity: ", np.allclose(rho, np.conj(rho.T)))
            print("--------")

    def check_spectra(self):

        Heven = self.calc_Hamiltonian(parity='even', bc='periodic')
        Hodd = self.calc_Hamiltonian(parity='odd', bc='periodic')
        Eeven, Veven = np.linalg.eigh(Heven)
        Eodd, Vodd = np.linalg.eigh(Hodd)
        E = np.concatenate((Eeven, Eodd))
        idx = E.argsort()
        E = E[idx]

        H2 = Ham_bdg(self.L, self.t, self.Delta, self.mu)
        E2 = spectrum(H2)[0]
        Ep = E2[E2 >= 0]
        Esp = ManyBodySpectrum(Ep)

        plt.figure()
        plt.plot(range(len(E)), E, '.y', markersize=15)
        plt.plot(range(len(Esp)), Esp, '.b', markersize=5)
        # plt.plot(r, Energy[r], '.r', markersize=6)
        # plt.ylim(-0.1, 0.1)
        plt.xlabel("Eigenstate number")
        plt.ylabel("Energy")
        plt.title('$H$')
        plt.show()

    def expand_psi_to_full_space(self, psi, parity='even'):

        r_to_a, a_to_r = self.get_ar(parity)
        psi_full = np.zeros(2 ** L, dtype=psi.dtype)

        for r in range(2 ** (self.L - 1)):
            psi_full[r_to_a[r]] = psi[r]

        return psi_full

if __name__ == '__main__':
    L = 10;
    t = 1.0;
    mu = 0.0;
    Delta = 1.0
    V = 0.0
    m = model(t, V, mu, Delta, L)
    H = m.calc_Hamiltonian(parity="even")
    E, psi = np.linalg.eigh(H)
    rho = m.calc_opdm_from_psi(psi[:, L // 2], parity="even")
    print(np.linalg.eigvalsh(rho))
