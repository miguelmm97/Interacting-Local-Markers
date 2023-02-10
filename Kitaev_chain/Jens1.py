#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:48:54 2022

@author: jensba
"""

import numpy as np
from dataclasses import dataclass


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
    >>> bin_to_n(10, 4)
    array([0, 1, 0, 1])

    Explanation
    ----------
    bin(s) takes the integer s to a binary string, whose first to letters are a "decoration" and we don't need them
    so we tak the [2:]. We fill with zeros at the start of the binary string to acount for all sites with zfill(L) and
    then we reverse the binary string, in order to read the physical sites from left to right.

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
    >>> n_to_bin([0, 1, 0, 1])
    10

    Explanation
    ----------
    We take the array of occupation sites n and reverse it. int(reversed(n), 2) tells python that n is in a binary base
    and gives back the correspoinding integer.


    """
    
    return int(''.join(str(x) for x in reversed(n)), 2)


@dataclass
class model:
    """ spinless 1D fermions """
    t: float     # nearest neighbor hopping t(c^dagger_i c_{i+1} + h.c)
    V: float     # nearest neighbor density-density interaction Vc^dagger_ic_i
    mu: float    # chemical potential -mu (c^\dagger_i c_i -1/2)
    Delta: float # Pairing potential Delta c_i c_{i+1}
    L: int       # Number of sites
   
    
    def calc_basis(self):
        """
        Separates the full Hilbert space into even and odd sectors.
        The a integer runs over all integers, while the rp = 0,...,2^{L-1}-1
        indices the basis states of the even sector, and rm similarily the odd sector
        The mapping between the two set if integers is given by a_to_rp and rp_to_a
        and similar for a_to_rm and rm_to_a.

        Returns
        -------
        None.

        Explanation
        --------
        We take the basis states 0, 2^L, and write them in occupation number representation with bin_to_a. If there is
        an even number of occupied sites then we are in the even subspace, if the number is odd we are in the odd subspace.
        We save these two subspace by means of two strings: a_to_rp/rm gives the binary representation of a state within
        the even/odd subspace, and rp/rm_to_a gives the binary representation in the full Hilbert space of a state a with
        binary representation in rp(rm in the even/odd subspace.

        """
        L = self.L
        a_to_rp = {}
        a_to_rm = {}
        rp_to_a = {}
        rm_to_a = {}
        rp = 0; rm = 0
        for a in range(2 ** L):
            n_a = bin_to_n(a,L)
            if np.sum(n_a) % 2 == 0:
                a_to_rp[a] = rp
                rp_to_a[rp] = a
                rp += 1
            else:
                a_to_rm[a] = rm
                rm_to_a[rm] = a
                rm += 1
                

        self.a2rp = a_to_rp
        self.rp2a = rp_to_a
        self.a2rm = a_to_rm
        self.rm2a = rm_to_a
        
        
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
        Errors if the boundary or parity are not well defined

        Returns
        -------
        H : np.array
            The Hamiltonian in the given sector with given boundary conditions.

        """
        L = self.L
        dim = 2 ** (L - 1)
        H = np.zeros((dim, dim))
        self.calc_basis()

        # Selecting states in the parity sector and boundary condition
        if parity == 'even':
            r_to_a = self.rp2a
            a_to_r = self.a2rp
        elif parity == 'odd':
            r_to_a = self.rm2a
            a_to_r = self.a2rm
        else:
            raise ValueError('parity must be "even" or "odd"')

        if bc == 'periodic':
            Lhop = L
        elif bc == 'open':
            Lhop = L-1
        else:
                raise ValueError('boundary condition must be "periodic" or "open"')
    
        # Construction of the hamiltonian
        for r in range(dim):
            a = r_to_a[r]
            n = bin_to_n(a, L)

            # Diagonal terms (mu and V terms)
            H[r,r] += np.dot(n-0.5 ,self.mu * np.ones(L)) + self.V * np.dot(n[:Lhop], np.roll(n, -1)[:Lhop])

            # Off-diagonal terms, need to look at them separately for each site i
            for i in range(Lhop):
                j = np.mod(i+1, L)  # j is either i+1 or 0

                # normal hopping term
                try:
                    b, h = self.hopping(n, i, j)  # State connected by hopping in the binary representation
                    s = a_to_r[b]                 # Binary rep of b in the parity sector
                    ht = -self.t*h                # Hopping times the exponent of the hopping term
                    H[s, r] += ht
                    H[r, s] += np.conjugate(ht)
                except TypeError:
                    pass

                # Pairing term
                try:
                    b, h = self.pairing(n, i, j)
                    s = a_to_r[b]
                    hp = self.Delta*h
                    H[s, r] += hp
                    H[r, s] += np.conjugate(hp)
                except TypeError:
                    pass

        return H
    

    def hopping(self, n, i, j):
        """

        Parameters
        ----------
        n: Occupation number representation of an initial state
        i: Site to which we hopp
        j: Site from which we hopp

        Returns
        -------
        [0]: State connected to n by the hopping term (in the full binary representation)
        [1]: Exponent that gives us the sign depending on the commutations that we had to do

        Explanation
        -------
        If the hopping is onsite, we give back the same state. If not, we need site i to be empty and site j to be occupied.
        The state we get back is, in the full binary description, just setting i to 1 and j to 0 (a + 2**i - 2**j),
        times the exponent (which one can see by calculating the hopping term analytically)
        """
        a = n_to_bin(n)
        if i == j and n[i] == 1:
            return a, 1
        elif n[i] == 0 and n[j] == 1:
            exponent = np.sum(n[min(i, j)+1:max(i, j)])
            return a + 2 ** i - 2 ** j, (-1)**exponent
        else:
            return None

        
    def pairing(self, n, i, j):
        """

        Parameters
        ----------
        n: Occupation number representation of an initial state
        i: Site we pair with site j
        j: Site we want to pair

        Returns
        -------
        [0]: State connected to n by the hopping term (in the full binary representation)
        [1]: Exponent that gives us the sign depending on the commutations that we had to do

        Explanation
        -------
        Same flavour as with the hopping term, just different analytical implementation.


        """
        a = n_to_bin(n)
        if n[i] == 1 and n[j] == 1:
            if j == i + 1:
                exponent = -1
            elif j == 0:
                exponent = np.sum(n[1:self.L - 2])
            else:
                raise ValueError('pairing term only between nearest neighbors')
            return a - 2 ** i - 2 ** j, (-1)**exponent
        else:
            return None



