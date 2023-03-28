#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:35:59 2022

@author: jensba
"""

import numpy as np


def reshape_psi(psi, n, l):
    '''


    Parameters
    ----------
    psi : numpy array
        Vector of size 2**L where L is the number of sites.
    n : INT
        the leftmost site of the subset of interest A.
    l : INT
        The number of sites in the subset of interest A.

    Returns
    -------
    psi: numpy array (2**l,2**(L-l))
        Reshaped psi in a matrix psi_{(A),(\bar{A}) with (A) the combined
        indices in A, and similar for \bar{A}

    '''

    L = int(np.log2(psi.size))
    psi = np.transpose(
        np.reshape(psi, L * [2]))  # transpose since the binary representation is in opposite order in the basis
    psi = np.moveaxis(psi, list(range(n, n + l)), list(range(0, l)))
    psi = np.reshape(psi, (2 ** l, 2 ** (L - l)))

    return psi


def S_vN(psi):
    '''


    Parameters
    ----------
    psi : np.array 2**Na x 2**Nb
        The wave function. Assumed to have been reshpaed into a matrix with
        the correcet subspace dimensions

    Returns
    -------
    S : FLOAT
        the von Neumann entanglement entropy

    '''

    sv = np.linalg.svd(psi, compute_uv=False)
    sv = sv[sv > 1e-16]
    sv = sv ** 2
    return -np.sum(sv * np.log2(sv))


def calc_entropies(psi):
    '''


    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    SvN : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l

    '''

    L = int(np.log2(psi.size))
    SvN = {}

    for l in range(1, L + 1):
        SvN[l] = []
        for n in range(L - l + 1):
            psi_r = reshape_psi(psi, n, l)
            SvN[l].append(S_vN(psi_r))
        SvN[l] = np.array(SvN[l])
    return SvN


def calc_info(psi):
    '''


    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''
    L = int(np.log2(psi.size))
    SvN = calc_entropies(psi)
    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - SvN[l]
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt
