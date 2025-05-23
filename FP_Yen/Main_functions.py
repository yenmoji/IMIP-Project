#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
AlterMin Implements alternating minimization sequentially on a stack of
measurement I (n1 x n2 x nz). It consists of 2 loops. The main loop updates
the reconstruction results O and P. The inner loop applies projectors/minimizers
P1 and P2 on each image I and steps through the entire dataset.

Outputs:
O: reconstructed high-resolution complex object
P: reconstructed complex pupil function
err: errors at each iteration
scale: LED brightness map
Ns: estimated LED positions using local search algorithms

Inputs:
Measurements data
I: intensity measurements by different LEDs
Reconstruction parameters
No = [Ny_obj,Nx_obj]: size of the reconstructed image
Illumination coding parameters
Ns = [Nsy,Nsx]: centers of corresponding lpf regions for
the illumination pattern

Iteration parameters: opts
- tol: maximum change of error allowed in two consecutive iterations
- maxIter: maximum iterations
- minIter: minimum iterations
- monotone (1, default): if monotone, error has to monotonically drop when iters > minIter
- display: display results (0: no (default) 1: yes)
- saveIterResult: save results at each step as images (0: no (default) 1: yes)
- mode: display in 'real' space or 'fourier' space
- out_dir: saving directory
- O0, P0: initial guesses for O and P
- OP_alpha: regularization parameter for O
- OP_beta: regularization parameter for P
- scale: LED brightness map
- H0: known portion of the aberration function
- poscalibrate: flag for LED position correction ('0', 'sa', 'ga')
- calbratetol: tolerance in optimization-based position correction
- F, Ft: operators of Fourier transform and inverse

"""

import numpy as np
import os
import time
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import dual_annealing
from matplotlib import pyplot as plt


# In[2]:


def GDUpdate_Multiplication_rank1(O, P, dpsi, Omax, cen, Ps, alpha, beta, StepSize):
    """
    Update estimate of object O and pupil P using gradient descent where psi = O * P.

    Parameters:
    - O: complex object estimate (2D numpy array)
    - P: complex pupil function estimate (2D numpy array)
    - dpsi: update difference in field estimate (2D numpy array)
    - Omax: maximum amplitude of object used for normalization (scalar)
    - cen: center position of the pupil on the object field (2-element list or array)
    - Ps: support mask for the pupil function (same shape as P)
    - alpha: regularization parameter for object update (scalar)
    - beta: regularization parameter for pupil update (scalar)
    - StepSize: gradient descent step size for object update (scalar)

    Returns:
    - O: updated object estimate
    - P: updated pupil estimate
    """
    if dpsi.ndim == 3 and dpsi.shape[2] == 1:
        dpsi = dpsi[..., 0]
    
    Np = np.array(P.shape)
    cen = np.array(cen).flatten()  # ensure cen is 1D array of scalars

    # Calculate top-left corner (n1) as integer tuple
    val = cen - np.floor(Np / 2)
    n1 = tuple(int(x) for x in val)

    # Calculate bottom-right corner (n2) as integer tuple
    val2 = np.array(n1) + Np
    n2 = tuple(int(x) for x in val2)

    # Crop region from O corresponding to pupil position
    O1 = O[n1[0]:n2[0], n1[1]:n2[1]]

    # Update object O in-place
    O[n1[0]:n2[0], n1[1]:n2[1]] += (
        StepSize * (1 / np.max(np.abs(P))) * np.abs(P) * np.conj(P) * dpsi / (np.abs(P)**2 + alpha)
    )

    # Update pupil P in-place
    P += (
        (1 / Omax) * (np.abs(O1) * np.conj(O1)) * dpsi / (np.abs(O1)**2 + beta) * Ps
    )

    return O, P


# In[3]:





# In[4]:


def Proj_Fourier_v2(psi0, I, I0, c, F):
    """
    Projection based on intensity measurement in the Fourier domain.
    Replaces the amplitude of the Fourier transform by measured amplitude sqrt(I).

    Parameters:
    - psi0: complex input field(s), shape (n1, n2) or (n1, n2, r)
    - I: measured intensity image (2D numpy array)
    - I0: estimated intensity (2D numpy array)
    - c: scaling factor (scalar or 1D array with length r)
    - F: Fourier transform operator (function handle)

    Returns:
    - psi: updated complex field(s) in Fourier domain, same shape as psi0
    """

    psi0 = np.asarray(psi0)
    I = np.asarray(I)
    I0 = np.asarray(I0)
    eps = np.finfo(float).eps  # small constant to avoid division by zero

    if psi0.ndim == 2:
        # Single LED case (r == 1)
        psi = F(np.sqrt(I / c) * np.exp(1j * np.angle(psi0)))
    else:
        # Multiple LEDs (r > 1)
        n1, n2, r = psi0.shape
        psi = np.zeros_like(psi0, dtype=np.complex128)
        for m in range(r):
            psi[:, :, m] = F(np.sqrt(I / c[m]) * psi0[:, :, m] / np.sqrt(I0 + eps))

    return psi


# In[6]:


# Main Alternating Minimization Function
def AlterMin(I, No, Ns, opts):
    # Derived constants
    Nmy, Nmx, Nimg = I.shape
    Np = (Nmy, Nmx)
    r0 = Ns.shape[0]
    cen0 = ((No[0]+1)//2, (No[1]+1)//2)
    row = lambda x: x.reshape(-1)

    # Default options setup
    opts.setdefault('tol', 1)
    opts.setdefault('maxIter', 50)
    opts.setdefault('minIter', 3)
    opts.setdefault('monotone', 1)
    opts.setdefault('display', 0)
    opts.setdefault('saveIterResult', 0)
    opts.setdefault('out_dir', 'IterResults')
    opts.setdefault('OP_alpha', 3)
    opts.setdefault('OP_beta', 3)
    opts.setdefault('mode', 'real')
    opts.setdefault('Ps', 1)
    opts.setdefault('iters', 10)
    opts.setdefault('scale', np.ones((r0, Nimg)))
    opts.setdefault('H0', np.ones(Np))
    opts.setdefault('poscalibrate', 0)
    opts.setdefault('calbratetol', 1e-1)
    opts.setdefault('StepSize', 0.01)
    opts.setdefault('F', lambda x: fftshift(fft2(x)))
    opts.setdefault('Ft', lambda x: ifft2(ifftshift(x)))

    F = opts['F']
    Ft = opts['Ft']

    if 'O0' not in opts:
        opts['O0'] = np.pad(Ft(np.sqrt(I[:, :, 0])) / r0,
                            [(No[0] - Np[0]) // 2, (No[1] - Np[1]) // 2])
    if 'P0' not in opts:
        opts['P0'] = np.ones(Np, dtype=np.complex128)

    H0 = opts['H0']

    downsamp = lambda x, cen: x[
        cen[0] - Np[0]//2 : cen[0] - Np[0]//2 + Np[0],
        cen[1] - Np[1]//2 : cen[1] - Np[1]//2 + Np[1]
    ]

    start_time = time.time()
    print("| iter |  rmse    |\n" + "-" * 20)

    # Initialization
    P = opts['P0']
    O = opts['O0']
    err1 = float('inf')
    err2 = 50
    err = []
    iter = 0
    scale = opts['scale']

    sp0 = np.max(np.abs(Ns[:, 0, :] - Ns[:, 1, :]))

    print(f"| {iter:2d}   | {err1:.2e} |")

    while abs(err1 - err2) > opts['tol'] and iter < opts['maxIter']:
        err1 = err2
        err2 = 0
        iter += 1

        for m in range(Nimg):
            Psi0 = np.zeros((Np[0], Np[1], r0), dtype=complex)
            Psi_scale = np.zeros_like(Psi0)
            cen = np.zeros((2, r0), dtype=int)
            scale0 = np.zeros(r0)

            for p in range(r0):
                cen[:, p] = np.array(cen0) - row(Ns[p, m, :]).astype(int)
                scale0[p] = scale[p, m]
                Psi0[:, :, p] = downsamp(O, cen[:, p]) * P * H0
                Psi_scale[:, :, p] = np.sqrt(scale0[p]) * Psi0[:, :, p]

            I_mea = I[:, :, m]
            psi0 = Ft(Psi_scale)
            I_est = np.sum(np.abs(psi0)**2, axis=2)
            Psi = Proj_Fourier_v2(psi0, I_mea, I_est, scale0, F)
            dPsi = Psi - Psi0

            Omax = np.abs(O[cen0[0], cen0[1]])
            if r0 == 1:
                P2 = GDUpdate_Multiplication_rank1

            O, P = P2(O, P, dPsi / H0[..., np.newaxis], Omax, cen, opts['Ps'], opts['OP_alpha'], opts['OP_beta'], opts.get('StepSize', 1))

            # Position correction
            if opts['poscalibrate'] == 'sa':
                def poscost(ss):
                    ss = np.round(ss).astype(int)
                    return np.sum((np.abs(Ft(downsamp(O, ss) * P * H0))**2 - I_mea)**2)

                bounds = [(cen[:, 0][0] - sp0 // 3, cen[:, 0][0] + sp0 // 3),
                          (cen[:, 0][1] - sp0 // 3, cen[:, 0][1] + sp0 // 3)]
                result = dual_annealing(poscost, bounds)
                cen_correct = np.round(result.x).astype(int)
                Ns[:, m, :] = np.array(cen0) - cen_correct.reshape(2, 1)

            err2 += np.sqrt(np.sum((I_mea - I_est)**2))

        err.append(err2)
        print(f"| {iter:2d}   | {err2:.2e} |")

        if opts['monotone'] and iter > opts['minIter']:
            if err2 > err1:
                break

    if opts['mode'] == 'fourier':
        O = Ft(O)

    print(f"elapsed time: {time.time() - start_time:.0f} seconds")

    return O, P, err, scale, Ns


# In[ ]:




