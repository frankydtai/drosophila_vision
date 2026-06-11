# -*- coding: utf-8 -*-
"""
Local replacement for the Borst-lab blindschleiche_py3 utility.

Implementations ported from FigureCode/Borst_Fig1.py and FigureCode/Borst_Fig4-6.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def setmyaxes(myxpos, myypos, myxsize, myysize):
    ax = plt.axes([myxpos, myypos, myxsize, myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return ax


def Gauss1D(FWHM, RFsize):
    myrange = RFsize / 2
    sigma = FWHM / (2.0 * np.sqrt(2 * np.log(2)))
    x = np.arange(-myrange, (myrange + 1), 1) * 1.0
    z = np.exp(-x ** 2 / (2 * (sigma ** 2)))
    z = z / np.sum(z)
    return z


def Gauss2D(FWHM, RFsize):
    myrange = RFsize / 2
    sigma = FWHM / (2.0 * np.sqrt(2 * np.log(2)))
    x = np.arange(-myrange, (myrange + 1), 1)
    y = np.arange(-myrange, (myrange + 1), 1)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.exp(-r ** 2 / (2 * (sigma ** 2)))
    z = z / np.sum(z)
    return z


def lowpass(x, tau):
    x = x.transpose(np.roll(np.arange(x.ndim), 1))
    n = x.shape[0]
    result = np.zeros_like(x)

    if tau < 1:
        result = x
    if tau >= 1:
        result[0] = x[0]
        for i in range(0, n - 1):
            result[i + 1] = 1.0 / tau * (x[i] - result[i]) + result[i]

    result = result.transpose(np.roll(np.arange(result.ndim), -1))
    return result


def highpass(x, tau):
    return x - lowpass(x, tau)


def bandpass(signal, hp_tau, lp_tau):
    result = lowpass(signal, lp_tau)
    if hp_tau != 0:
        result = highpass(result, hp_tau)
    return result


def rebin(x, f0, f1=0):
    mydim = x.ndim
    n = x.shape

    if mydim == 1:
        result = np.zeros((f0))
        if f0 <= n[0]:
            result = x[0:n[0]:int(n[0] / f0)]
        if f0 > n[0]:
            result = np.repeat(x, int(f0 / n[0]))

    if mydim == 2:
        result = np.zeros((f0, f1))
        interim = np.zeros((f0, n[1]))

        if f0 <= n[0]:
            interim = x[0:n[0]:int(n[0] / f0), :]
        if f0 > n[0]:
            interim = np.repeat(x, int(f0 / n[0]), axis=0)

        if f1 <= n[1]:
            result = interim[:, 0:n[1]:int(n[1] / f1)]
        if f1 > n[1]:
            result = np.repeat(interim, int(f1 / n[1]), axis=1)

    return result.copy()


def blurr(inp_image, FWHM):
    if inp_image.ndim == 1:
        z = Gauss1D(FWHM, 4 * FWHM)
    if inp_image.ndim == 2:
        z = Gauss2D(FWHM, 4 * FWHM)

    return scipy.ndimage.convolve(inp_image, z)
