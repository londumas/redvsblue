from __future__ import print_function

import os
import sys
import fitsio
import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d

from redvsblue.constants import Lyman_series

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

def print(*args, **kwds):
    __builtin__.print(*args,**kwds)
    sys.stdout.flush()
def weighted_var(values, weights):
    """
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """

    m = sp.average(values, weights=weights)
    var = sp.average((values-m)**2, weights=weights)

    return var
def get_dv(z, zref):

    c = 299792458./1000.
    dv = c*(z-zref)/(1.+zref)

    return dv
def get_dz(dv, zref):

    c = 299792458./1000.
    dz = (1.+zref)*dv/c

    return dz
def read_PCA(path,dim=False,smooth=None):
    """

    """

    h = fitsio.FITS(path)
    head = h['BASIS_VECTORS'].read_header()
    ll = sp.asarray(head['CRVAL1']+head['CDELT1']*sp.arange(head['NAXIS1']), dtype=sp.float64)
    if 'LOGLAM' in head and head['LOGLAM']!=0:
        ll = 10**ll
    fl = sp.asarray(h['BASIS_VECTORS'].read(),dtype=sp.float64)
    h.close()

    fl[0,:] /= fl[0,:].mean()
    if not dim:
        dim = 1
    else:
        dim = fl.shape[0]

    copyfl = fl.copy()
    if smooth!=0:
        for i in range(copyfl.shape[0]):
            copyfl[i,:] = sp.ndimage.filters.gaussian_filter(copyfl[i,:],sigma=smooth)

    qso_pca = [ interp1d(ll,copyfl[i,:],fill_value='extrapolate',kind='linear') for i in range(dim) ]

    return qso_pca
def read_flux_calibration(path):
    """

    """

    h = fitsio.FITS(path)
    ll = h[1]['LOGLAM'][:]
    st = h[1]['STACK'][:]
    wst = h[1]['WEIGHT'][:]
    w = (st==0.) | (wst<=10.) | (st<0.8) | (st>1.2)
    st[w] = 1.
    flux_calib = interp1d(10**ll,st,fill_value='extrapolate',kind='linear')
    h.close()

    return flux_calib
def read_ivar_calibration(path):

    h = fitsio.FITS(path)
    ll = h[2]['LOGLAM'][:]
    eta = h[2]['ETA'][:]
    ivar_calib = interp1d(10**ll,eta,fill_value='extrapolate',kind='linear')
    h.close()

    return ivar_calib
def read_mask_lines(path):

    usr_mask_obs = []
    with open(os.path.expandvars(path), 'r') as f:
        for l in f:
            if l[0]=='#': continue
            l = l.split()
            if l[3]=='OBS':
                usr_mask_obs += [ [float(l[1]),float(l[2])] ]
    usr_mask_obs = sp.asarray(usr_mask_obs)

    return usr_mask_obs
def transmission_Lyman(zObj,lObs):
    '''Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
        1 -> everything is transmitted (medium is transparent)
        0 -> nothing is transmitted (medium is opaque)
    Args:
        zObj (float): Redshift of object
        lObs (array of float): wavelength grid
    Returns:
        array of float: transmitted flux fraction
    '''

    lRF = lObs/(1.+zObj)
    T = sp.ones(lObs.size)

    for l in Lyman_series.keys():
        w = lRF<Lyman_series[l]['line']
        zpix = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w] *= sp.exp(-tauEff)

    return T
def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    '''
    https://github.com/sczesla/PyAstronomy
    in /src/pyasl/asl/unred
    '''

    x = 10000./wave # Convert to inverse microns
    curve = x*0.

    # Set some standard values:
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c2 = -0.824 + 4.717/R_V
    c1 = 2.030 - 3.007*c2

    if LMC2:
        x0    =  4.626
        gamma =  1.05
        c4   =  0.42
        c3    =  1.92
        c2    = 1.31
        c1    =  -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4   =  0.64
        c3    =  2.73
        c2    = 1.11
        c1    =  -1.28

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xcutuv = np.array([10000.0/2700.0])
    xspluv = 10000.0/np.array([2700.0,2600.0])

    iuv = sp.where(x >= xcutuv)[0]
    N_UV = iuv.size
    iopir = sp.where(x < xcutuv)[0]
    Nopir = iopir.size
    if N_UV>0:
        xuv = sp.concatenate((xspluv,x[iuv]))
    else:
        xuv = xspluv

    yuv = c1 + c2*xuv
    yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
    yuv = yuv + c4*(0.5392*(sp.maximum(xuv,5.9)-5.9)**2+0.05644*(sp.maximum(xuv,5.9)-5.9)**3)
    yuv = yuv + R_V
    yspluv = yuv[0:2]  # save spline points

    if N_UV>0:
        curve[iuv] = yuv[2::] # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR
    xsplopir = sp.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
    ysplir = np.array([0.0,0.26469,0.82925])*R_V/3.1
    ysplop = np.array((sp.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ),
            sp.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ),
            sp.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ),
            sp.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
    ysplopir = sp.concatenate((ysplir,ysplop))

    if Nopir>0:
      tck = interpolate.splrep(sp.concatenate((xsplopir,xspluv)),sp.concatenate((ysplopir,yspluv)),s=0)
      curve[iopir] = interpolate.splev(x[iopir], tck)

    #Now apply extinction correction to input flux vector
    curve *= ebv
    corr = 1./(10.**(0.4*curve))

    return corr
