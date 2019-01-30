import os
import sys
import fitsio
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import redvsblue.constants

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
def read_PCA(path,dim=False):
    """

    """

    h = fitsio.FITS(path)
    head = h['BASIS_VECTORS'].read_header()
    ll = sp.asarray(head['CRVAL1']+head['CDELT1']*sp.arange(head['NAXIS1']), dtype=sp.float64)
    if 'LOGLAM' in head and head['LOGLAM']!=0:
        ll = 10**ll
    fl = sp.asarray(h['BASIS_VECTORS'].read(),dtype=sp.float64)
    fl[0,:] /= fl[0,:].mean()
    if not dim:
        dim = 1
    else:
        dim = fl.shape[0]
    qso_pca = [ interp1d(ll,fl[i,:],fill_value='extrapolate',kind='linear') for i in range(dim) ]
    h.close()

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
def plot_flux_calibration(path):
    """

    """

    h = fitsio.FITS(path)
    ll_st = h[1]['LOGLAM'][:]
    st = h[1]['STACK'][:]
    w = st!=0.
    h.close()

    plt.plot(10**ll_st[w],st[w])
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}}\,[\mathrm{\AA}]$')
    plt.ylabel(r'$C(\lambda_{\mathrm{Obs.}})$')
    plt.grid()
    plt.show()

    return
def plot_ivar_calibration(path):

    h = fitsio.FITS(path)
    ll = h[2]['LOGLAM'][:]
    eta = h[2]['ETA'][:]
    h.close()

    plt.plot(10**ll,eta)
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}}\,[\mathrm{\AA}]$')
    plt.ylabel(r'$\eta(\lambda_{\mathrm{Obs.}})$')
    plt.grid()
    plt.show()

    return
def plot_PCA(path,dwave_side=85):

    h = fitsio.FITS(path)
    head = h['BASIS_VECTORS'].read_header()
    ll = sp.asarray(head['CRVAL1']+head['CDELT1']*sp.arange(head['NAXIS1']), dtype=sp.float64)
    if 'LOGLAM' in head and head['LOGLAM']!=0:
        ll = 10**ll
    fl = sp.asarray(h['BASIS_VECTORS'].read(),dtype=sp.float64)
    fl[0,:] /= fl[0,:].mean()
    h.close()
    qso_pca = [ interp1d(ll,fl[i,:],fill_value='extrapolate',kind='linear') for i in range(fl.shape[0]) ]

    lines = redvsblue.constants.emissionLines

    i = 0
    for ln, lv in lines.items():
        if ln=='PCA': continue
        plt.plot(ll,fl[i,:],linewidth=2,color='black')
        x = sp.linspace(lv-85,lv+85,2*85*10)
        y = qso_pca[0](x)
        plt.fill_between(x,y,alpha=0.8)
        plt.plot([lv,lv],[0.,qso_pca[0](lv)],linewidth=2,alpha=0.8,color='red')
        plt.xlabel(r'$\lambda_{\mathrm{Obs.}}\,[\mathrm{\AA}]$',fontsize=20)
        plt.ylabel(r'$f_{\mathrm{PCA,'+str(i)+'}}$',fontsize=20)
        plt.grid()
        plt.show()

    return
def read_mask_lines(path):

    usr_mask_obs = []
    with open(os.path.expandvars(path), 'r') as f:
        for l in f:
            if l[0]=='#': continue
            l = l.split()
            if l[3]=='OBS':
                usr_mask_obs += [ [float(l[1]),float(l[2])] ]
        f.closed
    usr_mask_obs = sp.asarray(usr_mask_obs)

    return usr_mask_obs
