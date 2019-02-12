import fitsio
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import redvsblue.constants

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
        y = qso_pca[i](x)
        plt.fill_between(x,y,alpha=0.8)
        plt.plot([lv,lv],[0.,qso_pca[i](lv)],linewidth=2,alpha=0.8,color='red')
        plt.xlabel(r'$\lambda_{\mathrm{R.F.}}\,[\mathrm{\AA}]$',fontsize=20)
        plt.ylabel(r'$f_{\mathrm{PCA,'+str(i)+'}}$',fontsize=20)
        plt.grid()
        plt.show()

    return
