import os
import fitsio
import scipy as sp
from scipy.interpolate import interp1d

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
    ll_st = h[1]['LOGLAM'][:]
    st = h[1]['STACK'][:]
    w = st!=0.
    flux_calib = interp1d(ll_st[w],st[w],fill_value='extrapolate',kind='linear')
    h.close()

    return flux_calib
def read_ivar_calibration(path):

    h = fitsio.FITS(path)
    ll = h[2]['LOGLAM'][:]
    eta = h[2]['ETA'][:]
    ivar_calib = interp1d(ll,eta,fill_value='extrapolate',kind='linear')
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
        f.closed
    usr_mask_obs = sp.asarray(usr_mask_obs)

    return usr_mask_obs
