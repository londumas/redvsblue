import os
import scipy as sp
import fitsio
from functools import partial

from redvsblue import utils

def get_mask_lines(pathLines):

    usr_mask_obs = []
    with open(os.path.expandvars(pathLines), 'r') as f:
        for l in f:
            if l[0]=='#': continue
            l = l.split()
            if l[3]=='OBS':
                usr_mask_obs += [ [float(l[1]),float(l[2])] ]
        f.closed
    usr_mask_obs = sp.asarray(usr_mask_obs)

    return usr_mask_obs
def cat_DR12Q(pathData,zmin,zmax,zkey='Z_VI'):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)
    for k in ['PLATE','MJD','FIBERID','THING_ID']:
        dic[k] = h['DR12Q_v2_10.fits'][k][:]
    dic['Z'] = h['DR12Q_v2_10.fits'][zkey][:]
    h.close()

    w = (dic['Z']!=-1.) & (dic['Z']>zmin) & (dic['Z']<zmax)
    for k in dic.keys():
        dic[k] = dic[k][w]

    return dic
def read_spec_spplate(p,m,f,path_spec=None, lambda_min=3600., lambda_max=7235., veto_lines=None, flux_calib=None, ivar_calib=None):
    """


    """
    path = path_spec+'{}/spPlate-{}-{}.fits'.format(p,p,m)

    h = fitsio.FITS(path)
    fl = h[0].read()
    iv = h['IVAR'].read()*(h['ANDMASK'].read()==0)
    head = h[0].read_header()
    h.close()

    ll = head['CRVAL1'] + head['CD1_1']*sp.arange(head['NAXIS1'])
    if head['DC-FLAG']:
        ll = 10**ll

    w = (iv[f-1,:]>0.) & (ll>lambda_min) & (ll<lambda_max)
    if not veto_lines is None:
        for lmin,lmax in veto_lines:
            w &= (ll<lmin) | (ll>lmax)

    ll = ll[w]
    fl = fl[f-1,:][w]
    iv = iv[f-1,:][w]

    if not flux_calib is None:
        correction = flux_calib(sp.log10(ll))
        fl /= correction
        iv *= correction**2
    if not ivar_calib is None:
        correction = ivar_calib(sp.log10(ll))
        iv /= correction

    return ll, fl, iv
def read_SDSS_data(path_DR12Q, path_spec, lines, zmin=0., zmax=10., zkey='Z_VI', lambda_min=3600., lambda_max=7235.,
    veto_lines=None, flux_calib=None, ivar_calib=None, nspec=None):
    """

    """

    ### Read quasar catalog
    catQSO = cat_DR12Q(path_DR12Q,zmin,zmax,zkey)
    print('Found {} quasars'.format(catQSO['Z'].size))

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib)

    if nspec is None:
        nspec = catQSO['Z'].size

    data = {}
    for i in range(nspec):
        t = catQSO['THING_ID'][i]
        p = catQSO['PLATE'][i]
        m = catQSO['MJD'][i]
        f = catQSO['FIBERID'][i]
        z = catQSO['Z'][i]

        lam, fl, iv = p_read_spec_spplate(p,m,f)
        lamRF = lam/(1.+z)

        data[t] = { 'Z':z }

        for ln, lv in lines.items():
            valline = {}
            for side in ['BLUE','RED']:
                w = (lamRF>lv[side+'_MIN']) & (lamRF<lv[side+'_MAX'])
                if w.sum()>10:
                    valline[side+'_VAR'] = utils.weighted_var(fl[w],iv[w])
                    valline[side+'_SNR'] = ( (fl[w]*iv[w])**2 ).mean()
                else:
                    valline[side+'_VAR'] = 0.
                    valline[side+'_SNR'] = 0.
            data[t][ln] = valline

    return data
