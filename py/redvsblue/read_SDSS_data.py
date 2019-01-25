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
def read_cat(pathData,zmin,zmax,zkey='Z_VI'):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)
    for k in ['PLATE','MJD','FIBERID','THING_ID','RA','DEC']:
        dic[k] = h[1][k][:]
    dic['Z'] = h[1][zkey][:]
    h.close()

    print(" start               : nb object in cat = {}".format(dic['Z'].size) )
    w = dic['THING_ID']>0
    print(" and thid>0          : nb object in cat = {}".format(w.sum()) )
    w &= dic['RA']!=dic['DEC']
    print(" and ra!=dec         : nb object in cat = {}".format(w.sum()) )
    w &= dic['RA']!=0.
    print(" and ra!=0.          : nb object in cat = {}".format(w.sum()) )
    w &= dic['DEC']!=0.
    print(" and dec!=0.         : nb object in cat = {}".format(w.sum()) )
    w &= dic['Z']>0.
    print(" and z>0.            : nb object in cat = {}".format(w.sum()) )

    w &= (dic['Z']>zmin) & (dic['Z']<zmax)
    print(" and z in range      : nb object in cat = {}".format(w.sum()) )
    for k in dic.keys():
        dic[k] = dic[k][w]

    return dic
def read_spec_spplate(p,m,path_spec=None, lambda_min=3600., lambda_max=7235., veto_lines=None, flux_calib=None, ivar_calib=None):
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

    w = (ll>lambda_min) & (ll<lambda_max)
    if not veto_lines is None:
        for lmin,lmax in veto_lines:
            w &= (ll<lmin) | (ll>lmax)

    ll = ll[w]
    fl = fl[:,w]
    iv = iv[:,w]

    if not flux_calib is None:
        correction = flux_calib(sp.log10(ll))
        fl /= correction[None,:]
        iv *= correction[None,:]**2
    if not ivar_calib is None:
        correction = ivar_calib(sp.log10(ll))
        iv /= correction[None,:]

    return ll, fl, iv
def read_SDSS_data(DRQ, path_spec, lines, zmin=0., zmax=10., zkey='Z_VI', lambda_min=3600., lambda_max=7235.,
    veto_lines=None, flux_calib=None, ivar_calib=None, nspec=None):
    """

    """

    ### Read quasar catalog
    catQSO = read_cat(DRQ,zmin,zmax,zkey)
    print('Found {} quasars'.format(catQSO['Z'].size))

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib)

    ### Sort PLATE-MJD
    pm = catQSO['PLATE']*100000 + catQSO['MJD']
    upm = sp.sort(sp.unique(pm))
    npm = sp.bincount(pm)
    w = npm>0
    npm = npm[w]
    w = sp.argsort(npm)
    upm = upm[w][::-1]
    npm = npm[w][::-1]

    data = {}
    for tpm in upm:
        p = tpm//100000
        m = tpm%100000
        w = pm==tpm

        try:
            lam, fl, iv = p_read_spec_spplate(p,m)
        except OSError:
            print('WARNING: Can not find PLATE={}, MJD={}'.format(p,m))
            continue
        print('{}: read {} objects from PLATE={}, MJD={}'.format(len(data.keys()),w.sum(),p,m))

        thids = catQSO['THING_ID'][w]
        fibs = catQSO['FIBERID'][w]
        zs = catQSO['Z'][w]

        for i in range(w.sum()):

            t = thids[i]
            f = fibs[i]
            z = zs[i]

            tfl = fl[f-1]
            tiv = iv[f-1]
            lamRF = lam/(1.+z)

            data[t] = { 'Z':z }

            for ln, lv in lines.items():
                valline = {}
                for side in ['BLUE','RED']:
                    w = (tiv>0.) & (lamRF>lv[side+'_MIN']) & (lamRF<lv[side+'_MAX'])
                    if w.sum()>10:
                        valline[side+'_VAR'] = utils.weighted_var(tfl[w],tiv[w])
                        valline[side+'_SNR'] = ( (tfl[w]*tiv[w])**2 ).mean()
                    else:
                        valline[side+'_VAR'] = 0.
                        valline[side+'_SNR'] = 0.
                data[t][ln] = valline

        if (not nspec is None) and (len(data.keys())>nspec):
            print('{}:'.format(len(data.keys())))
            return data

    return data
