from __future__ import print_function
import scipy as sp
import fitsio
import iminuit
from functools import partial

from redvsblue import utils
from redvsblue.utils import print
from redvsblue.zwarning import ZWarningMask as ZW

counter = None
lock = None
ndata = None

def platemjdfiber2targetid(plate, mjd, fiber):
    return plate*1000000000 + mjd*10000 + fiber
def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return (plate, mjd, fiber)

def read_cat(pathData,zmin=None,zmax=None,zkey='Z_VI',unique=True):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)
    for k in ['PLATE','MJD','FIBERID','THING_ID','RA','DEC']:
        dic[k] = h[1][k][:]
    dic['Z'] = h[1][zkey][:]
    h.close()
    dic['TARGETID'] = platemjdfiber2targetid(dic['PLATE'].astype('int64'),dic['MJD'].astype('int64'),dic['FIBERID'].astype('int64'))
    print('Found {} quasars'.format(dic['Z'].size))

    w = sp.argsort(dic['TARGETID'])
    for k in dic.keys():
        dic[k] = dic[k][w]

    w = dic['Z']!=-1.
    if unique:
        w &= dic['THING_ID']>0
        w &= dic['RA']!=dic['DEC']
        w &= dic['RA']!=0.
        w &= dic['DEC']!=0.
    if not zmin is None:
        w &= dic['Z']>zmin
    if not zmax is None:
        w &= dic['Z']<zmax
    for k in dic.keys():
        dic[k] = dic[k][w]

    if unique:
        _, w = sp.unique(dic['THING_ID'], return_index=True)
        print('Unique: {}'.format(w.size))
        for k in dic.keys():
            dic[k] = dic[k][w]

    return dic
def read_spec_spplate(p,m,fiber=None,path_spec=None, lambda_min=None, lambda_max=None, veto_lines=None, flux_calib=None, ivar_calib=None):
    """


    """
    path = path_spec+'/{}/spPlate-{}-{}.fits'.format(str(p).zfill(4),str(p).zfill(4),m)

    h = fitsio.FITS(path)
    fl = h[0].read()
    iv = h[1].read()*(h[2].read()==0)
    head = h[0].read_header()
    h.close()

    ll = head['CRVAL1'] + head['CD1_1']*sp.arange(head['NAXIS1'])
    if head['DC-FLAG']:
        ll = 10**ll

    w = sp.ones(ll.size,dtype=bool)
    if not lambda_min is None:
        w &= ll>lambda_min
    if not lambda_max is None:
        w &= ll<lambda_min
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

    if not fiber is None:
        fl = fl[fiber-1]
        iv = iv[fiber-1]

    return ll, fl, iv
def fit_spec(lamRF, flux, ivar, qso_pca=None):

    model = sp.array([ el(lamRF) for el in qso_pca ])
    def chi2(a0):
        y = flux-a0*model[0]
        return (y**2*ivar).sum()

    a0 = abs(flux.mean())
    mig = iminuit.Minuit(chi2,a0=a0,error_a0=a0/2.,errordef=1.,print_level=-1)
    mig.migrad()

    return mig.values['a0']*model[0]
def fit_spec_redshift(z, lam, flux, ivar, qso_pca=None, dv_prior=None):
    """

    """

    def chi2(zl,a0,a1,a2,a3):
        par = sp.array([a0,a1,a2,a3])
        model = sp.array([ el(lam/(1.+zl)) for el in qso_pca ])
        model = (model*par[:,None]).sum(axis=0)
        y = flux-model
        return (y**2*ivar).sum()

    a0 = abs(flux.mean())
    dz = utils.get_dz(dv_prior,z)
    limit_zl = (z-dz/2.,z+dz/2.)
    mig = iminuit.Minuit(chi2,
        zl=z,error_zl=0.01,limit_zl=limit_zl,
        a0=a0,error_a0=a0/2.,
        a1=0.,error_a1=0.1,
        a2=0.,error_a2=0.1,
        a3=0.,error_a3=0.1,
        errordef=1.,print_level=-1)
    mig.migrad()

    z = mig.values['zl']
    zerr = mig.errors['zl']
    zwarn = mig.get_fmin()['is_valid']
    fval = mig.get_fmin()['fval']

    model = sp.ones(flux.size)
    def chi2(a0):
        y = flux-a0*model
        return (y**2*ivar).sum()

    a0 = abs(flux.mean())
    mig = iminuit.Minuit(chi2,
        a0=a0,error_a0=a0/2.,
        errordef=1.,print_level=-1)
    mig.migrad()
    deltachi2 = mig.get_fmin()['fval']-fval

    return z, zerr, zwarn, fval, deltachi2
def get_VAR_SNR(DRQ, path_spec, lines, qso_pca, zmin=0., zmax=10., zkey='Z_VI', lambda_min=3600., lambda_max=7235.,
    veto_lines=None, flux_calib=None, ivar_calib=None, nspec=None):
    """

    """

    ### Read quasar catalog
    catQSO = read_cat(DRQ,zmin,zmax,zkey)
    print('Found {} quasars'.format(catQSO['Z'].size))

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib)

    p_fit_spec = partial(fit_spec, qso_pca=qso_pca)

    ### Sort PLATE-MJD
    pm = catQSO['PLATE'].astype('int64')*100000 + catQSO['MJD'].astype('int64')
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
                    valline[side+'_NB'] = w.sum()
                    if w.sum()>=2:
                        model = p_fit_spec(lamRF[w], tfl[w], tiv[w])
                        valline[side+'_VAR'] = utils.weighted_var(tfl[w]/model-1.,tiv[w])
                        valline[side+'_SNR'] = ( (tfl[w]*tiv[w])**2 ).mean()
                    else:
                        valline[side+'_VAR'] = 0.
                        valline[side+'_SNR'] = 0.
                data[t][ln] = valline

        if not nspec is None and len(data.keys())>nspec:
            print('{}:'.format(len(data.keys())))
            return data

    return data
def fit_line(catQSO, path_spec, lines, qso_pca, dv_prior, lambda_min=None, lambda_max=None,
    veto_lines=None, flux_calib=None, ivar_calib=None, dwave_side=100):
    """

    """

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib)

    p_fit_spec = partial(fit_spec_redshift, qso_pca=qso_pca, dv_prior=dv_prior)

    ### Sort PLATE-MJD
    pm = catQSO['PLATE'].astype('int64')*100000 + catQSO['MJD'].astype('int64')
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
            path = path_spec+'/{}/spPlate-{}-{}.fits'.format(str(p).zfill(4),str(p).zfill(4),m)
            print('WARNING: Can not find PLATE={}, MJD={}: {}'.format(p,m,path))
            continue
        #print('{}: read {} objects from PLATE={}, MJD={}'.format(len(data.keys()),w.sum(),p,m))

        thids = catQSO['TARGETID'][w]
        fibs = catQSO['FIBERID'][w]
        zs = catQSO['Z'][w]

        for i in range(w.sum()):
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1

            t = thids[i]
            f = fibs[i]
            z = zs[i]

            tfl = fl[f-1]
            tiv = iv[f-1]
            lamRF = lam/(1.+z)
            data[t] = { 'ZPRIOR':z }

            for ln, lv in lines.items():

                valline = {'Z':-1., 'ZERR':-1., 'ZWARN': 0, 'CHI2':-1., 'DCHI2':0., 'NPIXBLUE':0, 'NPIXRED':0, 'NPIX':0}

                w = tiv>0.
                if not ln=='PCA':
                    valline['NPIXBLUE'] = ( (tiv>0.) & (lamRF>lv-dwave_side) & (lamRF<lv) ).sum()
                    valline['NPIXRED'] = ( (tiv>0.) & (lamRF>=lv) & (lamRF<lv+dwave_side) ).sum()
                    w &= (lamRF>lv-dwave_side) & (lamRF<lv+dwave_side)
                valline['NPIX'] = w.sum()

                if valline['NPIX']>0:
                    valline['Z'], valline['ZERR'], zwarn, valline['CHI2'], valline['DCHI2'] = p_fit_spec(z, lam[w], tfl[w], tiv[w])
                    if not zwarn:
                        valline['ZWARN'] |= ZW.BAD_MINFIT

                data[t][ln] = valline

    return data
