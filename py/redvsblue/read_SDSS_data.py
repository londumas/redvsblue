from __future__ import print_function
import fitsio
import iminuit
from functools import partial
import scipy as sp
import scipy.special

from redvsblue import utils
from redvsblue.utils import print
from redvsblue.zwarning import ZWarningMask as ZW
from redvsblue._zscan import _zchi2_one
from redvsblue.fitz import minfit, maxLine, find_minima

counter = None
lock = None
ndata = None

def platemjdfiber2targetid(plate, mjd, fiber):
    return plate*1000000000 + mjd*10000 + fiber
def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return plate, mjd, fiber


def fit_spec(lamRF, flux, ivar, qso_pca=None):

    model = sp.array([ el(lamRF) for el in qso_pca ])
    def chi2(a0):
        y = flux-a0*model[0]
        return (y**2*ivar).sum()

    a0 = abs(flux.mean())
    mig = iminuit.Minuit(chi2,a0=a0,error_a0=a0/2.,errordef=1.,print_level=-1)
    mig.migrad()

    return mig.values['a0']*model[0]
def fit_spec_redshift(z, lam, flux, weight, wflux, modelpca, legendre, zrange, line,
    qso_pca=None, dv_coarse=None, dv_fine=None, nb_zmin=3, dwave_model=0.1):
    """

    """

    ### Coarse scan
    zcoeff = sp.zeros(modelpca.shape[2])
    p_zchi2_one = partial(_zchi2_one, weights=weight, flux=flux, wflux=wflux, zcoeff=zcoeff)
    chi2 = sp.array([ p_zchi2_one(el) for el in modelpca ])

    ### Loop over different minima
    results = {}
    for idxmin in find_minima(chi2)[:nb_zmin]:

        zwarn = 0

        if (chi2==9e99).sum()>0:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (zrange[idxmin], -1., zwarn, chi2[idxmin])
            continue

        zPCA = zrange[idxmin]
        if (idxmin<=1) | (idxmin>=zrange.size-2):
            zwarn |= ZW.Z_FITLIMIT

        ### Fine scan
        Dz = utils.get_dz(dv_coarse,zPCA)
        dz = utils.get_dz(dv_fine,zPCA)
        tzrange = sp.linspace(zPCA-2.*Dz,zPCA+2.*Dz,1+int(round(4.*Dz/dz)))
        tchi2 = sp.array([ p_zchi2_one(sp.append( sp.array([ el(lam/(1.+tz)) for el in qso_pca ]).T,legendre,axis=1)) for tz in tzrange ])
        tidxmin = 2+sp.argmin(tchi2[2:-2])

        if (tchi2==9e99).sum()>0:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (tzrange[tidxmin], -1., zwarn, tchi2[tidxmin])
            continue

        ### Precise z_PCA
        tresult = minfit(tzrange[tidxmin-1:tidxmin+2],tchi2[tidxmin-1:tidxmin+2])
        if tresult is None:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (tzrange[tidxmin], -1., zwarn, tchi2[tidxmin])
        else:
            zPCA, zerr, fval, tzwarn = tresult
            zwarn |= tzwarn
            results[idxmin] = (zPCA, zerr, zwarn, fval)

    idx_min = sp.array([ k for k in results.keys() ])[sp.argmin([ v[3] for v in results.values() ])]
    zPCA, zerr, zwarn, fval = results[idx_min]

    ### Observed wavelength of maximum of line
    if line!='PCA':

        ### Get coefficient of the model
        model = sp.append( sp.array([ el(lam/(1.+zPCA)) for el in qso_pca ]).T,legendre,axis=1)
        p_zchi2_one(model)

        ### Get finer model
        tlam = sp.arange(lam.min(), lam.max(), dwave_model)
        tlegendre = sp.array([scipy.special.legendre(i)( (tlam-tlam.min())/(tlam.max()-tlam.min())*2.-1. ) for i in range(legendre.shape[1])]).T
        model = sp.append( sp.array([ el(tlam/(1.+zPCA)) for el in qso_pca ]).T,tlegendre,axis=1)
        model = model.dot(zcoeff)

        ### Find min
        idxmin = sp.argmax(model)
        if (idxmin<=1) | (idxmin>=model.size-2):
            zwarn |= ZW.Z_FITLIMIT

        tresult = maxLine(tlam[idxmin-1:idxmin+2],model[idxmin-1:idxmin+2])
        if tresult is None:
            zwarn |= ZW.BAD_MINFIT
            lLine = tlam[idxmin]
        else:
            zwarn |= tresult[3]
            lLine = tresult[0]
    else:
        lLine = -1.

    ### No peak fit
    zcoeff = sp.zeros(legendre.shape[1])
    zchi2 = _zchi2_one(legendre, weight, flux, wflux, zcoeff)
    deltachi2 = zchi2

    return lLine, zPCA, zerr, zwarn, fval, deltachi2


def read_cat(pathData,zmin=None,zmax=None,zkey='Z_VI',extinction=True):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)
    h[1].read_header()
    if 'MJD' in h[1].get_colnames():
        lst = {'PLATE':'PLATE','MJD':'MJD','FIBERID':'FIBERID'}
    else:
        lst = {'PLATE':'PLATE','MJD':'SMJD','FIBERID':'FIBER'}
    for k,v in lst.items():
        dic[k] = h[1][v][:]
    dic['Z'] = h[1][zkey][:]
    if extinction:
        dic['G_EXTINCTION'] = h[1]['EXTINCTION'][:][:,1]
    h.close()

    dic['TARGETID'] = platemjdfiber2targetid(dic['PLATE'].astype('int64'),dic['MJD'].astype('int64'),dic['FIBERID'].astype('int64'))
    print('Found {} quasars'.format(dic['Z'].size))

    w = sp.argsort(dic['TARGETID'])
    for k in dic.keys():
        dic[k] = dic[k][w]

    w = dic['Z']>-1.
    w &= dic['Z']!=0.
    if not zmin is None:
        w &= dic['Z']>zmin
    if not zmax is None:
        w &= dic['Z']<zmax
    for k in dic.keys():
        dic[k] = dic[k][w]

    return dic
def read_spec_spplate(p,m,fiber=None,path_spec=None,
        lambda_min=None, lambda_max=None, cutANDMASK=True,
        veto_lines=None, flux_calib=None, ivar_calib=None):
    """


    """
    path = path_spec+'/{}/spPlate-{}-{}.fits'.format(str(p).zfill(4),str(p).zfill(4),m)

    h = fitsio.FITS(path)
    fl = h[0].read()
    iv = h[1].read()
    iv *= iv>0.
    an = h[2].read()
    head = h[0].read_header()
    h.close()

    if cutANDMASK:
        iv *= an==0

    ll = head['CRVAL1'] + head['CD1_1']*sp.arange(head['NAXIS1'])
    if head['DC-FLAG']:
        ll = 10**ll

    w = sp.ones(ll.size,dtype=bool)
    if not lambda_min is None:
        w &= ll>=lambda_min
    if not lambda_max is None:
        w &= ll<=lambda_max
    if not veto_lines is None:
        for lmin,lmax in veto_lines:
            w &= (ll<lmin) | (ll>lmax)

    ll = ll[w]
    fl = fl[:,w]
    iv = iv[:,w]

    if not flux_calib is None:
        correction = flux_calib(ll)
        fl /= correction[None,:]
        iv *= correction[None,:]**2
    if not ivar_calib is None:
        correction = ivar_calib(ll)
        iv /= correction[None,:]

    if not fiber is None:
        w = iv[fiber-1]>0.
        ll = ll[w]
        fl = fl[fiber-1,w]
        iv = iv[fiber-1,w]

    return ll, fl, iv








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
    veto_lines=None, flux_calib=None, ivar_calib=None, dwave_side=85., deg_legendre=3,
    dv_coarse=100., dv_fine=10., nb_zmin=3, extinction=True, cutANDMASK=True):
    """

    """

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,cutANDMASK=cutANDMASK)

    p_fit_spec = partial(fit_spec_redshift, qso_pca=qso_pca, dv_coarse=dv_coarse, dv_fine=dv_fine, nb_zmin=nb_zmin)

    ### get PLATE-MJD
    pm = catQSO['PLATE'].astype('int64')*100000 + catQSO['MJD'].astype('int64')
    upm = sp.sort(sp.unique(pm))

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

        if lam.size==0:
            print('WARNING: No data in PLATE={}, MJD={}: {}'.format(p,m,path))
            continue

        thids = catQSO['TARGETID'][w]
        fibs = catQSO['FIBERID'][w]
        zs = catQSO['Z'][w]
        if extinction:
            extg = catQSO['G_EXTINCTION'][w]
        wfl = fl*iv

        for i in range(w.sum()):
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1

            t = thids[i]
            f = fibs[i]
            z = zs[i]

            tfl = fl[f-1]
            tiv = iv[f-1]
            twfl = wfl[f-1]
            lamRF = lam/(1.+z)
            if extinction:
                unred = utils.unred(lam,extg[i])
                tfl /= unred
                tiv *= unred**2
                twfl *= unred

            Dz = utils.get_dz(dv_prior,z)
            dz = utils.get_dz(dv_coarse,z)
            zrange = sp.linspace(z-Dz,z+Dz,1+int(round(2.*Dz/dz)))
            modelpca = sp.array([ sp.array([ el(lam/(1.+tz)) for el in qso_pca ]).T for tz in zrange ])

            data[t] = { 'ZPRIOR':z }
            for ln, lv in lines.items():
                valline = {'ZLINE':-1., 'ZPCA':-1., 'ZERR':-1., 'ZWARN': 0, 'CHI2':9e99, 'DCHI2':9e99,
                'NPIXBLUE':0, 'NPIXRED':0, 'NPIX':0, 'NPIXBLUEBEST':0, 'NPIXREDBEST':0, 'NPIXBEST':0}

                w = tiv>0.
                if not ln=='PCA':
                    valline['NPIXBLUE'] = ( w & (lamRF>lv-dwave_side) & (lamRF<lv) ).sum()
                    valline['NPIXRED'] = ( w & (lamRF>=lv) & (lamRF<lv+dwave_side) ).sum()
                    w &= (lamRF>lv-dwave_side) & (lamRF<lv+dwave_side)
                valline['NPIX'] = w.sum()

                if valline['NPIX']>0:
                    legendre = sp.array([scipy.special.legendre(i)( (lam[w]-lam[w].min())/(lam[w].max()-lam[w].min())*2.-1. ) for i in range(deg_legendre)]).T
                    tmodelpca = sp.array([ sp.append(modelpca[i,w,:],legendre,axis=1) for i in range(modelpca.shape[0]) ])
                    valline['ZLINE'], valline['ZPCA'], valline['ZERR'], valline['ZWARN'], valline['CHI2'], valline['DCHI2'] = p_fit_spec(z,
                        lam[w], tfl[w], tiv[w], twfl[w], tmodelpca, legendre, zrange, ln)

                if (not ln=='PCA') and (valline['ZLINE']!=-1.):
                    w = tiv>0.
                    tlamRF = lam*lv/valline['ZLINE']
                    valline['NPIXBLUEBEST'] = ( w & (tlamRF>lv-dwave_side) & (tlamRF<lv) ).sum()
                    valline['NPIXREDBEST'] = ( w & (tlamRF>=lv) & (tlamRF<lv+dwave_side) ).sum()
                    valline['NPIXBEST'] = (w & (tlamRF>lv-dwave_side) & (tlamRF<lv+dwave_side) ).sum()

                data[t][ln] = valline

    return data
