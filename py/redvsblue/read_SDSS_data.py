from __future__ import print_function
import os
import fitsio
from functools import partial
import scipy as sp
import scipy.special
import glob

from redvsblue.utils import print, get_dz, unred, transmission_Lyman, weighted_var
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


def fit_spec(z, lam, flux, weight, wflux, qso_pca=None):

    zcoeff = sp.zeros(len(qso_pca))
    model = sp.array([ el(lam/(1.+z)) for el in qso_pca ]).T
    _zchi2_one(model, weights=weight, flux=flux, wflux=wflux, zcoeff=zcoeff)
    model = model.dot(zcoeff)

    return model
def fit_spec_redshift(z, lam, flux, weight, wflux, modelpca, legendre, zrange, line,
    qso_pca=None, dv_coarse=None, dv_fine=None, nb_zmin=3, dwave_model=0.1, correct_lya=False):
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
        Dz = get_dz(dv_coarse,zPCA)
        dz = get_dz(dv_fine,zPCA)
        tzrange = sp.linspace(zPCA-2.*Dz,zPCA+2.*Dz,1+int(round(4.*Dz/dz)))
        tmodelpca = sp.array([ sp.append( sp.array([ el(lam/(1.+tz)) for el in qso_pca ]).T,legendre,axis=1) for tz in tzrange ])
        if correct_lya:
            tmodelpca[:,:,0] *= sp.array([ transmission_Lyman(tz,lam) for tz in tzrange ])
        tchi2 = sp.array([ p_zchi2_one(el) for el in tmodelpca ])
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
        if correct_lya:
            model[:,0] *= transmission_Lyman(zPCA,lam)
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


def read_cat(pathData,zmin=None,zmax=None,zkey='Z_VI',extinction=True,stack_obs=False,in_dir=None,nspec=None):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)
    h[1].read_header()
    if 'MJD' in h[1].get_colnames():
        lst = {'PLATE':'PLATE','MJD':'MJD','FIBERID':'FIBERID', 'THING_ID':'THING_ID' }
    else:
        lst = {'PLATE':'PLATE','MJD':'SMJD','FIBERID':'FIBER', 'THING_ID':'THING_ID' }
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

    print('Found {} quasars'.format(dic['Z'].size))
    if not nspec is None and nspec<dic['Z'].size:
        for k in dic.keys():
            dic[k] = dic[k][:nspec]
        print('Limit to {} quasars'.format(dic['Z'].size))

    if stack_obs:

        _, w = sp.unique(dic['THING_ID'],return_index=True)
        for k in dic.keys():
            dic[k] = dic[k][w]
        print('Get unique THING_ID: {}'.format(dic['Z'].size))

        w = sp.argsort(dic['THING_ID'])
        for k in dic.keys():
            dic[k] = dic[k][w]

        spall = glob.glob(os.path.expandvars(in_dir+'/spAll-*.fits'))
        assert len(spall)==1

        h = fitsio.FITS(spall[0])
        print('INFO: reading spAll from {}'.format(spall[0]))
        thid_spall = h[1]['THING_ID'][:]
        plate_spall = h[1]['PLATE'][:]
        mjd_spall = h[1]['MJD'][:]
        fid_spall = h[1]['FIBERID'][:]
        qual_spall = h[1]['PLATEQUALITY'][:]
        zwarn_spall = h[1]['ZWARNING'][:]
        h.close()

        w = sp.in1d(thid_spall, dic['THING_ID']) & (qual_spall == b'good')
        ## Removing spectra with the following ZWARNING bits set:
        ## SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET, NODATA
        ## https://www.sdss.org/dr14/algorithms/bitmasks/#ZWARNING
        for zwarnbit in [0,1,7,8,9]:
            w &= (zwarn_spall&2**zwarnbit)==0
        thid = thid_spall[w]
        plate = plate_spall[w]
        mjd = mjd_spall[w]
        fid = fid_spall[w]
        targetid = platemjdfiber2targetid(plate.astype('int64'),mjd.astype('int64'),fid.astype('int64'))

        print('INFO: # unique objs: ',dic['THING_ID'].size)
        print('INFO: # unique objs in spAll: ',sp.unique(thid).size)
        print('INFO: # spectra: ',w.sum())

        w = sp.argsort(thid)
        thid = thid[w]
        targetid = targetid[w]

        dic['ALLOBS'] = [ sp.sort(targetid[thid==t]) for t in dic['THING_ID'] ]

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
def read_spec_spec(p,m,f,path_spec=None,
        lambda_min=None, lambda_max=None, cutANDMASK=True,
        veto_lines=None, flux_calib=None, ivar_calib=None):
    """


    """
    path = path_spec+'/spectra/lite/{}/spec-{}-{}-{}.fits'.format(str(p).zfill(4),str(p).zfill(4),m,str(f).zfill(4))

    h = fitsio.FITS(path)
    fl = h['COADD']['FLUX'][:]
    iv = h['COADD']['IVAR'][:]
    an = h['COADD']['AND_MASK'][:]
    ll = h['COADD']['LOGLAM'][:]
    head = h[0].read_header()
    h.close()

    iv *= iv>0.
    if cutANDMASK:
        iv *= an==0

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
    iv[~w] = 0.

    if not flux_calib is None:
        correction = flux_calib(ll)
        fl /= correction
        iv *= correction**2
    if not ivar_calib is None:
        correction = ivar_calib(ll)
        iv /= correction

    return ll, fl, iv




def get_VAR_SNR(catQSO, path_spec, lines, qso_pca, lambda_min=None, lambda_max=None,
    veto_lines=None, flux_calib=None, ivar_calib=None, extinction=True, cutANDMASK=True):

    """

    """

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,cutANDMASK=cutANDMASK)

    p_fit_spec = partial(fit_spec, qso_pca=qso_pca)

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

            w = iv[f-1]>0.
            tlam = lam[w]
            tfl = fl[f-1,w]
            tiv = iv[f-1,w]
            twfl = wfl[f-1,w]
            lamRF = tlam/(1.+z)
            if extinction:
                tunred = unred(tlam,extg[i])
                tfl /= tunred
                tiv *= tunred**2
                twfl *= tunred

            data[t] = { 'Z':z }
            for ln, lv in lines.items():
                valline = {}
                for side in ['BLUE','RED']:
                    w = (tiv>0.) & (lamRF>lv[side+'_MIN']) & (lamRF<lv[side+'_MAX'])
                    valline[side+'_NB'] = w.sum()
                    if w.sum()>2*len(qso_pca):
                        model = p_fit_spec(z, tlam[w], tfl[w], tiv[w], twfl[w])
                        valline[side+'_VAR'] = weighted_var(tfl[w]/model-1.,tiv[w])
                        valline[side+'_SNR'] = ( (tfl[w]*tiv[w])**2 ).mean()
                    else:
                        valline[side+'_VAR'] = 0.
                        valline[side+'_SNR'] = 0.
                data[t][ln] = valline

    return data


def fit_line_spplate(catQSO, path_spec, lines, qso_pca, dv_prior, lambda_min=None, lambda_max=None,
    veto_lines=None, flux_calib=None, ivar_calib=None, dwave_side=85., deg_legendre=3,
    dv_coarse=100., dv_fine=10., nb_zmin=3, extinction=True, cutANDMASK=True, dwave_model=0.1,
    correct_lya=False):
    """

    """

    ###
    p_read_spec_spplate = partial(read_spec_spplate, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,cutANDMASK=cutANDMASK)

    p_fit_spec = partial(fit_spec_redshift, qso_pca=qso_pca, dv_coarse=dv_coarse, dv_fine=dv_fine, nb_zmin=nb_zmin,
        dwave_model=dwave_model, correct_lya=correct_lya)

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

        targetids = catQSO['TARGETID'][w]
        thids = catQSO['THING_ID'][w]
        fibs = catQSO['FIBERID'][w]
        zs = catQSO['Z'][w]
        if extinction:
            extg = catQSO['G_EXTINCTION'][w]
        wfl = fl*iv

        for i in range(w.sum()):
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1

            t = targetids[i]
            f = fibs[i]
            z = zs[i]

            w = iv[f-1]>0.
            tlam = lam[w]
            tfl = fl[f-1,w]
            tiv = iv[f-1,w]
            twfl = wfl[f-1,w]
            lamRF = tlam/(1.+z)
            if extinction:
                tunred = unred(tlam,extg[i])
                tfl /= tunred
                tiv *= tunred**2
                twfl *= tunred

            Dz = get_dz(dv_prior,z)
            dz = get_dz(dv_coarse,z)
            zrange = sp.linspace(z-Dz,z+Dz,1+int(round(2.*Dz/dz)))
            modelpca = sp.array([ sp.array([ el(tlam/(1.+tz)) for el in qso_pca ]).T for tz in zrange ])
            if correct_lya:
                modelpca[:,:,0] *= sp.array([ transmission_Lyman(tz,tlam) for tz in zrange ])

            data[t] = { 'ZPRIOR':z, 'THING_ID':thids[i] }
            for ln, lv in lines.items():
                valline = {'ZLINE':-1., 'ZPCA':-1., 'ZERR':-1., 'ZWARN': 0, 'CHI2':9e99, 'DCHI2':9e99,
                'NPIXBLUE':0, 'NPIXRED':0, 'NPIX':0, 'NPIXBLUEBEST':0, 'NPIXREDBEST':0, 'NPIXBEST':0}

                w = tiv>0.
                if not ln=='PCA':
                    valline['NPIXBLUE'] = ( w & (lamRF>lv-dwave_side) & (lamRF<lv) ).sum()
                    valline['NPIXRED'] = ( w & (lamRF>=lv) & (lamRF<lv+dwave_side) ).sum()
                    w &= (lamRF>lv-dwave_side) & (lamRF<lv+dwave_side)
                valline['NPIX'] = w.sum()

                if valline['NPIX']>1:
                    legendre = sp.array([scipy.special.legendre(i)( (tlam[w]-tlam[w].min())/(tlam[w].max()-tlam[w].min())*2.-1. ) for i in range(deg_legendre)]).T
                    tmodelpca = sp.array([ sp.append(modelpca[i,w,:],legendre,axis=1) for i in range(modelpca.shape[0]) ])
                    valline['ZLINE'], valline['ZPCA'], valline['ZERR'], valline['ZWARN'], valline['CHI2'], valline['DCHI2'] = p_fit_spec(z,
                        tlam[w], tfl[w], tiv[w], twfl[w], tmodelpca, legendre, zrange, ln)

                if (not ln=='PCA') and (valline['ZLINE']!=-1.):
                    w = tiv>0.
                    tlamRF = tlam*lv/valline['ZLINE']
                    valline['NPIXBLUEBEST'] = ( w & (tlamRF>lv-dwave_side) & (tlamRF<lv) ).sum()
                    valline['NPIXREDBEST'] = ( w & (tlamRF>=lv) & (tlamRF<lv+dwave_side) ).sum()
                    valline['NPIXBEST'] = (w & (tlamRF>lv-dwave_side) & (tlamRF<lv+dwave_side) ).sum()

                data[t][ln] = valline

    return data
def fit_line_spec(catQSO, path_spec, lines, qso_pca, dv_prior, lambda_min=None, lambda_max=None,
    veto_lines=None, flux_calib=None, ivar_calib=None, dwave_side=85., deg_legendre=3,
    dv_coarse=100., dv_fine=10., nb_zmin=3, extinction=True, cutANDMASK=True, dwave_model=0.1,
    correct_lya=False):
    """

    """

    ###
    p_read_spec_spec = partial(read_spec_spec, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,cutANDMASK=cutANDMASK)

    p_fit_spec = partial(fit_spec_redshift, qso_pca=qso_pca, dv_coarse=dv_coarse, dv_fine=dv_fine, nb_zmin=nb_zmin,
        dwave_model=dwave_model, correct_lya=correct_lya)

    data = {}
    for i, thids in enumerate(catQSO['THING_ID']):
        print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
        with lock:
            counter.value += 1

        t = catQSO['TARGETID'][i]
        z = catQSO['Z'][i]
        if extinction:
            extg = catQSO['G_EXTINCTION'][i]

        ll = None
        fl = None
        iv = None
        for tobs in catQSO['ALLOBS'][i]:
            p, m, f = targetid2platemjdfiber(tobs)
            tlam, tfl, tiv = p_read_spec_spec(p,m,f)
            if ll is None:
                ll = sp.log10(tlam)
                fl = tfl
                iv = tiv
            else:
                ll = sp.append(ll,sp.log10(tlam))
                fl = sp.append(fl,tfl)
                iv = sp.append(iv,tiv)

        dll = 1e-4
        lmin = ll.min()
        bins = sp.floor((ll-lmin)/dll+0.5).astype(int)
        ll = lmin + bins*dll
        w = ll>=sp.log10(lambda_min)
        w &= ll<sp.log10(lambda_max)
        w &= iv>0.
        bins = bins[w]
        ll = ll[w]
        fl = fl[w]
        iv = iv[w]

        cll = lmin + sp.arange(bins.max()+1)*dll
        cfl = sp.zeros(bins.max()+1)
        civ = sp.zeros(bins.max()+1)
        ccfl = sp.bincount(bins,weights=iv*fl)
        cciv = sp.bincount(bins,weights=iv)
        cfl[:len(ccfl)] += ccfl
        civ[:len(cciv)] += cciv
        w = civ>0.
        lam = 10**(cll[w])
        fl = cfl[w]/civ[w]
        iv = civ[w]

        if lam.size==0:
            print('WARNING: No data for THING_ID = {}'.format(thids))
            continue

        wfl = fl*iv
        lamRF = lam/(1.+z)
        if extinction:
            tunred = unred(lam,extg)
            fl /= tunred
            iv *= tunred**2
            wfl *= tunred

        Dz = get_dz(dv_prior,z)
        dz = get_dz(dv_coarse,z)
        zrange = sp.linspace(z-Dz,z+Dz,1+int(round(2.*Dz/dz)))
        modelpca = sp.array([ sp.array([ el(lam/(1.+tz)) for el in qso_pca ]).T for tz in zrange ])
        if correct_lya:
            modelpca[:,:,0] *= sp.array([ transmission_Lyman(tz,lam) for tz in zrange ])

        data[t] = { 'ZPRIOR':z, 'THING_ID':thids }
        for ln, lv in lines.items():
            valline = {'ZLINE':-1., 'ZPCA':-1., 'ZERR':-1., 'ZWARN': 0, 'CHI2':9e99, 'DCHI2':9e99,
            'NPIXBLUE':0, 'NPIXRED':0, 'NPIX':0, 'NPIXBLUEBEST':0, 'NPIXREDBEST':0, 'NPIXBEST':0}

            w = iv>0.
            if not ln=='PCA':
                valline['NPIXBLUE'] = ( w & (lamRF>lv-dwave_side) & (lamRF<lv) ).sum()
                valline['NPIXRED'] = ( w & (lamRF>=lv) & (lamRF<lv+dwave_side) ).sum()
                w &= (lamRF>lv-dwave_side) & (lamRF<lv+dwave_side)
            valline['NPIX'] = w.sum()

            if valline['NPIX']>1:
                legendre = sp.array([scipy.special.legendre(i)( (lam[w]-lam[w].min())/(lam[w].max()-lam[w].min())*2.-1. ) for i in range(deg_legendre)]).T
                tmodelpca = sp.array([ sp.append(modelpca[i,w,:],legendre,axis=1) for i in range(modelpca.shape[0]) ])
                valline['ZLINE'], valline['ZPCA'], valline['ZERR'], valline['ZWARN'], valline['CHI2'], valline['DCHI2'] = p_fit_spec(z,
                    lam[w], fl[w], iv[w], wfl[w], tmodelpca, legendre, zrange, ln)

            if (not ln=='PCA') and (valline['ZLINE']!=-1.):
                w = iv>0.
                tlamRF = lam*lv/valline['ZLINE']
                valline['NPIXBLUEBEST'] = ( w & (tlamRF>lv-dwave_side) & (tlamRF<lv) ).sum()
                valline['NPIXREDBEST'] = ( w & (tlamRF>=lv) & (tlamRF<lv+dwave_side) ).sum()
                valline['NPIXBEST'] = (w & (tlamRF>lv-dwave_side) & (tlamRF<lv+dwave_side) ).sum()

            data[t][ln] = valline

    return data
