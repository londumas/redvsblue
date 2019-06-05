from __future__ import print_function
import healpy
import sys
import fitsio
from functools import partial
import scipy as sp
import scipy.special

from redvsblue.utils import print, get_dz, unred, transmission_Lyman
from redvsblue.fitline import fit_spec_redshift

def read_cat(pathData,zmin=None,zmax=None,zkey='Z',spectype='QSO',
        extinction=True,stack_obs=False,in_dir=None,nspec=None,
        rvextinction=3.793,nside=64):
    """

    """

    dic = {}

    h = fitsio.FITS(pathData)

    lst = {'TARGETID':'TARGETID', 'THING_ID':'TARGETID', 'Z':zkey, 'RA':'RA', 'DEC':'DEC'}
    w = sp.char.strip(h[1]['SPECTYPE'][:].astype(str))==spectype
    for k,v in lst.items():
        dic[k] = h[1][v][:][w]
    if extinction:
        dic['G_EXTINCTION'] = h[1]['EXTINCTION'][:][w][:,1]/rvextinction
    h.close()

    nest = True
    ra = dic['RA']*sp.pi/180.
    dec = dic['DEC']*sp.pi/180.
    dic['HPXPIXEL'] = healpy.ang2pix(nside, sp.pi/2.-dec, ra,nest=nest)

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
        print('ERROR: stack_obs Not implemented')
        sys.exit()

    return dic

def read_spec(hpxpixel,targetid=None,path_spec=None,
        lambda_min=None, lambda_max=None, cutANDMASK=True,
        veto_lines=None, flux_calib=None, ivar_calib=None,
        nside=64):
    """


    """

    path = path_spec+'/{}/{}/spectra-{}-{}.fits'.format(int(hpxpixel//100),hpxpixel,nside,hpxpixel)

    h = fitsio.FITS(path)
    tids = h['FIBERMAP']['TARGETID'][:]
    specData = {}
    for spec in ['B','R','Z']:
        dic = {}
        dic['LL'] = h['{}_WAVELENGTH'.format(spec)].read()
        dic['FL'] = h['{}_FLUX'.format(spec)].read()
        dic['IV'] = h['{}_IVAR'.format(spec)].read()*(h['{}_MASK'.format(spec)].read()==0)
        w = sp.isnan(dic['FL']) | sp.isnan(dic['IV'])
        for k in ['FL','IV']:
            dic[k][w] = 0.
        dic['RESO'] = h['{}_RESOLUTION'.format(spec)].read()
        specData[spec] = dic
    h.close()

    lam = None
    for tspecData in specData.values():
        if lam is None:
            lam = tspecData['LL']
        else:
            lam = sp.append(lam,tspecData['LL'])

    data = {}
    for t in sp.unique(tids):
        data[t] = {'FLUX':None,'IVAR':None}
        wt = tids==t
        for tspecData in specData.values():
            iv = tspecData['IV'][wt]
            fl = (iv*tspecData['FL'][wt]).sum(axis=0)
            iv = iv.sum(axis=0)
            w = iv>0.
            fl[w] /= iv[w]

            if data[t]['FLUX'] is None:
                data[t]['FLUX'] = fl
                data[t]['IVAR'] = iv
            else:
                data[t]['FLUX'] = sp.append(data[t]['FLUX'],fl)
                data[t]['IVAR'] = sp.append(data[t]['IVAR'],iv)

    return lam, data

def fit_line(catQSO, path_spec, lines, qso_pca, dv_prior, lambda_min=None, lambda_max=None,
    veto_lines=None, flux_calib=None, ivar_calib=None, dwave_side=85., deg_legendre=3,
    dv_coarse=100., dv_fine=10., nb_zmin=3, extinction=True, cutANDMASK=True, dwave_model=0.1,
    correct_lya=False,no_slope=False):
    """

    """

    nside = int(path_spec.split('spectra-')[-1].replace('/',''))

    ###
    p_read_spec = partial(read_spec, path_spec=path_spec, lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,cutANDMASK=cutANDMASK,
        nside=nside)

    p_fit_spec = partial(fit_spec_redshift, qso_pca=qso_pca, dv_coarse=dv_coarse, dv_fine=dv_fine, nb_zmin=nb_zmin,
        dwave_model=dwave_model, correct_lya=correct_lya,no_slope=no_slope)

    hppixel = catQSO['HPXPIXEL']
    uhppixel = sp.sort(sp.unique(hppixel))

    data = {}
    for thppixel in uhppixel:
        w = hppixel==thppixel

        try:
            lam, fliv = p_read_spec(thppixel)
        except OSError:
            path = path_spec+'/{}/{}/spectra-{}-{}.fits'.format(int(thppixel//100),thppixel,nside,thppixel)
            print('WARNING: Can not find pixel {}: {}'.format(thppixel,path))
            continue

        if lam.size==0:
            path = path_spec+'/{}/{}/spectra-{}-{}.fits'.format(int(thppixel//100),thppixel,nside,thppixel)
            print('WARNING: No data in pixel {}: {}'.format(thppixel,path))
            continue

        targetids = catQSO['TARGETID'][w]
        thids = catQSO['THING_ID'][w]
        zs = catQSO['Z'][w]
        if extinction:
            extg = catQSO['G_EXTINCTION'][w]

        for i in range(w.sum()):
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1

            t = targetids[i]
            z = zs[i]

            w = fliv[t]['IVAR']>0.
            tlam = lam[w]
            tfl = fliv[t]['FLUX'][w]
            tiv = fliv[t]['IVAR'][w]
            twfl = tfl*tiv
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
