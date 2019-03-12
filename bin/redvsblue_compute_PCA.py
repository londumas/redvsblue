#!/usr/bin/env python
import argparse
import fitsio
import empca
from functools import partial
import scipy as sp
from scipy.interpolate import interp1d

from desispec.interpolation import resample_flux
from redvsblue import read_SDSS_data, utils
from redvsblue.utils import unred, print

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the object PCA')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to spectra files')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--z-key', type=str, default='Z', required=False,
        help='Name of the key giving redshifts in drq')

    parser.add_argument('--z-min',type=float,default=None,required=False,
        help='Minimum redshift')

    parser.add_argument('--z-max',type=float,default=None,required=False,
        help='Maximum redshift')

    parser.add_argument('--NBLL',type=int,default=13637,required=False,
        help='')

    parser.add_argument('--CRVAL1',type=float,default=2.6534,required=False,
        help='')

    parser.add_argument('--CDELT1',type=float,default=1.e-4,required=False,
        help='')

    parser.add_argument('--median-min',type=float,default=1.e-2,required=False,
        help='')

    parser.add_argument('--weight-max',type=float,default=10.,required=False,
        help='')

    parser.add_argument('--niter',type=int,default=25,required=False,
        help='')

    parser.add_argument('--nvec',type=int,default=4,required=False,
        help='')

    parser.add_argument('--no-weights', action='store_true', required=False,
        help='')

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=10000.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--no-cut-ANDMASK', action='store_true', required=False,
        help='Do not cut pixels with AND_MASK!=0')

    parser.add_argument('--npix-min',type=int,default=1000,required=False,
        help='Minimum number of pixels')

    parser.add_argument('--nmeasure-min',type=int,default=10,required=False,
        help='Minimum number of measurement at a given wavelength')

    parser.add_argument('--no-extinction-correction', action='store_true', required=False,
        help='Do not correct for galactic extinction')

    parser.add_argument('--lya-correction', action='store_true', required=False,
        help='Correct for Lyman-alpha absorption')

    parser.add_argument('--mask-file',type=str,default=None,required=False,
        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--stack-obs', action='store_true', required=False,
        help='Stack all valid observations')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')

    args = parser.parse_args()

    assert args.nmeasure_min>args.nvec

    ###
    flux_calib = args.flux_calib
    if not flux_calib is None:
        flux_calib = utils.read_flux_calibration(flux_calib)
    ivar_calib = args.ivar_calib
    if not ivar_calib is None:
        ivar_calib = utils.read_ivar_calibration(ivar_calib)
    mask_file = args.mask_file
    if not mask_file is None:
        mask_file = utils.read_mask_lines(mask_file)

    ### Read quasar catalog
    catQSO = read_SDSS_data.read_cat(args.drq,
        zmin=args.z_min, zmax=args.z_max, zkey=args.z_key,
        extinction=(not args.no_extinction_correction),
        stack_obs=args.stack_obs,in_dir=args.in_dir, nspec=args.nspec)

    ### TODO: add all re-observation, remove bad-observation, sort pm
    ### TODO: how to effitiently select a sample?
    pm = catQSO['PLATE'].astype('int64')*100000 + catQSO['MJD'].astype('int64')
    upm = sp.sort(sp.unique(pm))
    npm = sp.bincount(pm)
    w = npm>0
    npm = npm[w]
    w = sp.argsort(npm)
    upm = upm[w][::-1]
    npm = npm[w][::-1]
    #upm = upm[:500]

    ###
    p_read_spec_spplate = partial(read_SDSS_data.read_spec_spplate, path_spec=args.in_dir,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        veto_lines=mask_file, flux_calib=flux_calib, ivar_calib=ivar_calib,
        cutANDMASK=(not args.no_cut_ANDMASK))

    ###
    nbQso = sp.in1d(pm,upm).sum()
    pcawave = 10**(args.CRVAL1+sp.arange(args.NBLL)*args.CDELT1)
    pcaflux = sp.zeros((nbQso, pcawave.size),dtype='float32')
    pcaivar = sp.zeros((nbQso, pcawave.size),dtype='float32')
    print('Use {} quasars'.format(pcaflux.shape[0]))

    ###
    ### TODO: apply Lya transmission
    idxQso = 0
    for i,tpm in enumerate(upm):
        print(i)
        p = tpm//100000
        m = tpm%100000
        w = pm==tpm
        fibs = catQSO['FIBERID'][w]
        zs = catQSO['Z'][w]
        if not args.no_extinction_correction:
            extg = catQSO['G_EXTINCTION'][w]

        ll, fl, iv = p_read_spec_spplate(p,m)
        for i,f in enumerate(fibs):
            w = iv[f-1]>0.
            if w.sum()<args.npix_min:
                continue

            ttwave = ll[w]
            ttflux = fl[f-1,w]
            ttivar = iv[f-1,w]

            if not args.no_extinction_correction:
                tunred = unred(ttwave,extg[i])
                ttflux /= tunred
                ttivar *= tunred**2

            ttwave /= 1.+zs[i]

            pcaflux[idxQso],pcaivar[idxQso] = resample_flux(pcawave, ttwave, ttflux, ttivar)
            idxQso += 1

    ### TODO: still needed?
    pcaivar[pcaivar<0.] = 0.

    ### Remove if no data
    w = sp.sum(pcaivar>0.,axis=1)>0
    pcaflux = pcaflux[w]
    pcaivar = pcaivar[w]
    print('Keep {} quasars'.format(pcaflux.shape[0]))

    ### Reject too small median and normalize by median
    med = sp.array([ sp.median(pcaflux[i][pcaivar[i]>0.]) for i in range(pcaflux.shape[0]) ])
    w = med>args.median_min
    pcaflux = pcaflux[w]/med[w][:,None]
    pcaivar = pcaivar[w]*(med[w][:,None])**2
    print('Keep {} quasars'.format(pcaflux.shape[0]))

    ### Remove if all measured bins are zero
    w = sp.sum(pcaivar>0.,axis=0)>args.nmeasure_min
    pcaflux = pcaflux[:,w]
    pcaivar = pcaivar[:,w]

    ### Cap the weights
    if args.no_weights:
        pcaivar[pcaivar>0.] = 1.
    else:
        pcaivar[pcaivar>args.weight_max] = args.weight_max

    ### Get the mean
    tmeanspec = sp.zeros(pcaflux.shape[1],dtype='float32')
    for i in range(10):
        step = sp.average(pcaflux,weights=pcaivar,axis=0)
        print('INFO: Removing mean at step: ',i,step.min(), step.max())
        tmeanspec += step
        pcaflux -= step

    ### PCA
    print('INFO: Starting EMPCA')
    model = empca.empca(pcaflux, weights=pcaivar, niter=args.niter, nvec=args.nvec)
    for i in range(model.coeff.shape[0]):
        model.coeff[i] /= sp.linalg.norm(model.coeff[i])

    ### TODO: if not edge interpolate
    pcamodel = sp.zeros((model.eigvec.shape[0], pcawave.size))
    meanspec = sp.zeros((1, pcawave.size))
    meanspec[0][w] = tmeanspec
    for i in range(model.eigvec.shape[0]):
        #f = interp1d(pcawave[w],model.eigvec[i])
        pcamodel[i,w] = model.eigvec[i]

    ### Save
    ### TODO: Add missing comments
    out = fitsio.FITS(args.out,'rw',clobber=True)

    if args.z_min is None:
        args.z_min = False
    if args.z_max is None:
        args.z_max = False
    if args.mask_file is None:
        args.mask_file = ''
    if args.flux_calib is None:
        args.flux_calib = ''
    if args.ivar_calib is None:
        args.ivar_calib = ''

    head = [ {'name':'SPEC','value':args.in_dir.split('/')[-1],'comment':'Path to spectra'},
            {'name':'DRQ','value':args.drq.split('/')[-1],'comment':'Object catalog with redshift prior'},
            {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'},
            {'name':'ZMIN','value':args.z_min,'comment':'Minimum redshift'},
            {'name':'ZMAX','value':args.z_max,'comment':'Maximum redshift'},
            {'name':'NBLL','value':args.NBLL,'comment':''},
            {'name':'CRVAL1','value':args.CRVAL1,'comment':''},
            {'name':'CDELT1','value':args.CDELT1,'comment':''},
            {'name':'LOGLAM','value':True,'comment':''},
            {'name':'MEDMIN','value':args.median_min,'comment':''},
            {'name':'NITER','value':args.niter,'comment':''},
            {'name':'NVEC','value':args.nvec,'comment':''},
            {'name':'WMAX','value':args.weight_max,'comment':''},
            {'name':'NOW','value':args.no_weights,'comment':''},
            {'name':'LAMMIN','value':args.lambda_min,'comment':'Lower limit on observed wavelength [A]'},
            {'name':'LAMMAX','value':args.lambda_max,'comment':'Upper limit on observed wavelength [A]'},
            {'name':'CUTAN','value':(not args.no_cut_ANDMASK),'comment':'Do not cut pixels with AND_MASK!=0'},
            {'name':'NPIXMIN','value':args.npix_min,'comment':'Minimum number of pixels'},
            {'name':'NPIXMIN','value':args.nmeasure_min,'comment':''},
            {'name':'GALEXT','value':(not args.no_extinction_correction),'comment':'Correct for Galactic extinction'},
            {'name':'LYAABS','value':args.lya_correction,'comment':'Correct for Lya absorption'},
            {'name':'WMASK','value':args.mask_file.split('/')[-1],'comment':'Path to observed wavelength mask'},
            {'name':'FCALIB','value':args.flux_calib.split('/')[-1],'comment':'Path to flux calibration'},
            {'name':'ICALIB','value':args.ivar_calib.split('/')[-1],'comment':'Path to ivar calibration'},
            {'name':'STACKOBS','value':args.stack_obs,'comment':'Stack all good observations'},
            ]
    out.write(pcamodel,header=head,extname='BASIS_VECTORS')
    out.write(model.coeff,extname='ARCHETYPE_COEFF')
    out.write(meanspec,extname='MEAN_SPEC')
    out.close()
