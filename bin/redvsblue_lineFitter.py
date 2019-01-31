#!/usr/bin/env python
import argparse
import fitsio
import scipy as sp
import copy
from functools import partial
from multiprocessing import Pool,Lock,cpu_count,Value

from redvsblue import read_SDSS_data, constants, utils
from redvsblue.zwarning import ZWarningMask as ZW


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the redshift of different quasar emmision lines')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to spectra files')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--qso-pca',type=str,default=None,required=True,
        help='Path to quasar PCA')

    parser.add_argument('--z-key', type=str, default='Z', required=False,
        help='Name of the key giving redshifts in drq')

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=10000.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--dwave-side',type=float,default=85.,required=False,
        help='Wavelength interval on both side of the line [Angstrom]')

    parser.add_argument('--npix-min',type=int,default=10,required=False,
        help='Minimum number of pixels on each side of the line')

    parser.add_argument('--nb-zmin',type=int,default=3,required=False,
        help='Number of redshift minima too inspect with a fine grid')

    parser.add_argument('--dv-prior',type=float,default=10000.,required=False,
        help='Velocity difference box prior for each side of the line [km/s]')

    parser.add_argument('--dv-coarse',type=float,default=100.,required=False,
        help='Velocity grid for the coarse determination of the minimum [km/s]')

    parser.add_argument('--dv-fine',type=float,default=10.,required=False,
        help='Velocity grid for the fine determination of the minimum [km/s]')

    parser.add_argument('--sigma-smooth',type=int,default=2,required=False,
        help='Smoothing kernel sigma for the PCA, in number of points')

    parser.add_argument('--deg-legendre',type=int,default=3,required=False,
        help='Number of Legendre Polynoms')

    parser.add_argument('--no-extinction-correction', action='store_true', required=False,
        help='Do not correct for galactic extinction')

    parser.add_argument('--mask-file',type=str,default=None,required=False,
        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    args = parser.parse_args()
    if args.nproc is None:
        args.nproc = cpu_count()//2

    ###
    lines = constants.emissionLines

    ###
    qso_pca = utils.read_PCA(args.qso_pca,dim=True,smooth=args.sigma_smooth)
    if not args.flux_calib is None:
        args.flux_calib = utils.read_flux_calibration(args.flux_calib)
    if not args.ivar_calib is None:
        args.ivar_calib = utils.read_ivar_calibration(args.ivar_calib)
    if not args.mask_file is None:
        args.mask_file = utils.read_mask_lines(args.mask_file)

    ### Read quasar catalog
    catQSO = read_SDSS_data.read_cat(args.drq,zkey=args.z_key,extinction=(not args.no_extinction_correction))
    print('Found {} quasars'.format(catQSO['Z'].size))

    if not args.nspec is None and args.nspec<catQSO['Z'].size:
        for k in catQSO.keys():
            catQSO[k] = catQSO[k][:args.nspec]
        print('Limit to {} quasars'.format(catQSO['Z'].size))

    ### Read spectra
    p_fit_line = partial(read_SDSS_data.fit_line, path_spec=args.in_dir, lines=lines, qso_pca=qso_pca,dv_prior=args.dv_prior,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        veto_lines=args.mask_file, flux_calib=args.flux_calib, ivar_calib=args.ivar_calib,
        dwave_side=args.dwave_side, deg_legendre=args.deg_legendre, dv_coarse=args.dv_coarse,
        dv_fine=args.dv_fine, nb_zmin=args.nb_zmin,extinction=(not args.no_extinction_correction))

    ### Send
    cpu_data = {}
    pm = catQSO['PLATE'].astype('int64')*100000 + catQSO['MJD'].astype('int64')
    upm = sp.sort(sp.unique(pm))
    for tupm in upm:
        w = pm==tupm
        cpu_data[tupm] = copy.deepcopy(catQSO)
        for k in cpu_data[tupm].keys():
            cpu_data[tupm][k] = cpu_data[tupm][k][w]
    read_SDSS_data.ndata = catQSO['Z'].size
    read_SDSS_data.counter = Value('i',0)
    read_SDSS_data.lock = Lock()
    pool = Pool(processes=args.nproc)
    tdata = pool.map(p_fit_line,cpu_data.values())
    pool.close()

    ### Put in one data set
    data = tdata[0]
    for td in tdata[1:]:
        for t in td.keys():
            data[t] = td[t]

    ### Save
    out = fitsio.FITS(args.out,'rw',clobber=True)

    if args.mask_file is None:
        args.mask_file = ''
    if args.flux_calib is None:
        args.flux_calib = ''
    if args.ivar_calib is None:
        args.ivar_calib = ''

    head = [ {'name':'DRQ','value':args.drq.split('/')[-1],'comment':'Object catalog with redshift prior'},
            {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'},
            {'name':'LAMMIN','value':args.lambda_min,'comment':'Lower limit on observed wavelength [Angstrom]'},
            {'name':'LAMMAX','value':args.lambda_max,'comment':'Upper limit on observed wavelength [Angstrom]'},
            {'name':'DWAVE','value':args.dwave_side,'comment':'Wavelength interval on both side of the line [Angstrom]'},
            {'name':'NPIXMIN','value':args.npix_min,'comment':'Minimum number of pixels on each side of the line'},
            {'name':'NZMIN','value':args.nb_zmin,'comment':'Number of redshift minima too inspect with a fine grid'},
            {'name':'DVPRIOR','value':args.dv_prior,'comment':'Velocity difference box prior on both side of the line [km/s]'},
            {'name':'DVCOARSE','value':args.dv_coarse,'comment':'Velocity grid for the coarse determination of the minimum [km/s]'},
            {'name':'DVFINE','value':args.dv_fine,'comment':'Velocity grid for the fine determination of the minimum [km/s]'},
            {'name':'QSOPCA','value':args.qso_pca.split('/')[-1],'comment':'Path to quasar PCA'},
            {'name':'SMOOTH','value':args.sigma_smooth,'comment':'Smoothing kernel sigma for the PCA, in number of points'},
            {'name':'NPOLY','value':args.deg_legendre,'comment':'Number of Legendre Polynoms'},
            {'name':'GALEXT','value':(not args.no_extinction_correction),'comment':'Correct for Galactic extinction'},
            {'name':'WMASK','value':args.mask_file.split('/')[-1],'comment':'Path to observed wavelength mask'},
            {'name':'FCALIB','value':args.flux_calib.split('/')[-1],'comment':'Path to flux calibration'},
            {'name':'ICALIB','value':args.ivar_calib.split('/')[-1],'comment':'Path to ivar calibration'},
            ]
    dic = {}
    dic['TARGETID'] = sp.array([ t for t in data.keys() ])
    dic['ZPRIOR'] = sp.array([ data[t]['ZPRIOR'] for t in data.keys() ])

    tw = sp.argsort(dic['TARGETID'])
    for k in dic.keys():
        dic[k] = dic[k][tw]

    out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname='CAT')

    for ln, lv in lines.items():
        dic = {}
        head = [ {'name':'LINENAME','value':ln,'comment':'Line name'},
                {'name':'LINERF','value':lv,'comment':'Line rest frame [Angstrom]'}]

        for k in ['ZLINE','ZPCA','ZERR','ZWARN','CHI2','DCHI2',
            'NPIXBLUE','NPIXRED','NPIX','NPIXBLUEBEST','NPIXREDBEST','NPIXBEST']:
            dic[k] = sp.array([ data[t][ln][k] for t in data.keys() ])
        for k in dic.keys():
            dic[k] = dic[k][tw]

        w = dic['CHI2']==9e99
        dic['ZWARN'][w] |= ZW.BAD_MINFIT

        w = (dic['CHI2']!=9e99) & (dic['DCHI2']!=9e99)
        dic['DCHI2'][w] -= dic['CHI2'][w]
        dic['DCHI2'][~w] = 0.

        w = dic['NPIX']==0.
        dic['ZWARN'][w] |= ZW.NODATA

        w = dic['NPIX']<=args.deg_legendre+len(qso_pca)
        dic['ZWARN'][w] |= ZW.LITTLE_COVERAGE

        if not ln=='PCA':
            w = (dic['NPIXBLUE']==0) | (dic['NPIXBLUEBEST']==0)
            dic['ZWARN'][w] |= ZW.NODATA_BLUE

            w = (dic['NPIXRED']==0) | (dic['NPIXREDBEST']==0)
            dic['ZWARN'][w] |= ZW.NODATA_RED

            w = (dic['NPIXBLUE']<args.npix_min) | (dic['NPIXRED']<args.npix_min)
            dic['ZWARN'][w] |= ZW.LITTLE_COVERAGE

            w = (dic['NPIXBLUEBEST']<args.npix_min) | (dic['NPIXREDBEST']<args.npix_min)
            dic['ZWARN'][w] |= ZW.LITTLE_COVERAGE

            w = dic['ZLINE']!=-1.
            dic['ZLINE'][w] = dic['ZLINE'][w]/lv-1.
        else:
            dic['ZLINE'] = -sp.ones(dic['ZLINE'].size)

        out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname=ln)

    out.close()
