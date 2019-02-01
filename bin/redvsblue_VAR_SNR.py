#!/usr/bin/env python
import argparse
import fitsio
import scipy as sp
import copy
from functools import partial
from multiprocessing import Pool,Lock,cpu_count,Value

from redvsblue import read_SDSS_data, constants, utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the variance at the blue and red side of some given emission lines')

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

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=7235.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--no-cut-ANDMASK', action='store_true', required=False,
        help='Do not cut pixels with AND_MASK!=0')

    parser.add_argument('--qso-pca',type=str,default=None,required=True,
        help='Path to quasar PCA')

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

    args = parser.parse_args()

    ###
    lines = constants.lines
    zmin = 10.
    zmax = 0.
    for lv in lines.values():
        if not args.z_min is None:
            args.z_min = min(args.z_min,args.lambda_min/lv['RED_MAX']-1.)
        if not args.z_max is None:
            args.z_max = max(args.z_max,args.lambda_max/lv['BLUE_MIN']-1.)
    print('zmin = {}'.format(zmin))
    print('zmax = {}'.format(zmax))

    ###
    qso_pca = utils.read_PCA(args.qso_pca,dim=True,smooth=None)
    if not args.flux_calib is None:
        args.flux_calib = utils.read_flux_calibration(args.flux_calib)
    if not args.ivar_calib is None:
        args.ivar_calib = utils.read_ivar_calibration(args.ivar_calib)
    if not args.mask_file is None:
        args.mask_file = utils.read_mask_lines(args.mask_file)

    ### Read quasar catalog
    catQSO = read_SDSS_data.read_cat(args.drq,
        zmin=args.z_min, zmax=args.z_max, zkey=args.z_key,
        extinction=(not args.no_extinction_correction))
    print('Found {} quasars'.format(catQSO['Z'].size))

    if not args.nspec is None and args.nspec<catQSO['Z'].size:
        for k in catQSO.keys():
            catQSO[k] = catQSO[k][:args.nspec]
        print('Limit to {} quasars'.format(catQSO['Z'].size))

    ### Read spectra
    p_get_VAR_SNR = partial(read_SDSS_data.get_VAR_SNR, path_spec=args.in_dir, lines=lines, qso_pca=qso_pca,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        veto_lines=args.mask_file, flux_calib=args.flux_calib, ivar_calib=args.ivar_calib,
        extinction=(not args.no_extinction_correction), cutANDMASK=(not args.no_cut_ANDMASK))

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
    tdata = pool.map(p_get_VAR_SNR,cpu_data.values())
    pool.close()

    ### Put in one data set
    data = tdata[0]
    for td in tdata[1:]:
        for t in td.keys():
            data[t] = td[t]

    ### Save
    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'} ]
    dic = {}
    dic['TARGETID'] = sp.array([ t for t in data.keys() ])
    dic['Z'] = sp.array([ data[t]['Z'] for t in data.keys() ])
    for ln in lines.keys():
        for side in ['BLUE','RED']:
            for k in ['NB','VAR','SNR']:
                dic[ln+'_'+side+'_'+k] = sp.array([ data[t][ln][side+'_'+k] for t in data.keys() ])
    out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname='CAT')
    out.close()
