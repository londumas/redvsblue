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
        description='Measure the equivalent width (EW) for a list of emmision lines')

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

    parser.add_argument('--lambda-max',type=float,default=10000.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--no-cut-ANDMASK', action='store_true', required=False,
        help='Do not cut pixels with AND_MASK!=0')

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
    lines = constants.emissionLinesEW

    ###
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
    p_get_EW = partial(read_SDSS_data.get_EW, path_spec=args.in_dir, lines=lines,
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
    tdata = pool.map(p_get_EW,cpu_data.values())
    pool.close()

    ### Put in one data set
    data = tdata[0]
    for td in tdata[1:]:
        for t in td.keys():
            data[t] = td[t]

    ### Save
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

    head = [ {'name':'DRQ','value':args.drq.split('/')[-1],'comment':'Object catalog with redshift prior'},
            {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'},
            {'name':'ZMIN','value':args.z_min,'comment':'Minimum redshift'},
            {'name':'ZMAX','value':args.z_max,'comment':'Maximum redshift'},
            {'name':'LAMMIN','value':args.lambda_min,'comment':'Lower limit on observed wavelength [A]'},
            {'name':'LAMMAX','value':args.lambda_max,'comment':'Upper limit on observed wavelength [A]'},
            {'name':'CUTAN','value':(not args.no_cut_ANDMASK),'comment':'Do not cut pixels with AND_MASK!=0'},
            {'name':'GALEXT','value':(not args.no_extinction_correction),'comment':'Correct for Galactic extinction'},
            {'name':'WMASK','value':args.mask_file.split('/')[-1],'comment':'Path to observed wavelength mask'},
            {'name':'FCALIB','value':args.flux_calib.split('/')[-1],'comment':'Path to flux calibration'},
            {'name':'ICALIB','value':args.ivar_calib.split('/')[-1],'comment':'Path to ivar calibration'},
            ]
    dic = {}
    dic['TARGETID'] = sp.array([ t for t in data.keys() ])
    dic['Z'] = sp.array([ data[t]['Z'] for t in data.keys() ])

    tw = sp.argsort(dic['TARGETID'])
    for k in dic.keys():
        dic[k] = dic[k][tw]

    out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname='CAT')

    for ln, lv in lines.items():
        dic = {}
        head = [ {'name':'LINENAME','value':ln,'comment':'Line name'}]
        for eln, elv in lv.items():
            head += [{'name':eln,'value':elv}]

        for k in ['F_ON','F_OFF','F_LINE','NPIX_ON','NPIX_OFF']:
            dic[k] = sp.array([ data[t][ln][k] for t in data.keys() ])
        for k in dic.keys():
            dic[k] = dic[k][tw]

        out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname=ln)

    out.close()
