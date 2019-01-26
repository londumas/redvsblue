#!/usr/bin/env python
import argparse
import fitsio
import scipy as sp
from scipy.interpolate import interp1d

from redvsblue import read_SDSS_data, constants, utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the redshift of different quasar emmision lines')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to spectra files')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--z-key', type=str, default='Z', required=False,
        help='Name of the key giving redshifts in drq')

    parser.add_argument('--lambda-min',type=float,default=None,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=None,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--dwave-side',type=int,default=100,required=False,
        help='Wavelength interval on both side of the line [Angstrom]')

    parser.add_argument('--dv-prior',type=float,default=20000.,required=False,
        help='Velocity difference box prior [km/s]')

    parser.add_argument('--qso-pca',type=str,default=None,required=True,
        help='Path to quasar PCA')

    parser.add_argument('--mask-file',type=str,default=None,required=False,
        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')

    args = parser.parse_args()

    ###
    lines = constants.emissionLines

    ###
    qso_pca = utils.read_PCA(args.qso_pca,dim=True)
    if not args.flux_calib is None:
        args.flux_calib = utils.read_flux_calibration(args.flux_calib)
    if not args.ivar_calib is None:
        args.ivar_calib = utils.read_ivar_calibration(args.ivar_calib)
    if not args.mask_file is None:
        args.mask_file = utils.read_mask_lines(args.mask_file)

    ### Read spectra
    data = read_SDSS_data.fit_line(DRQ=args.drq, path_spec=args.in_dir, lines=lines, qso_pca=qso_pca,dv_prior=args.dv_prior,
        zkey=args.z_key, lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        veto_lines=args.mask_file, flux_calib=args.flux_calib, ivar_calib=args.ivar_calib,
        nspec=args.nspec, dwave_side=args.dwave_side)

    ### Save
    out = fitsio.FITS(args.out,'rw',clobber=True)

    head = [ {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'},
            {'name':'DWAVE','value':args.dwave_side,'comment':'Wavelength interval on both side of the line [Angstrom]'},
            {'name':'DVPRIOR','value':args.dv_prior,'comment':'Velocity difference box prior [km/s]'},]
    dic = {}
    dic['TARGETID'] = sp.array([ t for t in data.keys() ])
    dic['ZPRIOR'] = sp.array([ data[t]['ZPRIOR'] for t in data.keys() ])
    out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname='CAT')

    for ln, lv in lines.items():
        dic = {}
        head = [ {'name':'LINENAME','value':ln,'comment':'Line name'},
                {'name':'LINERF','value':lv,'comment':'Line rest frame [Angstrom]'}]
        for k in ['Z','ZERR','ZWARN','CHI2','NPIXBLUE','NPIXRED','NPIX','DCHI2']:
            dic[k] = sp.array([ data[t][ln][k] for t in data.keys() ])
        out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname=ln)

    out.close()
