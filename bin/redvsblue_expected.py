#!/usr/bin/env python
import argparse
import fitsio
import scipy as sp
from scipy.interpolate import interp1d

from redvsblue import read_SDSS_data, constants


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

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=7235.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

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
    lines = constants.lines
    zmin = 10.
    zmax = 0.
    for lv in lines.values():
        zmin = min(zmin,args.lambda_min/lv['RED_MAX']-1.)
        zmax = max(zmax,args.lambda_max/lv['BLUE_MIN']-1.)
    print('zmin = {}'.format(zmin))
    print('zmax = {}'.format(zmax))

    ### Read veto lines:
    veto_lines = read_SDSS_data.get_mask_lines(args.mask_file)

    ### Read flux calib
    h = fitsio.FITS(args.flux_calib)
    ll_st = h[1]['loglam'][:]
    st = h[1]['stack'][:]
    w = st!=0.
    flux_calib = interp1d(ll_st[w],st[w],fill_value="extrapolate",kind="linear")
    h.close()

    ### Read ivar calib
    h = fitsio.FITS(args.ivar_calib)
    ll = h[2]['LOGLAM'][:]
    eta = h[2]['ETA'][:]
    ivar_calib = interp1d(ll,eta,fill_value="extrapolate",kind="linear")
    h.close()

    ### Read spectra
    data = read_SDSS_data.read_SDSS_data(DRQ=args.drq, path_spec=args.in_dir, lines=lines,
        zmin=zmin, zmax=zmax, zkey=args.z_key,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,
        nspec=args.nspec)

    ###
    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'ZKEY','value':args.z_key,'comment':'Fitsio key for redshift'} ]
    dic = {}
    dic['THING_ID'] = sp.array([ t for t in data.keys() ])
    for ln in lines.keys():
        for side in ['BLUE','RED']:
            dic[ln+'_'+side+'_VAR'] = sp.array([ data[t][ln][side+'_VAR'] for t in data.keys() ])
            dic[ln+'_'+side+'_SNR'] = sp.array([ data[t][ln][side+'_SNR'] for t in data.keys() ])
    out.write([v for v in dic.values()],names=[k for k in dic.keys()],header=head,extname='CAT')
    out.close()