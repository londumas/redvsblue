#!/usr/bin/env python
import fitsio
import scipy as sp
from scipy.interpolate import interp1d

from redvsblue import read_SDSS_data, constants

def main():

    ###
    nspec = None
    #path_DR12Q = '$HOME/Data/Catalogs/DR12Q_v2_10.fits'
    path_DR12Q = '$EBOSS_ROOT/qso/DR14Q/DR14Q_v3_1.fits'
    path_spplate = '$BOSS_SPECTRO_REDUX/v5_11_0/'
    mask_file = '$HOME/Run_programs/igmhub/picca_DR16_paper_analysis/dr16-line-sky-mask.txt'
    flux_calib_file = '$HOME/Run_programs/igmhub/picca_DR16_paper_analysis/Delta_calibration/Log/delta_attributes.fits.gz'
    ivar_calib_file = '$HOME/Run_programs/igmhub/picca_DR16_paper_analysis/Delta_calibration2/Log/delta_attributes.fits.gz'
    lambda_min = 3600.
    lambda_max = 7235.
    zkey = 'Z'

    ###
    lines = constants.lines
    zmin = 10.
    zmax = 0.
    for lv in lines.values():
        zmin = min(zmin,lambda_min/lv['RED_MAX']-1.)
        zmax = max(zmax,lambda_max/lv['BLUE_MIN']-1.)
    print('zmin = {}'.format(zmin))
    print('zmax = {}'.format(zmax))

    ### Read veto lines:
    veto_lines = read_SDSS_data.get_mask_lines(mask_file)

    ### Read flux calib
    h = fitsio.FITS(flux_calib_file)
    ll_st = h[1]['loglam'][:]
    st = h[1]['stack'][:]
    w = st!=0.
    flux_calib = interp1d(ll_st[w],st[w],fill_value="extrapolate",kind="linear")
    h.close()

    ### Read ivar calib
    h = fitsio.FITS(ivar_calib_file)
    ll = h[2]['LOGLAM'][:]
    eta = h[2]['ETA'][:]
    ivar_calib = interp1d(ll,eta,fill_value="extrapolate",kind="linear")
    h.close()

    ### Read spectra
    data = read_SDSS_data.read_SDSS_data(path_DR12Q=path_DR12Q, path_spec=path_spplate, lines=lines,
        zmin=zmin, zmax=zmax, zkey=zkey,
        lambda_min=lambda_min, lambda_max=lambda_max,
        veto_lines=veto_lines, flux_calib=flux_calib, ivar_calib=ivar_calib,
        nspec=nspec)

    ###
    for ln in lines.keys():
        thid = sp.array([ t for t in data.keys() ])
        blueVar = sp.array([ d[ln]['BLUE_VAR'] for d in data.values() ])
        redVar = sp.array([ d[ln]['RED_VAR'] for d in data.values() ])
        blueSNR = sp.array([ d[ln]['BLUE_SNR'] for d in data.values() ])

        sp.savetxt(ln+'_redvsblue.txt',sp.array(list(zip(thid,blueVar,redVar,blueSNR))),fmt='%u %e %e %e')

    return


main()
