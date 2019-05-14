# redvsblue
Quasar and emission line precise redshift fitting from prior

## Install
To install simply clone the repository, add to your PATH and
to your PYTHONPATH
```bash
git clone https://github.com/londumas/redvsblue
PYTHONPATH=$PYTHONPATH:<path_to_folder>/redvsblue/py/
PATH=$PATH:<path_to_folder>/redvsblue/bin/
```

## Run

To get all available options:
```bash
redvsblue_lineFitter.py --help
```

To run on SDSS data, do:
```bash
redvsblue_lineFitter.py
--out <path_to_write_output>.fits
--in-dir $BOSS_SPECTRO_REDUX/v5_13_0/
--drq DR12Q_v2_10.fits
--z-key Z_VI
--qso-pca <path_to_folder>/redvsblue/etc/rrtemplate-qso.fits
```

To run on DESI data, for example:

```bash
redvsblue_lineFitter.py
--out <path_to_write_output>.fits
--in-dir /project/projectdirs/desi/datachallenge/redwood/spectro/redux/redwood/spectra-64/
--drq /project/projectdirs/desi/datachallenge/redwood/spectro/redux/redwood/zcatalog-redwood-target-truth.fits
--z-key Z
--qso-pca <path_to_folder>/redvsblue/etc/rrtemplate-qso.fits
--data-format DESI
--no-extinction-correction
```

## Output

The output is a FITS file, with one HDU per redshift type:
    * CAT, HDU=1: redshift prior
    * PCA, HDU=2: PCA redshift
    * HALPHA, HDU=3: redshift of the HALPHA line
    * HBETA, HDU=4: redshift of the HBETA line
    * MGII, HDU=5: redshift of the MGII line
    * CIII, HDU=6: redshift of the CIII line
    * CIV, HDU=7: redshift of the CIV line
    * LYA, HDU=8: redshift of the LYA line

For each best fit readshift, the code gives the following quantities:
    * ZLINE: best fit redshift of the line according to maximum of PCA
    * ZPCA: best fit redshift according to PCA
    * ZERR: redshift error
    * ZWARN: redshift warning, use ZWARN=0 for reliable redshifts
    * CHI2: chi^2 of the best fit
    * DCHI2: Delta chi^2 against a Legendre polynomial, i.e. gives
    the significance of the redshift
    * NPIXBLUE: number of pixel on the blue side around the prior redshift
    * NPIXRED: number of pixel on the red side around the prior redshift
    * NPIX: total number of pixels around the prior redshift
    * NPIXBLUEBEST: number of pixel on the blue side around the best redshift
    * NPIXREDBEST: number of pixel on the red side around the best redshift
    * NPIXBEST: total number of pixels around the best redshift

## Why Red vs. Blue
RED vs BLUE, from Corridor: <https://www.youtube.com/watch?v=arg_aHzviQw>
