# redvsblue
Quasar and emission line precise redshift fitting from prior

## Install
To install simply clone the repository, add to your PATH and
to your PYTHONPATH
```
git clone https://github.com/londumas/redvsblue
PYTHONPATH=$PYTHONPATH:<path_to_folder>/redvsblue/py/
PATH=$PATH:<path_to_folder>/redvsblue/bin/
```

## Run

To get all available options:
```
redvsblue_lineFitter.py --help
```

To run on SDSS data, do:
```
redvsblue_lineFitter.py
--out <path_to_write_output>.fits
--in-dir $BOSS_SPECTRO_REDUX/v5_13_0/
--drq DR12Q_v2_10.fits
--z-key Z_VI
--qso-pca <path_to_folder>/redvsblue/etc/rrtemplate-qso.fits
```

To run on DESI data, for example:

```
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
 - HDU=1: redshift prior
 - HDU=2: PCA redshift
 - HDU=3: redshift of the HALPHA line
 - HDU=4: redshift of the HBETA line
 - HDU=5: redshift of the MGII line
 - HDU=6: redshift of the CIII line
 - HDU=7: redshift of the CIV line
 - HDU=8: redshift of the LYA line

## Why Red vs. Blue
RED vs BLUE, from Corridor: <https://www.youtube.com/watch?v=arg_aHzviQw>
