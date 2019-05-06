```
quickquasars -i /project/projectdirs/desi/mocks/lya_forest/london/v6.0/v6.0.0/0/0/transmission-16-0.fits --outdir test-0-0/spectra-16/ --zbest --bbflux --eboss --seed 123 --sigma_kms_fog 0. --zmin 1.8 --metals LYB --nmax 1
desi_zcatalog -i test-0-0/spectra-16/ -o test-0-0/zcat.fits --fibermap
```
