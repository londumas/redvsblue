
class ZWarningMask(object):
    SKY               = 2**0  #- sky fiber
    LITTLE_COVERAGE   = 2**1  #- too little wavelength coverage
    SMALL_DELTA_CHI2  = 2**2  #- chi-squared of best fit is too close to that of second best
    NEGATIVE_MODEL    = 2**3  #- synthetic spectrum is negative
    MANY_OUTLIERS     = 2**4  #- fraction of points more than 5 sigma away from best model is too large (>0.05)
    Z_FITLIMIT        = 2**5  #- chi-squared minimum at edge of the redshift fitting range
    NEGATIVE_EMISSION = 2**6  #- a QSO line exhibits negative emission, triggered only in QSO spectra, if  C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0
    UNPLUGGED         = 2**7  #- the fiber was unplugged/broken, so no spectrum obtained
    BAD_TARGET        = 2**8  #- catastrophically bad targeting data
    NODATA            = 2**9  #- No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)
    BAD_MINFIT        = 2**10 #- Bad parabola fit to the chi2 minimum
    NODATA_BLUE       = 2**11 #- No data on the blue side of the line
    NODATA_RED        = 2**12 #- No data on the blue side of the line

    from_bit_to_name = {
    0:'SKY',
    1:'LITTLE_COVERAGE',
    2:'SMALL_DELTA_CHI2',
    3:'NEGATIVE_MODEL',
    4:'MANY_OUTLIERS',
    5:'Z_FITLIMIT',
    6:'NEGATIVE_EMISSION',
    7:'UNPLUGGED',
    8:'BAD_TARGET',
    9:'NODATA',
    10:'BAD_MINFIT',
    11:'NODATA_BLUE',
    12:'NODATA_RED',
    }
