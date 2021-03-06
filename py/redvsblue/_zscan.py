import scipy as sp
import warnings
warnings.filterwarnings('error')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def _zchi2_one(Tb, weights, flux, wflux, zcoeff):
    """

    """

    M = Tb.T.dot(sp.multiply(weights[:,None], Tb))
    y = Tb.T.dot(wflux)

    try:
        zcoeff[:] = sp.linalg.solve(M, y)
    except (sp.linalg.LinAlgError, sp.linalg.LinAlgWarning):
        return 9e99

    model = Tb.dot(zcoeff)
    zchi2 = sp.dot( (flux - model)**2, weights )

    return zchi2
