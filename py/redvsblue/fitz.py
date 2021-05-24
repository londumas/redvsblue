import scipy as sp

from .zwarning import ZWarningMask as ZW

def find_minima(x):

    ii = sp.where(sp.r_[True, x[1:]<=x[:-1]] & sp.r_[x[:-1]<=x[1:], True])[0]
    jj = sp.argsort(x[ii])

    return ii[jj]
def minfit(x, y):
    """

    """

    if x.size<3:
        return None

    try:
        #- y = a x^2 + b x + c
        a,b,c = sp.polyfit(x,y,2)
    except sp.linalg.LinAlgError:
        return None

    if a==0.:
        return None

    #- recast as y = y0 + ((x-x0)/xerr)^2
    x0 = -b/(2.*a)
    y0 = -(b**2)/(4.*a)+c

    zwarn = 0
    if x0<=x.min() or x0>=x.max():
        return None
    if y0<=0.:
        return None

    if a>0.:
        xerr = 1./sp.sqrt(a)
    else:
        xerr = 1./sp.sqrt(-a)
        zwarn |= ZW.BAD_MINFIT

    return x0, xerr, y0, zwarn
def maxLine(x, y):
    """

    """

    if x.size<3:
        return None

    try:
        #- y = a x^2 + b x + c
        a,b,c = sp.polyfit(x,y,2)
    except sp.linalg.LinAlgError:
        return None

    if a==0.:
        return None

    #- recast as y = y0 + ((x-x0)/xerr)^2
    x0 = -b/(2.*a)
    y0 = -(b**2)/(4.*a)+c

    zwarn = 0
    if x0<=x.min() or x0>=x.max():
        return None
    if y0<=0.:
        return None

    if a>0.:
        xerr = 1./sp.sqrt(a)
        zwarn |= ZW.BAD_MINFIT
    else:
        xerr = 1./sp.sqrt(-a)

    return x0, xerr, y0, zwarn
