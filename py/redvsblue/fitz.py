import scipy as sp
from redvsblue.zwarning import ZWarningMask as ZW

def minfit(x, y):
    """

    """

    idxmin = sp.argmin(y)
    x0Min = x[idxmin]
    y0Min = y[idxmin]

    if x.size<3:
        return (x0Min,-1,y0Min,ZW.BAD_MINFIT)

    try:
        #- y = a x^2 + b x + c
        a,b,c = sp.polyfit(x,y,2)
    except sp.linalg.LinAlgError:
        return (x0Min,-1,y0Min,ZW.BAD_MINFIT)

    if a==0.:
        return (x0Min,-1,y0Min,ZW.BAD_MINFIT)

    #- recast as y = y0 + ((x-x0)/xerr)^2
    x0 = -b/(2.*a)
    y0 = -(b**2)/(4.*a)+c

    zwarn = 0
    if x0<=x.min() or x0>=x.max():
        zwarn |= ZW.BAD_MINFIT
    if y0<=0.:
        zwarn |= ZW.BAD_MINFIT

    if a>0.:
        xerr = 1./sp.sqrt(a)
    else:
        xerr = 1./sp.sqrt(-a)
        zwarn |= ZW.BAD_MINFIT

    return x0, xerr, y0, zwarn
