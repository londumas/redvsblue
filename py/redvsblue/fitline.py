from functools import partial
import scipy as sp
import scipy.special

from redvsblue.utils import get_dz, transmission_Lyman
from redvsblue.zwarning import ZWarningMask as ZW
from redvsblue._zscan import _zchi2_one
from redvsblue.fitz import minfit, maxLine, find_minima

def fit_spec_redshift(z, lam, flux, weight, wflux, modelpca, legendre, zrange, line,
    qso_pca=None, dv_coarse=None, dv_fine=None, nb_zmin=3, dwave_model=0.1, correct_lya=False,
    no_slope=False):
    """

    """

    import matplotlib.pyplot as plt

    ### Coarse scan
    zcoeff = sp.zeros(modelpca.shape[2])
    p_zchi2_one = partial(_zchi2_one, weights=weight, flux=flux, wflux=wflux, zcoeff=zcoeff)
    chi2 = sp.array([ p_zchi2_one(el) for el in modelpca ])

    plt.title(line,fontsize=15)
    plt.plot(zrange,chi2,label=r'$Coarse\,scan$',color='black')

    ### Loop over different minima
    results = {}
    for iii,idxmin in enumerate(find_minima(chi2)[:nb_zmin]):

        zwarn = 0

        if (chi2==9e99).sum()>0:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (zrange[idxmin], -1., zwarn, chi2[idxmin])
            continue

        zPCA = zrange[idxmin]
        if (idxmin<=1) | (idxmin>=zrange.size-2):
            zwarn |= ZW.Z_FITLIMIT

        ### Fine scan
        Dz = get_dz(dv_coarse,zPCA)
        dz = get_dz(dv_fine,zPCA)
        tzrange = sp.linspace(zPCA-2.*Dz,zPCA+2.*Dz,1+int(round(4.*Dz/dz)))
        tmodelpca = sp.array([ sp.append( sp.array([ el(lam/(1.+tz)) for el in qso_pca ]).T,legendre,axis=1) for tz in tzrange ])
        if correct_lya:
            T = sp.array([ transmission_Lyman(tz,lam) for tz in tzrange ])
            for iii in range(tmodelpca.shape[-1]-legendre.shape[-1]):
                tmodelpca[:,:,iii] *= T
        tchi2 = sp.array([ p_zchi2_one(el) for el in tmodelpca ])
        tidxmin = 2+sp.argmin(tchi2[2:-2])
        p = plt.plot(tzrange,tchi2,label=r'$Local\,minimum \#'+str(iii)+'$')

        if (tchi2==9e99).sum()>0:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (tzrange[tidxmin], -1., zwarn, tchi2[tidxmin])
            continue

        ### Precise z_PCA
        tresult = minfit(tzrange[tidxmin-1:tidxmin+2],tchi2[tidxmin-1:tidxmin+2])
        if tresult is None:
            zwarn |= ZW.BAD_MINFIT
            results[idxmin] = (tzrange[tidxmin], -1., zwarn, tchi2[tidxmin])
        else:
            zPCA, zerr, fval, tzwarn = tresult
            zwarn |= tzwarn
            results[idxmin] = (zPCA, zerr, zwarn, fval)
        plt.plot([zPCA,zPCA],[tchi2.min(),chi2.max()],color=p[0].get_color())

    idx_min = sp.array([ k for k in results.keys() ])[sp.argmin([ v[3] for v in results.values() ])]
    zPCA, zerr, zwarn, fval = results[idx_min]
    plt.plot([zPCA,zPCA],[fval,chi2.max()],label=r'$ZPCA$')

    plt.subplots_adjust(top=0.90,right=0.95,left=0.15)
    plt.xlabel(r'$z$',fontsize=15)
    plt.ylabel(r'$\chi^{2}$',fontsize=15)
    plt.legend(fontsize=10)
    plt.grid()
    plt.show()

    ### Observed wavelength of maximum of line
    if True:#line!='PCA':

        import matplotlib.pyplot as plt
        plt.title(line+': zpca = {}'.format(round(zPCA,3)),fontsize=15)
        plt.plot(lam/10., flux, color='black', label=r'$\mathrm{Data}$')

        ### Get coefficient of the model
        model = sp.array([ el(lam/(1.+zPCA)) for el in qso_pca ]).T
        if correct_lya:
            T = transmission_Lyman(zPCA,lam)
            for iii in range(model.shape[-1]):
                model[:,iii] *= T
        model = sp.append( model,legendre,axis=1)
        p_zchi2_one(model)

        ### Get finer model
        tlam = sp.arange(lam.min(), lam.max(), dwave_model)
        tlegendre = sp.array([scipy.special.legendre(i)( (tlam-tlam.min())/(tlam.max()-tlam.min())*2.-1. ) for i in range(legendre.shape[1])]).T
        model = sp.array([ el(tlam/(1.+zPCA)) for el in qso_pca ]).T
        if correct_lya:
            T = transmission_Lyman(zPCA,tlam)
            plt.plot(tlam/10., T, label=r'$\mathrm{Lya\,transmission}$')
            for iii in range(model.shape[-1]):
                model[:,iii] *= T
        model = sp.append( model,tlegendre,axis=1)
        model = model.dot(zcoeff)
        plt.plot(tlam/10., model, label=r'$\mathrm{Best-fit\,model}$')

        if no_slope:
            zcoeff = sp.zeros(2)
            slope = legendre[:,:2]
            zchi2 = _zchi2_one(slope, weight, flux, wflux, zcoeff)
            slope = sp.array([scipy.special.legendre(i)( (tlam-tlam.min())/(tlam.max()-tlam.min())*2.-1. ) for i in range(2)]).T
            slope = slope.dot(zcoeff)
            model -= slope

        ### Find min
        idxmin = sp.argmax(model)
        if (idxmin<=1) | (idxmin>=model.size-2):
            zwarn |= ZW.Z_FITLIMIT

        tresult = maxLine(tlam[idxmin-1:idxmin+2],model[idxmin-1:idxmin+2])
        if tresult is None:
            zwarn |= ZW.BAD_MINFIT
            lLine = tlam[idxmin]
        else:
            zwarn |= tresult[3]
            lLine = tresult[0]

        if line!='PCA':
            from redvsblue.constants import emissionLines
            tttt_line = (zPCA+1.)*emissionLines[line]
            plt.plot([tttt_line/10.,tttt_line/10.],[flux.min(),flux.max()],label=r'$ZPCA: z = '+str(round(zPCA,3))+'$')
            plt.plot([lLine/10.,lLine/10.],[flux.min(),flux.max()],label=r'$ZLINE: z = '+str(round(lLine/emissionLines[line]-1.,3))+'$')

    else:
        lLine = -1.

    ### No peak fit
    zcoeff = sp.zeros(legendre.shape[1])
    zchi2 = _zchi2_one(legendre, weight, flux, wflux, zcoeff)
    deltachi2 = zchi2
    model = legendre.dot(zcoeff)
    plt.plot(lam/10., model, label=r'$\mathrm{Best-fit\,broadband}$')

    plt.subplots_adjust(top=0.90,right=0.95)
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\mathrm{nm}]$',fontsize=15)
    plt.ylabel(r'$flux$',fontsize=15)
    plt.legend(fontsize=10)
    plt.grid()
    plt.show()

    return lLine, zPCA, zerr, zwarn, fval, deltachi2
