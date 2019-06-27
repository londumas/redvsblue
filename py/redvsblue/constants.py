# Lyman-alpha from eqn 5 of Calura et al. 2012 (Arxiv: 1201.5121)
# Other from eqn 1.1 of Irsic et al. 2013 , (Arxiv: 1307.3403)
# Lyman-limit from abstract of Worseck et al. 2014 (Arxiv: 1402.4154)
Lyman_series = {
    'LYA'     : { 'line':1215.67,  'A':0.0023,          'B':3.64, 'var_evol':3.8 },
    'LYB'     : { 'line':1025.72,  'A':0.0023/5.2615,   'B':3.64, 'var_evol':3.8 },
    'LY3'     : { 'line':972.537,  'A':0.0023/14.356,   'B':3.64, 'var_evol':3.8 },
    'LY4'     : { 'line':949.7431, 'A':0.0023/29.85984, 'B':3.64, 'var_evol':3.8 },
    'LY5'     : { 'line':937.8035, 'A':0.0023/53.36202, 'B':3.64, 'var_evol':3.8 },
    #'LYLIMIT' : { 'line':911.8,    'A':None,            'B':None, 'var_evol':None },
}
lines = {
    'MGII' : {'LINE':2796.3511, 'BLUE_MIN':2600., 'BLUE_MAX':2760., 'RED_MIN':2900. , 'RED_MAX':3120.},
    'CIV'  : {'LINE':1548.2049, 'BLUE_MIN':1420., 'BLUE_MAX':1520., 'RED_MIN':1600.,  'RED_MAX':1700. },
    'LYA'  : {'LINE':1215.67,   'BLUE_MIN':1040., 'BLUE_MAX':1200., 'RED_MIN':1260.,  'RED_MAX':1375. },
}
emissionLines = {
    'PCA'    : 0.,
    'HALPHA' : 6562.8,
    'HBETA'  : 4862.68,
    'MGII'   : 2799.942,
    'CIII'   : 1908.734,
    'CIV'    : 1549.492,
    'LYA'    : 1215.67,
}
