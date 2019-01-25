import scipy as sp

def weighted_var(values, weights):
    """
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """

    m = sp.average(values, weights=weights)
    var = sp.average((values-m)**2, weights=weights)

    return var
