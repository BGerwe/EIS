import numpy as np


def calcS(Y):
    """Calculates sum of distances between pairs of admittance points

    Parameters
    ----------
    Y : np.ndarray
        Array of admittances

    Returns
    -------
    s :  float
        Array of distances between admittance point pairs
    """
    a = np.diff(Y)
    b = np.diff(np.conj(Y))

    s = np.real(np.sum(a*b))
    return s


def calcw(frequencies):
    """Calculates angular frequencies from an array of linear
    frequencies.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of linear frequencies

    Returns
    -------
    w : np.ndarray
        Array of angular frequencies

    """
    w = 2 * np.pi * frequencies
    return w


def calcLY(C, frequencies, Z):
    """Calculates the sum of squares of distances between admittance
    point pairs after subtracting a parallel capacitance.

    S = sum(Y_i-Y_i-1)(Y*_i-Y*_i-1)

    Where Y* indicates complex conjugate of Y

    Parameters
    ----------
    C : float or np.ndarray
        Scalar or array of capacitances to be subtracted.

    frequencies : np.ndarray
        Array of linear frequencies for corresponding impedance data

    Z : np.ndarray
        Array of impedance data

    Returns
    -------
    LY : floar or np.ndarray
        Scalar or array of root of sum of squares of distances between
        admittance points after subtracting parallel capacitance.
    """
    w = calcw(frequencies)
    Yel = 1/Z
    S = np.zeros(np.size(C))

    try:
        C.dtype
        print('Yes array')
        for i in range(np.size(C)):
            S[i] = calcS(Yel - C[i] * 1j * w)

    except AttributeError:
        print('Not array')
        S = calcS(Yel - C * 1j * w)

    LY = np.sqrt(S)
    return LY


def Par_Cap_Res(C, frequencies, Z):
    """Residual function for finding parallel capacitance
	"""
    w = calcw(frequencies)
    Yel = 1/Z
    S = calcS(Yel - C * 1j * w)
    LY = np.sqrt(S)
    return LY

def Par_CPE_Res(E, frequencies, Z):
    """Residual function for finding parallel constant phase element
	"""
    w = calcw(frequencies)
    Yel = 1/Z
    S = calcS(Yel - E[0] * (1j * w)**E[1])
    LY = np.sqrt(S)
    return LY

def Par_RC_Res(RC, frequencies, Z):
    """Residual function for finding parallel constant phase element
	"""
    w = calcw(frequencies)
    Yel = 1/Z
    S = calcS(Yel - 1 / RC[0] - RC[1] * (1j * w))
    LY = np.sqrt(S)
    return LY



def par_cap_subtract(C_sub, frequencies, Z):
    """Corrects impedance data for parallel capacitance.

    Parameters
    ----------
    C_sub : float
        Value of parallel capacitance subtracted from impedance data

    frequencies : np.ndarray
        Array of linear frequencies for corresponding impedance data

    Z : np.ndarray
        Array of impedance data

    Returns
    -------
    Z_corr : np.ndarray
        Impedance data corrected for parallel capacitance
    """
    Yel = 1/Z
    Y_corr = Yel - 1j * calcw(frequencies) * C_sub
    Z_corr = 1/Y_corr
    return Z_corr


def par_RC_subtract(RC_sub, frequencies, Z):
    """Corrects impedance data for parallel capacitance.

    Parameters
    ----------
    C_sub : float
        Value of parallel capacitance subtracted from impedance data

    frequencies : np.ndarray
        Array of linear frequencies for corresponding impedance data

    Z : np.ndarray
        Array of impedance data

    Returns
    -------
    Z_corr : np.ndarray
        Impedance data corrected for parallel capacitance
    """
    Yel = 1/Z
    Y_corr = Yel - 1 / RC_sub[0] - 1j * calcw(frequencies) * RC_sub[1]
    Z_corr = 1/Y_corr
    return Z_corr