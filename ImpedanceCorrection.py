import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits.elements import G
from impedance.visualization import plot_nyquist


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
    Z_corr = 1/(Yel - 10**E[0] * (1j * w)**E[1])
    S2 = sum(n > 0 for n in Z_corr.imag)
    LY = np.sqrt(S)
    return S + S2


def Par_CPE_Res_log(E, frequencies, Z):
    """Residual function for finding parallel constant phase element
    """
    w = calcw(frequencies)
    Yel = 1/Z
    S = calcS(Yel - 10**E[0] * (1j * w)**E[1])
    Z_corr = 1/(Yel - 10**E[0] * (1j * w)**E[1])
    S2 = calcS(Z_corr)
    S2 = sum(n > 0 for n in Z_corr.imag)

    return S


def Par_RC_Res(RC, frequencies, Z):
    """Residual function for finding parallel RC circuit
    """
    w = calcw(frequencies)
    Yel = 1/Z
    # print(1/RC[0], RC[1] * (1j * w))
    Y_adj = Yel - 1 / RC[0] - RC[1] * (1j * w)
    S = calcS(Y_adj)
    LY = np.sqrt(S)    # + np.abs(np.sum(np.imag(Z))) #+ np.sum(np.angle(Yel))
    return LY


def Par_RC_Res_log(RC, frequencies, Z):
    """Residual function for finding parallel RC circuit
    """
    w = calcw(frequencies)
    Yel = 1/Z
    # print(1/RC[0], RC[1] * (1j * w))
    Y_adj = Yel - 1 / 10 ** RC[0] - 10 ** RC[1] * (1j * w)
    Z_adj = 1 / Y_adj
    S = calcS(Y_adj) + np.sum(Z_adj.imag)
    LY = np.sqrt(S)   # + np.sum(np.angle(Yel))
    return LY


def Par_RCPE_Res(RCPE, frequencies, Z):
    """Residual function for finding parallel RC circuit
    """
    w = calcw(frequencies)
    Yel = 1/Z
    # print(1/RC[0], RC[1] * (1j * w))
    Y_adj = Yel - 1 / RCPE[0] - RCPE[1] * (1j * w) ** RCPE[2]
    Z_adj = 1 / Y_adj

    S = calcS(Y_adj) + np.sqrt(np.sum(Z_adj[Z_adj.imag > 0].imag))
    #LY = np.sqrt(S)   # + np.sum(np.angle(Yel))
    return S


def Par_Zg_Res(p, f, Z, tg):
    from impedance.models.circuits.elements import G
    Y = 1 / Z
#     p = [Rg[0], tg]
    try:
        Rg = p[0]
        tg = p[1]
    except TypeError:
        print("Rg passed is not iterable. Using as float")
        Rg = p
    except IndexError:
        Rg = p[0]
    
    Zg = G([Rg, tg], f)
    Yg = 1 / Zg
    Y_adj = Y - Yg
    Z_adj = 1 / Y_adj
    diffed = np.diff(Z_adj.real)
    min_ind = np.argmin(Z_adj.imag)
    
    Z_adj = Z_adj - Z_adj[min_ind].real
    curv = np.gradient(np.gradient(-Z_adj.imag, Z_adj.real), Z_adj.real)

    xdum = np.zeros(np.shape(curv), dtype=bool)
    xdum[-10:] = curv[-10:] > 0
    curv_res = sum(curv[xdum])

    res = 0
    for n in range(1, len(Z_adj)-min_ind):
        res += (Z_adj[min_ind+n].imag - Z_adj[min_ind-n].imag)**2
        res += (Z_adj[min_ind+n].real + Z_adj[min_ind-n].real)**2
    return res #+ sum(diffed[diffed < 0]**2)*1000 + curv_res*5000 #  + dum*1000


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
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - 1j * w * C_sub
    Z_corr = 1/Y_corr
    return Z_corr


def par_CPE_subtract(CPE_sub, frequencies, Z):
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
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - (CPE_sub[0] * (1j * w)**CPE_sub[1])
    Z_corr = 1/Y_corr
    return Z_corr


def par_CPE_subtract_log(CPE_sub, frequencies, Z):
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
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - (10**CPE_sub[0] * (1j * w)**CPE_sub[1])
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
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - 1 / RC_sub[0] - 1j * w * RC_sub[1]
    Z_corr = 1/Y_corr
    return Z_corr


def par_RC_subtract_log(RC_sub, frequencies, Z):
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
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - 1 / (10 ** RC_sub[0]) - 1j * w * (10 ** RC_sub[1])
    Z_corr = 1/Y_corr
    return Z_corr


def par_RCPE_subtract(RCPE_sub, frequencies, Z):
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
    p = RCPE_sub
    w = calcw(frequencies)
    Yel = 1/Z
    Y_corr = Yel - 1 / (p[0]) - p[1] * (1j * w) ** p[2]
    Z_corr = 1/Y_corr
    return Z_corr


def sub_Zg_parallel(f, Z, tg, Rg_range, num, show_plot=True):

    Y = 1/Z

    Zgs, Ygs = [], []
    Z_adjs = []
    
    if len(Rg_range) > 1:
        Rgs = np.logspace(Rg_range[0], Rg_range[1], num=num)
    else:
        Rgs = Rg_range

    for Rg in Rgs:
        Zg = G([Rg, tg], f)
        Yg = 1 / Zg
        Zgs.append(Zg)
        Ygs.append(1/Zg)

    Zgs = np.array(Zgs)
    Ygs = np.array(Ygs)

    if show_plot is False:
        for i, Yg in enumerate(Ygs):
            Y_adj = Y - Yg
            Z_adj = 1 / Y_adj
            Z_adjs.append(Z_adj)

        return np.array(Z_adjs)
 
    else:
        fig, ax = plt.subplots(figsize=(18, 12))
        fig2, (ax1, ax2) = plt.subplots(nrows=2, figsize=(30, 20))
        plot_nyquist(ax, Z, label='Initial Data')

        for i, Yg in enumerate(Ygs):
            Y_adj = Y - Yg
            Z_adj = 1 / Y_adj
            Z_adjs.append(Z_adj)

            f_p_idx = np.argmin(np.imag(Z_adj))
            f_p = f[f_p_idx]

            ax.plot(Z_adj.real, -Z_adj.imag, '.-', label='#: %i %.2f Hz' % (i, f_p),
                    c=(0, i/len(Ygs), .4))
            ax.plot(Z_adj[f_p_idx].real, -Z_adj[f_p_idx].imag, 's',
                    c=(0, 0, 0))
            ax1.plot(np.log10(f), Z_adj.real, label='%.2f' % Rgs[i],
                     c=(0, i/len(Ygs), .4))
            ax2.plot(np.log10(f), Z_adj.imag, label='%.2f' % Rgs[i],
                     c=(0, i/len(Ygs), .4))

        # ax.set_xlim(-80, 300)
        # ax.set_ylim(-80, 300)
        ax1.grid(True)
        ax2.grid(True)
        ax.legend()

        return np.array(Z_adjs), [ax, ax1, ax2]


def sub_Zg_series(f, Z, tg, Rg_range, num, show_plot=True):

    Zgs = []
    Z_adjs = []

    Rgs = np.logspace(Rg_range[0], Rg_range[1], num=num)

    for Rg in Rgs:
        Zg = G([Rg, tg], f)
        Zgs.append(Zg)

    Zgs = np.array(Zgs)

    fig, ax = plt.subplots(figsize=(18, 12))
    fig2, (ax1, ax2) = plt.subplots(nrows=2, figsize=(30, 20))
    plot_nyquist(ax, Z)

    for i, Zg in enumerate(Zgs):
        Z_adj = Z - Zg
        Z_adjs.append(Z_adj)

        f_p_idx = np.argmin(np.imag(Z_adj))
        f_p = f[f_p_idx]

        ax.plot(Z_adj.real, -Z_adj.imag, label='%.2f Hz' % f_p,
                c=(0, i/len(Zgs), .4))
        ax.plot(Z_adj[f_p_idx].real, -Z_adj[f_p_idx].imag, 's', c=(0, 0, 0))
        ax1.plot(np.log10(f), Z_adj.real, label='%.2f' % Rgs[i],
                 c=(0, i/len(Zgs), .4))
        ax2.plot(np.log10(f), Z_adj.imag, label='%.2f' % Rgs[i],
                 c=(0, i/len(Zgs), .4))

    # ax.set_xlim(-80, 300)
    # ax.set_ylim(-80, 300)
    ax1.grid(True)
    ax2.grid(True)
    ax.legend()
    plt.show()

    return Z_adjs, ax, ax1, ax2


def detect_peak_f(f, Z):
    min_ind = np.argmin(Z.imag)
    fp = f[min_ind]
    return fp
