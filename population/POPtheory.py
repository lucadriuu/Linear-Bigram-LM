import numpy as np
from scipy.special import zeta
from scipy.special import beta as beta_fn      
import mpmath

# Reference curves of Thm 3.1, valid only when eta = 1/pi_1


def tau(alpha, d, t_steps):
    if alpha < 1.0:
        return 2*t_steps / (d**alpha)
    elif alpha == 1.0:
        return np.emath.logn(d, 2 * t_steps)
    else:
        return t_steps


def refline_t_grid(alpha, d, t_steps):
    # alpha = 1: the reference 1 - tau is defined for tau in [0, 1], i.e. t in [1/2, d/2]
    if alpha == 1.0:
        t_min = max(t_steps[0], 0.5)
        t_max = min(t_steps[-1], d / 2)
        return np.geomspace(t_min, t_max, 200)
    if alpha == 1.5:
        t_min = 1.0
        t_max = t_steps[-1]
        return np.geomspace(t_min, t_max, 200)
    return t_steps


def refline(alpha, d, t_steps):
    ref = []
    for tv in tau(alpha, d, t_steps):
        if alpha < 1.0:
            # exact E_{1/alpha}(tau) (Thm 3.1); e^{-tau}/(tau+1) is only its large-tau equivalent
            ref.append(((1 - alpha) / alpha) * float(mpmath.expint(1/alpha, tv)))
        elif alpha == 1.0:
            ref.append(max(1 - tv, 0.0))
        else:
            # Beta form of Thm 3.1; its large-t limit is C/(2t)^{1-1/alpha}, not C/t^{1-1/alpha}
            ref.append(beta_fn(1 - 1/alpha, 1 + 2*tv) / (alpha * zeta(alpha)))
    return np.array(ref)


def refline_label(alpha):
    if alpha < 1.0:
        return r'$\frac{1-\alpha}{\alpha} E_{1/\alpha}(\tau)$'
    elif alpha == 1.0:
        return r'$1 - \tau$'
    else:
        return r'$\mathrm{B}(1-1/\alpha,\, 1+2t)\,/\,\alpha\zeta(\alpha)$'


def t_star(alpha):
    # Theoretical transition point: tau = [-2 log(1 - 2^(-alpha))]^(-1)
    return 1.0 / (-2.0 * np.log(1.0 - 2.0 ** (-alpha)))
