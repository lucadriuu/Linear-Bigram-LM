import numpy as np


def loss_terms(pi, u_j, eta, t):
    base = 1.0 - eta*pi
    base = np.where(base < 1e-12, 0.0, base)   # eta = 1/pi_1 kills mode 1 exactly
    t_depend = base**(2*t)

    b_t = pi @ t_depend                          # Bach term
    A_t = (pi * u_j) @ t_depend
    q_t = A_t * ((pi ** 2) @ u_j) / (pi @ pi)

    return b_t, q_t, A_t


def loss(pi, u_j, beta, eta, t=1):
    b_t, q_t, A_t = loss_terms(pi, u_j, eta, t)
    l_t = b_t - (2*beta / (1 + beta**2)) * q_t
    return l_t, b_t, q_t, A_t
