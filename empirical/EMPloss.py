import numpy as np


def loss(eta, marg, row_sq_sum, t = 1):

    t_dep = (1.0 - eta* marg)**(2*t)

    num_term1 = marg*t_dep

    num = num_term1 @ row_sq_sum
    den = marg @ row_sq_sum

    if den == 0:
        return 0.0

    loss_t = num/den


    return loss_t


def evo(t_min, t_max, seq, eta, d, jump_probs):
    dynamics = []
    t_steps = np.geomspace(t_min, t_max) 
    marg = np.bincount(seq, minlength = d) / len(seq)
    row_sq_sum = (jump_probs**2).sum(axis = 1)

    for t in t_steps:
        loss_t = loss(eta, marg, row_sq_sum, t = t)
        dynamics.append(loss_t)

    return dynamics, t_steps


