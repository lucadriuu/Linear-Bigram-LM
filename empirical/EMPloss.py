import numpy as np

def make_t_steps(t_min, t_max, num = 50):
    t_steps = np.unique(np.round(np.geomspace(t_min, t_max, num)).astype(int))      #GD steps are integers: (1 - eta*marg)**(2t) stays finite even if the base is negative
    return t_steps

def emp_loss(eta, marg, row_sq_sum, t = 1):

    t_dep = (1.0 - eta* marg)**(2*t)

    num_term1 = marg*t_dep

    num = num_term1 @ row_sq_sum
    den = marg @ row_sq_sum

    if den == 0:
        return 0.0

    emp_loss_t = num/den

    return emp_loss_t



def pop_loss(pi, u_j, beta, eta, t=1):
    

    base = 1.0 - eta*pi
    base = np.where(base < 1e-12, 0.0, base)   # eta = 1/pi_1 kills mode 1 exactly

    t_depend = base**(2*t)

    b_t =  pi @ t_depend                                     #Bach term

    num2_sum1 = ((pi * u_j) @ t_depend )
    num2_sum2 = (pi ** 2) @ u_j
    den2 = pi @ pi

    q_t = (num2_sum1 * num2_sum2) / den2

    pop_loss_t = b_t - (2*beta / (1 + (beta) ** 2)) * q_t


    return pop_loss_t



def emp_evo(t_steps, eta, marg, row_sq_sum):
    emp_dynamics = []

    for t in t_steps:
        emp_loss_t = emp_loss(eta, marg, row_sq_sum, t = t)
        emp_dynamics.append(emp_loss_t)

    return np.array(emp_dynamics)


def pop_evo(t_steps, eta, pi, u_j, beta):
    pop_dynamics = []

    for t in t_steps:
        pop_loss_t = pop_loss(pi, u_j, beta, eta, t=t)
        pop_dynamics.append(pop_loss_t)
    
    return np.array(pop_dynamics)