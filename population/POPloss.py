import numpy as np
import scipy 
from scipy.special import gamma, zeta, expn

#prova git prova git 

def token_freq(d,alpha):
        i = np.arange(1,d + 1)
        pi = i**(-alpha) / np.sum(i**(- alpha))
        return pi

def u_jn(v_size):
        u_j = np.random.choice([1, -1], size=v_size)
        return u_j

def u_j_greedy(d, pi):
        u_jg = np.zeros(d)
        u_jg[0] = 1
        orto_sum = pi[0]

        for i in range(1, len(pi) ):
            if  orto_sum > 0:
                u_jg[i] = -1
            else:
                u_jg[i] = 1
            orto_sum = orto_sum + u_jg[i] * pi[i]

        return u_jg

def u_j_flip_mp(d, pi, n_passes):
        rng = np.random.default_rng()
        u_jf = rng.choice([1, -1], size=d)

        cs_list = []
        current_sum = pi @ u_jf 
        cs_list.append(current_sum)
        
        for p in range(n_passes):
            
            if p == 0:
                shuffled_indices = np.arange(d)
            else:
                shuffled_indices = rng.permutation(d)

            for k in shuffled_indices:

                old_val = u_jf[k] 

                u_jf[k] = - old_val
                new_sum = current_sum + pi[k] * (u_jf[k] - old_val)

                if abs(new_sum) < abs(current_sum):
                    current_sum = new_sum
                else:
                    u_jf[k] = old_val
            current_sum = pi @ u_jf
            cs_list.append(current_sum)

        return u_jf


def loss(pi, u_j, beta, eta, t=1):
    

    base = 1.0 - eta*pi
    base = np.where(base < 1e-12, 0.0, base)   # eta = 1/pi_1 kills mode 1 exactly
    t_depend = base**(2*t)  

    t_depend = base**(2*t)

    b_t =  pi @ t_depend                                     #Bach term

    num2_sum1 = ((pi * u_j) @ t_depend )
    num2_sum2 = (pi ** 2) @ u_j
    den2 = pi @ pi

    q_t = (num2_sum1 * num2_sum2) / den2

    result = b_t - (2*beta / (1 + (beta) ** 2)) * q_t


    return result, b_t, q_t, num2_sum1



def evo_loss(d, alpha, pi, u_j, beta, eta, t_start, t_stop):
    dynamics = []
    dynamics_q = []
    dynamics_b = []
    dynamics_num2_sum1 = []
    refline = []
    t_steps = np.geomspace(t_start, t_stop) 
    
    if alpha < 1.0:
        #d = 1e7
        tau = 2*t_steps / (d**alpha)

    if alpha == 1.0:
        #d = 1e7
        tau = np.emath.logn(d, 2 * t_steps )
        
    if alpha > 1.0:
        tau = t_steps


    for t, tv in zip(t_steps,tau):
        loss_t, b_t, q_t, num2_sum1 = loss(pi, u_j, beta, eta, t=t)
        dynamics.append(loss_t)
        dynamics_q.append(q_t)
        dynamics_b.append(b_t)
        dynamics_num2_sum1.append(num2_sum1)

        if alpha < 1.0:
            #ref = ((1-alpha)/alpha) * expn(1/alpha, tv)
            ref = ((1 - alpha) / alpha) * (np.exp(-tv) / (tv + 1))
            refline.append(ref)

        if alpha == 1.0:
            ref = max(1 - tv, 0.0)
            #ref = 1 - tv
            refline.append(ref)

        if alpha > 1.0:
            ref = (gamma(1-(1/alpha))/(alpha*zeta(alpha)))/(tv**(1-(1/alpha)))      
            refline.append(ref)

    return t_steps, dynamics, dynamics_b, dynamics_q, dynamics_num2_sum1, refline





def evo_loss_multi(d, alpha, pi, betas, eta, t_start, t_stop, n_runs, n_passes):

    u_realizations = [u_j_flip_mp(d, pi, n_passes) for _ in range(n_runs)]   # sampled once, reused for every beta
    
    #u_real_plus = []
    #for u in u_realizations:
    #     if u @ pi > 0 and u[0] == 1:
    #          u_real_plus.append(u)

    mean_dyn = []
    std_dyn = []
    mean_dyn_q = []
    std_dyn_q = []

    for beta in betas:
        runs_dynamics = []
        runs_dynamics_q = []

        for u_jf in u_realizations:                 #select just the relizations with u@\pi > 0
            t_steps, dynamics, _, dynamics_q, _, refline = evo_loss(d, alpha, pi, u_jf, beta, eta, t_start, t_stop)
            runs_dynamics.append(dynamics)
            runs_dynamics_q.append(dynamics_q)

        runs_dynamics = np.array(runs_dynamics)
        mean_dyn.append(np.mean(runs_dynamics, axis=0))
        std_dyn.append(np.std(runs_dynamics, axis=0))

        runs_dynamics_q = np.array(runs_dynamics_q)
        mean_dyn_q.append(np.mean(runs_dynamics_q, axis=0))
        std_dyn_q.append(np.std(runs_dynamics_q, axis=0))

    return t_steps, np.array(mean_dyn), np.array(std_dyn), np.array(mean_dyn_q), np.array(std_dyn_q), refline
