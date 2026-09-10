import numpy as np
import pickle
from POPloss import evo_loss, token_freq, u_j_greedy, u_jn, u_j_flip_mp


def evoloss_beta_fixed(alphas, ds, beta, n_passes, method):  
    results = {
        'alphas': alphas,
        'ds': ds,
        'beta': beta,
        'method': method,
        'simulations': {}
    }

    for alpha in alphas:
        results['simulations'][alpha] = {}
        if alpha < 1.0:
            t_start, t_stop = 1e-3 , 1e5
        if alpha == 1.0:
            t_start, t_stop = 1e-3 , 1e7
        if alpha > 1.0:
            t_start, t_stop = 1 , 1e7
        
        for i, d in enumerate(ds):

            pi = token_freq(d, alpha)
            eta = 1/pi[0]

            if method == 'greedy':
                u_j = u_j_greedy(d, pi)
            if method == 'random':
                u_j = u_jn(d)
            if method == 'flip':
                u_j = u_j_flip_mp(d, pi, n_passes)
            
            t_steps, dynamics, dynamics_b, dynamics_q, _, refline = evo_loss(d, alpha, pi, u_j, beta, eta, t_start, t_stop)

            results['simulations'][alpha][d] = {
                't_steps': np.array(t_steps),
                'dynamics': np.array(dynamics),
                'dynamics_b': np.array(dynamics_b),
                'dynamics_q': np.array(dynamics_q),
                'refline': np.array(refline)
            }

    filename =  fr'evo_fixed_beta_{method}.pkl'
    filepath = fr'/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/population/data/evo_fixed_beta_{method}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    return filepath




def evoloss_d_fixed(alphas, d, betas, n_passes, method):  
    results = {
        'alphas': alphas,
        'd': d,
        'betas': betas,
        'method': method,
        'simulations': {}
    }

    for alpha in alphas:
        results['simulations'][alpha] = {}
        if alpha < 1.0:
            t_start, t_stop = 1e-2 , 1e3
        if alpha == 1.0:
            t_start, t_stop = 0.5 , d/2
        if alpha > 1.0:
            t_start, t_stop = 1 , 1e3

        # Computed once for the given vocabulary size d
        pi = token_freq(d, alpha)
        eta = 1 / pi[0]

        if method == 'greedy':
            u_j = u_j_greedy(d, pi)
        if method == 'random':
            u_j = u_jn(d)
        if method == 'flip':
            u_j = u_j_flip_mp(d, pi, n_passes)

        # Loop over beta values instead of vocabulary sizes
        for beta in betas:
            t_steps, dynamics, dynamics_b, dynamics_q, _, refline = evo_loss(
                d, alpha, pi, u_j, beta, eta, t_start, t_stop
            )

            results['simulations'][alpha][beta] = {
                't_steps': np.array(t_steps),
                'dynamics': np.array(dynamics),
                'dynamics_b': np.array(dynamics_b),
                'dynamics_q': np.array(dynamics_q),
                'refline': np.array(refline)
            }

    filename = fr'evo_variable_beta_{method}.pkl'
    filepath = fr'/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/population/data/evo_variable_beta_{method}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    return filepath