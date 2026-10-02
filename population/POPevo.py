import numpy as np
import pickle
from POPdata import token_freq, build_u
from POPloss import loss_terms


def run_simulation(cfg):
    method = cfg['method']
    n_t = cfg['n_t']
    n_runs = cfg['n_runs']
    if method == 'greedy':
        n_runs = 1                                   # greedy is deterministic

    rng = np.random.default_rng(cfg['seed'])

    results = {'config': cfg, 'simulations': {}}

    for alpha in cfg['alphas']:
        results['simulations'][alpha] = {}

        for d in cfg['ds']:
            pi = token_freq(d, alpha)

            # learning rate
            if cfg['eta'] == 'inv_pi1':
                eta = 1 / pi[0]
            else:
                eta = cfg['eta']

            # time grid
            t_start, t_stop = cfg['t_range'][alpha]
            if t_stop == 'd/2':
                t_stop = d / 2
            t_steps = np.geomspace(t_start, t_stop, n_t)

            # one row per run
            L = {}
            for beta in cfg['betas']:
                L[beta] = np.zeros((n_runs, n_t))
            q = np.zeros((n_runs, n_t))
            A = np.zeros((n_runs, n_t))

            for r in range(n_runs):
                u_j = build_u(method, d, pi, cfg['n_passes'], rng)
                for k in range(n_t):
                    b_t, q_t, A_t = loss_terms(pi, u_j, eta, t_steps[k])
                    q[r, k] = q_t
                    A[r, k] = A_t
                    for beta in cfg['betas']:
                        L[beta][r, k] = b_t - (2*beta / (1 + beta**2)) * q_t

            results['simulations'][alpha][d] = {
                't_steps': t_steps,
                'eta': eta,
                'pi1': pi[0],
                'refline_ok': bool(np.isclose(eta, 1 / pi[0])),
                'L': L,
                'q': q,
                'A': A,
            }

    filepath = '/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/population/data/' + cfg['name'] + '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    return filepath
