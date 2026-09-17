import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpmath
from scipy.special import zeta
from scipy.special import beta as beta_fn      #  beta_fn = beta function


def refline(t_steps, d, alpha):
    # Theoretical scaling-law reference line, same regimes as evo_loss in population/POPloss.py
    ref = []
    for t in t_steps:
        if alpha < 1.0:
            tau = 2*t / (d**alpha)
            ref.append(((1 - alpha) / alpha) * float(mpmath.expint(1/alpha, tau)))
        elif alpha == 1.0:
            tau = np.emath.logn(d, 2*t)
            ref.append(max(1 - tau, 0.0))
        else:
            ref.append(beta_fn(1 - 1/alpha, 1 + 2*t) / (alpha * zeta(alpha)))
    return np.array(ref)


def plot_single(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Older pickles have no metadata: recover it from sim{alpha}_{d}_{seq_len}.pkl
    if 'alpha' in data:
        alpha, d, seq_len = data['alpha'], data['d'], data['seq_len']
    else:
        alpha, d, seq_len = os.path.basename(filename)[3:-4].split('_')
        alpha, d, seq_len = float(alpha), int(d), int(seq_len)

    t_steps = data['t_steps']
    n_seeds = data.get('n_seeds', 1)
    beta_keys = [k for k in data.keys() if k.startswith('beta_')]
    num_betas = len(beta_keys)

    emp_cmap = plt.colormaps['Greens'] if hasattr(plt, 'colormaps') else cm.get_cmap('Greens')
    pop_cmap = plt.colormaps['Oranges'] if hasattr(plt, 'colormaps') else cm.get_cmap('Greens')

    color_intensities = np.linspace(0.4, 0.95, num_betas)
    trasp = np.linspace(0.50, 1.0, num_betas)

    plt.figure(figsize=(10, 6))

    for i, key in enumerate(beta_keys):
        dynamics = data[key]                                      #mean over seeds (single run if n_seeds = 1)
        beta_val = key.replace('beta_', '')
        std = data.get(f'std_{beta_val}')
        emp_color = emp_cmap(color_intensities[i])
        pop_color = pop_cmap(color_intensities[i])

        plt.plot(t_steps, dynamics, linestyle='-', color=emp_color,
                 alpha=trasp[i], linewidth=2.0, label=f'$\\beta = {beta_val}$')

        if std is not None and n_seeds > 1:
            plt.fill_between(t_steps, np.maximum(dynamics - std, 1e-16), dynamics + std,
                             color=emp_color, alpha=trasp[i] * 0.25)

        # population reference at this d and beta (dashed, same color as its empirical curve)
        pop_dynamics = data.get(f'pop_{beta_val}')
        if pop_dynamics is not None:
            plt.plot(t_steps, pop_dynamics, linestyle='--', color=pop_color,
                     alpha=trasp[i], linewidth=2.0, label=rf'$r(t)$, $\beta = {beta_val}$')

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Step ($t$)', fontsize=12)
    plt.ylabel(r'$\hat{\mathcal{L}}(t) / \hat{\mathcal{L}}(0)$', fontsize=12)
    plt.title(rf'Empirical Loss Dynamics over varying $\beta$ ($\alpha={alpha}$, $d={d}$, text len={seq_len}, {n_seeds} seeds)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_distance(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    alpha, d, beta = data['alpha'], data['d'], data['beta']
    ratios = data['ratios']
    n_seeds = data.get('n_seeds', 1)
    t_steps = data['t_steps']

    num_ratios = len(ratios)
    cmap = plt.colormaps['Greens'] if hasattr(plt, 'colormaps') else cm.get_cmap('Greens')
    color_intensities = np.linspace(0.4, 0.95, num_ratios)
    trasp = np.linspace(0.50, 1.0, num_ratios)

    plt.figure(figsize=(10, 6))

    for i, ratio in enumerate(ratios):
        distance = data[f'ratio_{ratio}']
        std = data.get(f'std_{ratio}')
        color = cmap(color_intensities[i])

        plt.plot(t_steps, distance, linestyle='-', color=color,
                 alpha=trasp[i], linewidth=2.0, label=f'$T/d = {ratio}$')

        if std is not None and n_seeds > 1:
            plt.fill_between(t_steps, np.maximum(distance - std, 1e-16), distance + std,
                             color=color, alpha=trasp[i] * 0.2)

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Step ($t$)', fontsize=12)
    plt.ylabel(r'$|\hat{r}(t) - r(t)|$', fontsize=12)
    plt.title(rf'Population vs empirical distance over varying $T/d$ ($\alpha={alpha}$, $d={d}$, $\beta={beta}$, {n_seeds} seeds)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


