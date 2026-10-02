import pickle
import numpy as np
import matplotlib.pyplot as plt
from POPtheory import refline, refline_t_grid, refline_label, t_star

PLOTS_DIR = '/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/population/plots/'


def load_results(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def power_label(d):
    if d >= 10:
        return f"10^{{{int(np.log10(d))}}}"
    return f"{d}"


def eta_label(cfg):
    if cfg['eta'] == 'inv_pi1':
        return r'$\eta = 1/\pi_1$'
    return rf"$\eta = {cfg['eta']}$"


def runs_label(runs):
    # runs has shape (n_runs, n_t)
    if runs.shape[0] > 1:
        return f', mean $\\pm$ std over {runs.shape[0]} runs'
    return ''


def plot_mean_std(ax, t_steps, runs, color, label):
    # runs has shape (n_runs, n_t): plot the mean and, if more than one run, a +-std band
    mean = np.mean(runs, axis=0)
    std = np.std(runs, axis=0)
    ax.plot(t_steps, mean, color=color, linewidth=2.0, label=label)
    if runs.shape[0] > 1:
        ax.fill_between(t_steps, mean - std, mean + std, color=color, alpha=0.3)




def plot_d_fixed(filepath, save=True):
    data = load_results(filepath)
    cfg = data['config']
    sims = data['simulations']

    alphas = cfg['alphas']
    ds = cfg['ds']
    betas = cfg['betas']
    method = cfg['method']

    colors = plt.colormaps['Oranges'](np.linspace(0.4, 0.95, len(betas)))

    for d in ds:
        power_d = power_label(d)

        # One panel per alpha regime, each holding the whole beta sweep
        fig, axes = plt.subplots(1, len(alphas), figsize=(6.0 * len(alphas), 5.5))
        axes = np.atleast_1d(axes)

        for ax, alpha in zip(axes, alphas):
            run = sims[alpha][d]
            t_steps = run['t_steps']

            for i, beta in enumerate(betas):
                plot_mean_std(ax, t_steps, run['L'][beta], colors[i], rf'$\beta = {beta}$')

            # Theoretical scaling law reference line, only when eta = 1/pi_1
            if run['refline_ok']:
                t_ref = refline_t_grid(alpha, d, t_steps)
                ax.plot(
                    t_ref, refline(alpha, d, t_ref),
                    linestyle='--',
                    color='black',
                    alpha=0.75,
                    linewidth=2.0,
                    label=refline_label(alpha)
                )

            # Panel formatting
            ax.set_title(rf'$\alpha = {alpha}$', fontsize=13)
            ax.set_xlabel(r'Step ($t$)', fontsize=12)
            ax.set_ylabel(r'$r_d(t)$', fontsize=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(False)
            ax.legend(fontsize=11)
            ax.set_ylim(1e-3, 1.2)

        fig.suptitle(
            rf'Population Loss dynamics varying $\beta$ ($d={power_d}$, {method} $u$, {eta_label(cfg)}{runs_label(run["q"])})',
            fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(PLOTS_DIR + f"loss_vs_beta/{cfg['name']}_loss_all_alphas_d{d}.png")
        plt.show()




def perturb_figure(cfg, sims, key, divide_by_pi1, ylabel, title, filename, save):
    alphas = cfg['alphas']
    ds = cfg['ds']

    colors = plt.colormaps['Oranges'](np.linspace(0.25, 0.95, len(ds)))

    fig, axes = plt.subplots(1, len(alphas), figsize=(18, 5.5))
    axes = np.atleast_1d(axes)

    for j, alpha in enumerate(alphas):
        ax = axes[j]

        # t* line only when eta = 1/pi_1 for every d of the panel
        all_ok = True
        for d in ds:
            if not sims[alpha][d]['refline_ok']:
                all_ok = False
        if divide_by_pi1 and all_ok:
            ax.axvline(
                x=t_star(alpha),
                color='black',
                linestyle=':',
                linewidth=1.8,
                label=r'$\tau = [-2log(1 - 2^{-\alpha})]^{-1}$'
            )

        for i, d in enumerate(ds):
            run = sims[alpha][d]
            runs = run[key]
            if divide_by_pi1:
                runs = runs / run['pi1']
            plot_mean_std(ax, run['t_steps'], runs, colors[i], rf'$d = {power_label(d)}$')

        ax.set_title(rf'$\alpha = {alpha}$', fontsize=13)
        ax.set_xlabel(r'Step ($t$)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xscale('log')
        ax.grid(False)

        if j == 0:
            ax.legend(fontsize=11, loc='lower right')

    fig.suptitle(
        title + rf" ({cfg['method']} $u$, {eta_label(cfg)}{runs_label(sims[alphas[0]][ds[0]]['q'])})",
        fontsize=14)
    fig.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR + 'perturbation/' + filename)
    plt.show()


def plot_perturb_evo(filepath, save=True):
    data = load_results(filepath)
    cfg = data['config']
    sims = data['simulations']
    name = cfg['name']

    perturb_figure(cfg, sims, 'q', False, r'$q_d(t)$',
                   r'$q(t)$ dynamics across vocabulary sizes',
                   f'{name}_perturb_evo_q.png', save)

    perturb_figure(cfg, sims, 'A', False, r'$A_d(t)$',
                   r'$A(t)$ dynamics across vocabulary sizes',
                   f'{name}_perturb_evo_A.png', save)

    perturb_figure(cfg, sims, 'A', True, r'$A_d(t)/\pi_1$',
                   r'$q(t)/q(0) = A(t)/\pi_1$ dynamics across vocabulary sizes',
                   f'{name}_perturb_evo_Anorm.png', save)
