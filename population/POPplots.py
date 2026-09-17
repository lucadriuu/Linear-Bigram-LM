import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from POPloss import token_freq

def plot_beta_fixed(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    alphas = data['alphas']
    ds = data['ds']
    beta = data['beta']
    method = data['method']
    sims = data['simulations']

    num_ds = len(ds)
    cmap = cm.get_cmap('Oranges')
    colors = cmap(np.linspace(0.4, 0.95, num_ds))

    for alpha in alphas:
        fig, (ax_L, ax_q) = plt.subplots(1, 2, figsize=(14, 6))

        for i, d in enumerate(ds):
            run = sims[alpha][d]
            t_steps = run['t_steps']
            dynamics = run['dynamics']
            dynamics_q = run['dynamics_q']

            power_label = f"10^{{{int(np.log10(d))}}}" if d >= 10 else f"{d}"
            
            # Left panel: r_d(t) relative loss dynamics
            ax_L.plot( t_steps, dynamics,color=colors[i],linewidth=2.0,label=rf'$d = {power_label}$')

            # Right panel: q_d(t) interaction dynamics
            ax_q.plot(t_steps, dynamics_q,color=colors[i],linewidth=2.0,label=rf'$d = {power_label}$')

        # Plot theoretical scaling law reference line from the largest d run
        largest_d = ds[-1]
        t_ref = sims[alpha][largest_d]['t_steps']
        refline = sims[alpha][largest_d]['refline']

        if alpha < 1.0:
            ref_label = r'$\frac{1-\alpha}{\alpha} E_{1/\alpha}(\tau)$'
        elif alpha == 1.0:
            ref_label = r'$1 - \tau$'
        else:
            ref_label = r'$\mathrm{B}(1-1/\alpha,\, 1+2t)\,/\,\alpha\zeta(\alpha)$'

        ax_L.plot(t_ref, refline,linestyle='--',color='black',alpha=0.75,linewidth=2.0,label=f'{ref_label}')

        # Left panel formatting
        ax_L.set_title(r'$r_d(t)$ dynamics', fontsize=13)
        ax_L.set_xlabel(r'Step ($t$)', fontsize=12)
        ax_L.set_ylabel(r'$r_d(t)$', fontsize=12)
        ax_L.set_xscale('log')
        #ax_L.set_yscale('log')
        ax_L.grid(False)
        ax_L.legend(fontsize=11)

        # Right panel formatting
        ax_q.set_title(r'$q_d(t)$ dynamics', fontsize=13)
        ax_q.set_xlabel(r'Step ($t$)', fontsize=12)
        ax_q.set_ylabel(r'$q_d(t)$', fontsize=12)
        ax_q.set_xscale('log')
        ax_q.grid(False)
        #ax_q.legend(fontsize=11)

        fig.suptitle(fr'Population Loss dynamics varying $d$ ($\alpha={alpha}$, $\beta={beta}$, $\eta = 1/\pi_1$) ', fontsize=14)
        plt.tight_layout()
        plt.show()




def plot_perturb_evo(filename):
   
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    alphas = data['alphas']
    ds = data['ds']
    beta = data.get('beta', None)
    method = data.get('method', '')
    sims = data['simulations']

    num_ds = len(ds)
    cmap = plt.colormaps['Oranges'] if hasattr(plt, 'colormaps') else cm.get_cmap('Oranges')
    colors = cmap(np.linspace(0.25, 0.95, num_ds))

    fig1, axes1 = plt.subplots(1, len(alphas), figsize=(18, 5.5))
    if len(alphas) == 1:
        axes1 = [axes1]

    for j, alpha in enumerate(alphas):
        ax = axes1[j]

        for i, d in enumerate(ds):
            run = sims[alpha].get(d, sims[alpha].get(beta, {}))
            t_steps = run['t_steps']
            dynamics_A = run.get('dynamics_A', run.get('dynamcs_A'))

            power_label = f"10^{{{int(np.log10(d))}}}" if d >= 10 else f"{d}"

            ax.plot(
                t_steps,
                dynamics_A,
                color=colors[i],
                linewidth=2.0,
                label=rf'$d = {power_label}$'
            )

        ax.set_title(rf'$\alpha = {alpha}$', fontsize=13)
        ax.set_xlabel(r'Step ($t$)', fontsize=12)
        ax.set_ylabel(r'$A(t)$', fontsize=12)
        ax.set_xscale('log')
        ax.grid(False)

        if j == 0:
            ax.legend(fontsize=11, loc='lower right')

    fig1.suptitle(r'$A(t)$ dynamics across vocabulary sizes ($\eta = 1/\pi_1$)', fontsize=14)
    fig1.tight_layout()
    plt.show()


    fig2, axes2 = plt.subplots(1, len(alphas), figsize=(18, 5.5))
    if len(alphas) == 1:
        axes2 = [axes2]

    for j, alpha in enumerate(alphas):
        ax = axes2[j]

        # Theoretical transition point: tau = [-2 log(1 - 2^(-alpha))]^(-1)
        t_star = 1.0 / (-2.0 * np.log(1.0 - 2.0 ** (-alpha)))
        ax.axvline(
            x=t_star,
            color='black',
            linestyle=':',
            linewidth=1.8,
            label=r'$\tau = [-2log(1 - 2^{-\alpha})]^{-1}$'
        )

        for i, d in enumerate(ds):
            run = sims[alpha].get(d, sims[alpha].get(beta, {}))
            t_steps = run['t_steps']
            dynamics_A = run.get('dynamics_A', run.get('dynamcs_A'))
            pi_1 = token_freq(d, alpha)[0]

            power_label = f"10^{{{int(np.log10(d))}}}" if d >= 10 else f"{d}"

            ax.plot(
                t_steps,
                dynamics_A / pi_1,
                color=colors[i],
                linewidth=2.0,
                label=rf'$d = {power_label}$'
            )

        ax.set_title(rf'$\alpha = {alpha}$', fontsize=13)
        ax.set_xlabel(r'Step ($t$)', fontsize=12)
        ax.set_ylabel(r'$A(t)/\pi_1$', fontsize=12)
        ax.set_xscale('log')
        ax.grid(False)

        if j == 0:
            ax.legend(fontsize=11, loc='lower right')

    fig2.suptitle(r'$q(t)/q(0) = A(t)/\pi_1$ dynamics across vocabulary sizes ($\eta = 1/\pi_1$)', fontsize=14)
    fig2.tight_layout()
    plt.show()




def plot_d_fixed(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    alphas = data['alphas']
    d = data['d']
    betas = data['betas']
    method = data['method']
    sims = data['simulations']

    num_betas = len(betas)
    cmap = plt.colormaps['Oranges'] if hasattr(plt, 'colormaps') else cm.get_cmap('Oranges')
    colors = cmap(np.linspace(0.4, 0.95, num_betas))

    power_d = f"10^{{{int(np.log10(d))}}}" if d >= 10 else f"{d}"

    for alpha in alphas:
        fig, (ax_L, ax_q) = plt.subplots(1, 2, figsize=(14, 6))

        for i, beta in enumerate(betas):
            run = sims[alpha][beta]
            t_steps = run['t_steps']
            dynamics = run['dynamics']
            dynamics_q = run['dynamics_q']

            # Left panel: r_d(t) for varying beta
            ax_L.plot(
                t_steps, dynamics,
                color=colors[i],
                linewidth=2.0,
                label=rf'$\beta = {beta}$'
            )

            # Right panel: q_d(t) for varying beta
            ax_q.plot(
                t_steps, dynamics_q,
                color=colors[i],
                linewidth=2.0,
                label=rf'$\beta = {beta}$'
            )

        # Plot theoretical scaling law reference line
        any_beta = betas[0]
        t_ref = sims[alpha][any_beta]['t_steps']
        refline = sims[alpha][any_beta]['refline']

        if alpha < 1.0:
            ref_label = r'$\frac{1-\alpha}{\alpha} E_{1/\alpha}(\tau)$'
        elif alpha == 1.0:
            ref_label = r'$1 - \tau$'
        else:
            ref_label = r'$\mathrm{B}(1-1/\alpha,\, 1+2t)\,/\,\alpha\zeta(\alpha)$'

        ax_L.plot(
            t_ref, refline,
            linestyle='--',
            color='black',
            alpha=0.75,
            linewidth=2.0,
            label=f'{ref_label}'
        )

        # Left panel formatting
        ax_L.set_title(r'$r_d(t)$ dynamics', fontsize=13)
        ax_L.set_xlabel(r'Step ($t$)', fontsize=12)
        ax_L.set_ylabel(r'$r_d(t)$', fontsize=12)
        ax_L.set_xscale('log')
        ax_L.set_yscale('log')
        ax_L.grid(False)
        ax_L.legend(fontsize=11)

        # Right panel formatting
        ax_q.set_title(r'$q_d(t)$ dynamics', fontsize=13)
        ax_q.set_xlabel(r'Step ($t$)', fontsize=12)
        ax_q.set_ylabel(r'$q_d(t)$', fontsize=12)
        ax_q.set_xscale('log')
        ax_q.grid(False)
        #ax_q.legend(fontsize=11)

        fig.suptitle(
            rf'Population Loss dynamics varying $\beta$ ($\alpha={alpha}$, $d={power_d}$, {method} $u$, $\eta = 1/\pi_1$)',
            fontsize=14
        )
        plt.tight_layout()
        plt.show()





