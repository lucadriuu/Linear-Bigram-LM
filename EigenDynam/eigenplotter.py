import matplotlib.pyplot as plt
import pickle 

def plot_alphaf(alpha, d, firstv, secv):

    file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/evo_data/eigenvecs_evo{alpha}.pkl"

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    t_ax = data["time"]
    c2v1n_bs = data["comp2_vec1_norm"]
    c1v2n_bs = data["comp1_vec2_norm"]
    mu_tn1_bs = data["mu1_t_norm"]
    mu_tn2_bs = data["mu2_t_norm"]

    evo_alphaf = plt.figure(figsize=(11, 6))

    if firstv == True:

        for beta, y in c2v1n_bs.items():
            plt.plot(t_ax, y, label = fr"$\beta = {beta}$")

        plt.title(fr"Second component of $\vec{{V}}^+(t)$ (d = {d}, $\alpha = {alpha}$)")
        plt.xscale("log")
        #plt.yscale("log")
        plt.xlabel("steps (t)")
        plt.ylabel(fr"$V_{{2}}^{{+}} (t)$")
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/plots/eigenvecs1_dynam({alpha}).png", bbox_inches='tight', dpi = 300)
        plt.show()
    
    else:

        for beta, y in c1v2n_bs.items():
            plt.plot(t_ax, y, label = fr"$\beta = {beta}$")

        plt.title(fr"First component of $\vec{{V}}^-(t)$ (d = {d}, $\alpha = {alpha}$)")
        plt.xscale("log")
        #plt.yscale("log")
        plt.xlabel("steps (t)")
        plt.ylabel(fr"$V_{{1}}^{{-}} (t)$")
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/plots/eigenvecs2_dynam({alpha}).png", bbox_inches='tight', dpi = 300)
        plt.show()

    return


def plot_alphas_offdiag(alphas, d, greedy):

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)

    for i, a in enumerate(alphas):

        ax1 = axes[0,i]
        ax2 = axes[1,i]

        file_path = f"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/evo_data/eigenvecs_evo({a}_{d}).pkl"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        t_ax = data["time"]
        c2v1n_bs = data["comp2_vec1_norm"]
        c1v2n_bs = data["comp1_vec2_norm"]
        pi = data["pi"]
        eta = data["eta"]
        tau_d = 1 / (eta * pi[-1])



        for beta, y in c2v1n_bs.items():
            ax1.plot(t_ax, y, label = fr"$\beta = {beta}$")

        ax1.axvline(tau_d, color="k", linestyle=":", linewidth=1.6,
                    label=r"$\tau_d = 1/(\eta \pi_d)$" if i == 0 else None)

        ax1.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        ax1.set_xlabel("Step (t)", fontsize=11)
        #ax1.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax1.legend(loc="best")
            ax1.set_ylabel("$V^{{+}}_{{2}}(t)/N^{{-}}(t)$")

        for beta, y in c1v2n_bs.items():
            ax2.plot(t_ax, y, label=fr"$\beta = {beta}$", linewidth=1.8)

        ax2.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax2.set_xscale("log")
        #ax2.set_yscale("log")
        ax2.set_xlabel("Step (t)", fontsize=11)
        #ax2.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax2.legend(loc="best")
            ax2.set_ylabel("$V^{{-}}_{{1}}(t)/N^{{-}}(t)$")
    if greedy == True:
        plt.suptitle(fr"Dynamics of normalized off-diagonal components ${{V}}^{{+}}_{{2}}(t)$ and ${{V}}^{{-}}_{{1}}(t)$ (greedy sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/plots/offdiag_gr({d}).png", bbox_inches='tight', dpi = 300)

    else:
        plt.suptitle(fr"Dynamics of normalized off-diagonal components ${{V}}^{{+}}_{{2}}(t)$ and ${{V}}^{{-}}_{{1}}(t)$ (flip sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/plots/offdiag_fl({d}).png", bbox_inches='tight', dpi = 300)

    
    plt.tight_layout()
    plt.show()

    return





def plot_alphas_diag(alphas, d, greedy):

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)

    for i, a in enumerate(alphas):

        ax1 = axes[0,i]
        ax2 = axes[1,i]

        file_path = f"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/evo_data/eigenvecs_evo({a}_{d}).pkl"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        t_ax = data["time"]
        mu_tn1_bs = data["mu1_t_norm"] 
        mu_tn2_bs = data["mu2_t_norm"] 
        pi = data["pi"]
        eta = data["eta"]

        tau_d = 1 / (eta * pi[-1])


        for beta, y in mu_tn1_bs.items():
            ax1.plot(t_ax, y,  label=fr"$\beta = {beta}$", linewidth=1.8)

        ax1.axvline(tau_d, color="k", linestyle=":", linewidth=1.6,
                    label=r"$\tau_d = 1/(\eta \pi_d)$" if i == 0 else None)



        ax1.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        ax1.set_xlabel("Step (t)", fontsize=11)
        #ax1.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax1.legend(loc="best")
            ax1.set_ylabel("$V^{{+}}_{{1}}(t)/N^{{-}}(t)$")

        for beta, y in mu_tn2_bs.items():
            ax2.plot(t_ax, y,  label=fr"$\beta = {beta}$", linewidth=1.8)

        ax2.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax2.set_xscale("log")
        #ax2.set_yscale("log")
        ax2.set_xlabel("Step (t)", fontsize=11)
        #ax2.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax2.legend(loc="best")
            ax2.set_ylabel("$V^{{-}}_{{2}}(t)/N^{{-}}(t)$")
    if greedy == True:
        plt.suptitle(fr"Dynamics of normalized diagonal components ${{V}}^{{+}}_{{1}}(t)$ and ${{V}}^{{-}}_{{2}}(t)$ (greedy sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/plots/diag_gr({d}).png", bbox_inches='tight', dpi = 300)

    else:
        plt.suptitle(fr"Dynamics of normalized diagonal components ${{V}}^{{+}}_{{1}}(t)$ and ${{V}}^{{-}}_{{2}}(t)$ (flip sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/population/EigenDynam/plots/diag_fl({d}).png", bbox_inches='tight', dpi = 300)

    plt.show()

    return



def plot_multirun(alphas, n_runs, d, flip_mp_hyb, init_norm):

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)

    for i, a in enumerate(alphas):

        ax1 = axes[0,i]
        ax2 = axes[1,i]

        if init_norm == True:

            file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/evo_data/norm_multi_eigenvecs_evo({a}_{d}).pkl"

        else:

            file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/evo_data/multi_eigenvecs_evo({a}_{d}).pkl"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        t_ax = data["time"]
        b_ax = data["beta"]
        c2v1n_mean = data["comp2_vec1_norm_mean"] 
        c2v1n_std = data["comp2_vec1_norm_std"] 
        c1v2n_mean = data["comp1_vec2_norm_mean"] 
        c1v2n_std = data["comp1_vec2_norm_std"] 

        for idx, beta in enumerate(b_ax):
            
            # --- Top Row: comp2_vec1_norm ---
            y1_mean = c2v1n_mean[idx, :]
            y1_std = c2v1n_std[idx, :]
            
            line1, = ax1.plot(t_ax, y1_mean, label=fr"$\beta = {beta}$", linewidth=1.8)
            ax1.fill_between(t_ax, y1_mean - y1_std, y1_mean + y1_std, color=line1.get_color(), alpha=0.15)

            # --- Bottom Row: comp1_vec2_norm ---
            y2_mean = c1v2n_mean[idx, :]
            y2_std = c1v2n_std[idx, :]
            
            line2, = ax2.plot(t_ax, y2_mean, label=fr"$\beta = {beta}$", linewidth=1.8)
            ax2.fill_between(t_ax, y2_mean - y2_std, y2_mean + y2_std, color=line2.get_color(), alpha=0.15)

        # Formatting ax1
        ax1.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax1.set_xscale("log")
        ax1.set_xlabel("Step (t)", fontsize=11)
        #ax1.grid(True, linestyle='--', alpha=0.5)

        # Formatting ax2
        ax2.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax2.set_xscale("log")
        ax2.set_xlabel("Step (t)", fontsize=11)
        #ax2.grid(True, linestyle='--', alpha=0.5)

        # Only add legend and Y-labels to the leftmost column to avoid clutter
        if i == 0:
            ax1.legend(loc="best")
            ax2.legend(loc="best")
            ax1.set_ylabel("$V^{{+}}_{{2}}(t)/N^{{-}}(t)$")
            ax2.set_ylabel("$V^{{-}}_{{1}}(t)/N^{{-}}(t)$")

    plt.suptitle(fr"Dynamics of normalized off-diagonal components ${{V}}^{{+}}_{{1}}(t)$ and ${{V}}^{{-}}_{{2}}(t)$ over {n_runs} runs ($d = {d}$)", fontsize=16, y=1.02)
    plt.tight_layout()
    if flip_mp_hyb == True:
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/plots/hyb_multi_dynam({n_runs},{d}).png", bbox_inches='tight', dpi = 300)
    else:
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/plots/pl_dynam_hyb({n_runs},{d}).png", bbox_inches='tight', dpi = 300)
    plt.show()


    return