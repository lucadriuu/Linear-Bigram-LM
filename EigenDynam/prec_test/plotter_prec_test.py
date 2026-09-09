import matplotlib.pyplot as plt
import pickle 


def plot_alphas_offdiag(alphas, d, greedy):

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)

    for i, a in enumerate(alphas):

        ax1 = axes[0,i]
        ax2 = axes[1,i]

        file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/data_prec_test/eigenvecs_evo_prectest{a}.pkl"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        t_ax = data["time"]
        c2v1n_bs = data["comp2_vec1_norm"]
        c1v2n_bs = data["comp1_vec2_norm"]

        for beta, y in c2v1n_bs.items():
            ax1.plot(t_ax, y, label = fr"$\beta = {beta}$")

        ax1.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        ax1.set_xlabel("Step (t)", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax1.legend(loc="best")
            ax1.set_ylabel("$V^{{+}}_{{2}}(t)/N^{{-}}(t)$")

        for beta, y in c1v2n_bs.items():
            ax2.plot(t_ax, y, label=fr"$\beta = {beta}$", linewidth=1.8)

        ax2.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax2.set_xscale("log")
        #ax2.set_yscale("log")
        ax2.set_xlabel("Step (t)", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax2.legend(loc="best")
            ax2.set_ylabel("$V^{{-}}_{{1}}(t)/N^{{-}}(t)$")
    if greedy == True:
        plt.suptitle(fr"(Prec test) Dynamics of normalized off-diagonal components ${{V}}^{{+}}_{{2}}(t)$ and ${{V}}^{{-}}_{{1}}(t)$ (greedy sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/plots_prec_test/eigenvecs_dynam_offdiag_gr({d}).png", bbox_inches='tight', dpi = 300)

    else:
        plt.suptitle(fr"(Prec test) Dynamics of normalized off-diagonal components ${{V}}^{{+}}_{{2}}(t)$ and ${{V}}^{{-}}_{{1}}(t)$ (flip sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/plots_prec_test/eigenvecs_dynam_offdiag_fl({d}).png", bbox_inches='tight', dpi = 300)

    
    plt.tight_layout()
    plt.show()

    return





def plot_alphas_diag(alphas, d, greedy):

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)

    for i, a in enumerate(alphas):

        ax1 = axes[0,i]
        ax2 = axes[1,i]

        file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/data_prec_test/eigenvecs_evo_prectest{a}.pkl"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        t_ax = data["time"]
        mu_tn1_bs = data["mu1_t_norm"] 
        mu_tn2_bs = data["mu2_t_norm"] 

        for beta, y in mu_tn1_bs.items():
            ax1.plot(t_ax, y,  label=fr"$\beta = {beta}$", linewidth=1.8)

        ax1.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        ax1.set_xlabel("Step (t)", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax1.legend(loc="best")
            ax1.set_ylabel("$V^{{+}}_{{1}}(t)/N^{{-}}(t)$")

        for beta, y in mu_tn2_bs.items():
            ax2.plot(t_ax, y,  label=fr"$\beta = {beta}$", linewidth=1.8)

        ax2.set_title(fr"$\alpha = {a}$", fontsize=14)
        ax2.set_xscale("log")
        #ax2.set_yscale("log")
        ax2.set_xlabel("Step (t)", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax2.legend(loc="best")
            ax2.set_ylabel("$V^{{-}}_{{2}}(t)/N^{{-}}(t)$")
    if greedy == True:
        plt.suptitle(fr"(Prec test) Dynamics of normalized diagonal components ${{V}}^{{+}}_{{1}}(t)$ and ${{V}}^{{-}}_{{2}}(t)$ (greedy sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/plots_prec_test/eigenvecs_dynam_diag_gr({d}).png", bbox_inches='tight', dpi = 300)

    else:
        plt.suptitle(fr"(Prec test) Dynamics of normalized diagonal components ${{V}}^{{+}}_{{1}}(t)$ and ${{V}}^{{-}}_{{2}}(t)$ (flip sampling, $d = {d}$)", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(fr"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/prec_test/plots_prec_test/eigenvecs_dynam_diag_fl({d}).png", bbox_inches='tight', dpi = 300)

    plt.show()

    return