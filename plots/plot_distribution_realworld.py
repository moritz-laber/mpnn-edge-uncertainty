import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy import stats
from utils import plotsetup, funcbin

# parameters

datasets = ['texas', 'cora', 'photo', 'squirrel']
f = 1
q = (5.0, 95.0)

labelsize = 11

nbins = 12
ylim_error = {
    datasets[0] : (9e-1, 5.1e1),
    datasets[1] : (9e-1, 3.1e1),
    datasets[2] : (9e-1, 3.1e1),
    datasets[3] : (0.35, 1.05e2)
}

ylim = {
    datasets[0] : (-0.01, 1.01),
    datasets[1] : (-0.01, 1.01),
    datasets[2] : (-0.01, 1.6),
    datasets[3] : (-0.01, 1.6),
}

xlim = {
    datasets[0] : (-4.1, 4.1),
    datasets[1] : (-1.6, 4.5),
    datasets[2] : (-1.71, 1.71),
    datasets[3] : (-2.1, 2.1)
}

letter = {
    (0, datasets[0]) : r"\textbf{\textsf{(a)}}",
    (0, datasets[1]) : r"\textbf{\textsf{(b)}}",
    (0, datasets[2]) : r"\textbf{\textsf{(c)}}",
    (0, datasets[3]) : r"\textbf{\textsf{(d)}}",
    (1, datasets[0]) : r"\textbf{\textsf{(e)}}",
    (1, datasets[1]) : r"\textbf{\textsf{(f)}}",
    (1, datasets[2]) : r"\textbf{\textsf{(g)}}",
    (1, datasets[3]) : r"\textbf{\textsf{(h)}}",
}

nv = 1

title = {
     datasets[0] : r'\texttt{texas}',
     datasets[1] : r'\texttt{cora}',
     datasets[2] : r'\texttt{photo}',
     datasets[3]: r'\texttt{squirrel}'
}

texpath = "./"
figsize = (7.2, 3.2)

def main():

    plotsetup(texpath)
    
    fig, axs = plt.subplots(
        2, 4,
        figsize=(7.2, 3.2),
        constrained_layout=True
    )

    for j, dataset in enumerate(datasets):

        print(dataset)
        
        theory = np.load(f'./theory/realworld/theory_and_errors_{dataset}.npz')
        mu = theory['means_theory']
        sig = theory['sigs_theory']
        

        kappa = (theory['kappas'][None]*np.ones_like(theory['pre_error_1'])).ravel()
        pre = theory['pre_error_1'].ravel()
        
        result = np.load(f'./results/realworld/results_{dataset}.npz')
        h = result['preactivations_1']
        hrw = result['preactivations_1_realworld']
        groups = result['predicted_groups']

        axs[0, j].set_title(title[dataset])
        axs[0, j].text(0.03, 0.85, letter[(0, dataset)], transform=axs[0, j].transAxes)
        axs[0, j].set_ylabel(r'$\mathbb{P}(h^{(0)})$', fontsize=labelsize)
        axs[0, j].set_xlabel(r'$h^{(0)}$', fontsize=labelsize)
        axs[0, j].set_ylim(ylim[dataset])
        axs[0, j].set_xlim(xlim[dataset])

        
        for g in range(nv):
            
            k_sorted = np.argsort(theory['kappas'])
            ng = k_sorted.shape[0]
            v = k_sorted[ng//2 + 3]

            print(ng)
            print(np.sum(theory['kappas'])/ng)
            
            hplt = np.linspace(mu[v, f] - 7 * sig[v, f], mu[v, f] + 7 * sig[v, f], num=250)
            axs[0, j].plot(
                hplt,
                stats.norm.pdf(hplt, mu[v, f], sig[v, f]),
                ls='-',
                lw=1.25,
                color='black'
            )

            bins = np.linspace(mu[v, f] - 3 * sig[v, f], mu[v, f] + 3 * sig[v, f], num=nbins+1)
            centers = 0.5 * (bins[1:] + bins[:-1])
            ph, _ = np.histogram(h[:, v, f], bins=bins, density=True)
            
            axs[0, j].plot(
                centers[np.isfinite(ph)],
                ph[np.isfinite(ph)],
                marker='o',
                markersize=5.0,
                markerfacecolor='#E3A34F',
                markeredgecolor='#516572',
                markeredgewidth=0.5,
                color='#516572',
                lw=0.0
            )

            axs[0,j].vlines(hrw[v, f], -0.02, 2.1, lw=1.0, ls='--', color='black')

        # plot errors
        center, ymean, ylo, yhi = funcbin(kappa[kappa>1], pre[kappa>1], nbins=10)
        valid = np.isfinite(ymean)
        axs[1,j].errorbar(
            center[valid],
            ymean[valid],
            yerr=[ymean[valid] - ylo[valid], yhi[valid] - ymean[valid]],
            fmt='o-',
            markersize=5.0,
            markerfacecolor='#b287eb',
            markeredgecolor='#516572',
            markeredgewidth=0.5,
            color='#516572',
            ecolor='#516572',
            lw=0.9,
            elinewidth=1.0,
            capsize=2.5)
                
        axs[1, j].set_xscale('log')
        axs[1, j].set_yscale('log')
        axs[1, j].set_xlabel(r'$\kappa$', fontsize=labelsize)
        axs[1, j].set_ylabel(r'$||\hat{h}_i^{(1)} - h_i^{(1)}||_2^2$', fontsize=labelsize)
        axs[1, j].set_ylim(ylim_error[dataset])
        #axs[1, j].set_xlim(9e1, 3e4)
        axs[1, j].text(0.03, 0.85, letter[(1, dataset)], transform=axs[1, j].transAxes)


    # legend
    simulation_handle = Line2D(
    [], [],
    marker='o',
    markersize=5.0,
    markerfacecolor='#E3A34F',
    markeredgecolor='#516572',
    markeredgewidth=0.5,
    linestyle='none'
    )
    
    theory_handle = Line2D(
        [], [],
        color='black',
        lw=1.25,
        ls='-'
    )

    rw_handle = Line2D(
        [], [],
        color='black',
        lw=1.0,
        ls='--'
    )
    
    error_handle = axs[1, j].errorbar(
        [],
        [],
        yerr=[[], []],
        fmt='o-',
        markersize=5.0,
        markerfacecolor='#b287eb',
        markeredgecolor='#516572',
        markeredgewidth=0.5,
        color='#516572',
        ecolor='#516572',
        lw=0.9,
        elinewidth=1.0,
        capsize=2.5,
    )
    
    fig.legend(
        [simulation_handle, theory_handle, rw_handle, error_handle],
        ['simulation', 'theory', r'$\hat{h}_i^{(1)}$', r'$||\hat{h}_i^{(1)} - h_i^{(1)}||_2^2$ ($90\%$ interval)'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        fontsize=8
    )

    fig.savefig('./testfig.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()