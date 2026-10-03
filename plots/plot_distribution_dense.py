import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy import stats

from utils import plotsetup

# parameters
ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
models = ['gnp', 'cl', 'sbm', 'dcsbm']
f = 0
q = (2.5,97.5)
nplt = 16384

labelsize = 11

nbins = 12
ylim_error = {
    'gnp' : (9e-4, 3e-1),
    'cl' : (9e-4, 3e-1),
    'sbm' : (5e-3, 3.5),
    'dcsbm' : (9e-4, 3.5),
}

ylim = {
    'gnp' : (-0.01, 10.2),
    'cl' : (-0.01, 10.2),
    'sbm' : (-0.01, 3.1),
    'dcsbm' : (-0.01, 6.1),
}

xlim = {
    'gnp' : (-0.41, 0.41),
    'cl' : (-0.21, 0.21),
    'sbm' : (-5.1, 5.1),
    'dcsbm' : (-5.1, 5.1)
}

letter = {
    (0, 'gnp') : r"\textbf{\textsf{(a)}}",
    (0, 'cl') : r"\textbf{\textsf{(b)}}",
    (0, 'sbm') : r"\textbf{\textsf{(c)}}",
    (0, 'dcsbm') : r"\textbf{\textsf{(d)}}",
    (1, 'gnp') : r"\textbf{\textsf{(e)}}",
    (1, 'cl') : r"\textbf{\textsf{(f)}}",
    (1, 'sbm') : r"\textbf{\textsf{(g)}}",
    (1, 'dcsbm') : r"\textbf{\textsf{(h)}}",
}

vplt = {
    'gnp' : [7000],
    'cl' : [13484],
    'sbm' : [2500, nplt-1000],
    'dcsbm' : [2500, nplt-1000]
}

title = {
    'gnp' : r'$\mathrm{ER}$',
    'cl' : r'$\mathrm{CL}$',
    'sbm' : r'$\mathrm{SBM}$',
    'dcsbm' : r'$\mathrm{DC}$-$\mathrm{SBM}$'
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

    for j, model in enumerate(models):
        mean_w1 = []
        lo_w1 = []
        hi_w1 = []
        for n in ns:
            print(model, n)
            theory = np.load(f'./theory/{model}/dense/theory_and_errors_n={n}.npz')
            mean_w1.append(np.mean(theory['errors_wass']))
            lo_w1.append(np.percentile(theory['errors_wass'], q[0]))
            hi_w1.append(np.percentile(theory['errors_wass'], q[1]))

        mean_w1 = np.asarray(mean_w1)
        lo_w1 = np.asarray(lo_w1)
        hi_w1 = np.asarray(hi_w1)

        result = np.load(f'./results/{model}/dense/results_n={nplt}.npz')
        h = result['preactivations_1']
        
        theory = np.load(f'./theory/{model}/dense/theory_and_errors_n={nplt}.npz')
        mu = theory['means_theory']
        sig = theory['sigs_theory']

        axs[0, j].set_title(title[model])
        axs[0, j].text(0.03, 0.85, letter[(0, model)], transform=axs[0, j].transAxes)
        axs[0, j].set_ylabel(r'$\mathbb{P}(h^{(1)})$', fontsize=labelsize)
        axs[0, j].set_xlabel(r'$h^{(1)}$', fontsize=labelsize)
        axs[0, j].set_ylim(ylim[model])
        axs[0, j].set_xlim(xlim[model])
        
        for vv, v in enumerate(vplt[model]):
            
            hplt = np.linspace(mu[v, f] - 7 * sig[v, f], mu[v, f] + 7 * sig[v, f], num=250)
            axs[0, j].plot(
                hplt,
                stats.norm.pdf(hplt, mu[v, f], sig[v, f]),
                ls='-',
                lw=1.25,
                color='black',
                label='theory' if vv == 0 else None
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
                lw=0.0,
                label='simulation' if vv == 0 else None
            )

        axs[1, j].errorbar(
            ns,
            mean_w1,
            yerr=[mean_w1 - lo_w1, hi_w1 - mean_w1],
            fmt='none',
            ecolor='#516572',
            capsize=1.5,
            alpha=1.0,
        )

        axs[1, j].plot(
            ns,
            mean_w1,
            marker='o',
            markersize=5.0,
            markerfacecolor='#8A6BBE',
            color='#516572',
            markeredgecolor='#516572',
            markeredgewidth=0.5,
            ls='-',
            lw=0.9)
        
        axs[1, j].set_xscale('log')
        axs[1, j].set_yscale('log')
        axs[1, j].set_xlabel(r'$n$', fontsize=labelsize)
        axs[1, j].set_ylabel(r'$W_1$', fontsize=labelsize)
        axs[1, j].set_ylim(ylim_error[model])
        axs[1, j].set_xlim(9e1, 3e4)
        axs[1, j].text(0.03, 0.85, letter[(1, model)], transform=axs[1, j].transAxes)


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
    
    error_handle = axs[1, j].errorbar(
        [],
        [],
        yerr=[[], []],
        fmt='o-',
        markersize=5.0,
        markerfacecolor='#8A6BBE',
        markeredgecolor='#516572',
        markeredgewidth=0.5,
        color='#516572',
        ecolor='#516572',
        lw=0.9,
        elinewidth=1.0,
        capsize=2.5,
    )
    
    fig.legend(
        [simulation_handle, theory_handle, error_handle],
        ['simulation', 'theory', r'$W_1$ distance ($95\%$ interval)'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        fontsize=8
    )

    fig.savefig('./distribution-dense.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()

        
        
    

    



