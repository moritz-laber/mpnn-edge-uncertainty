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
q = (25, 75)
labelsize = 11

nbins = 12
ylim = {
    'gnp' : (0.01, 5),
    'cl' : (0.01, 5),
    'sbm' : (0.005, 50),
    'dcsbm' : (0.005, 50),
}

color = ['#69c09a', '#cf7daa', '#6eaae0']

letter = {
    'gnp' : r"\textbf{\textsf{(a)}}",
    'cl' : r"\textbf{\textsf{(b)}}",
    'sbm' : r"\textbf{\textsf{(c)}}",
    'dcsbm' : r"\textbf{\textsf{(d)}}",
}

title = {
    'gnp' : r'$\mathrm{ER}$',
    'cl' : r'$\mathrm{CL}$',
    'sbm' : r'$\mathrm{SBM}$',
    'dcsbm' : r'$\mathrm{DC}$-$\mathrm{SBM}$'
}

texpath = "./"

def main():

    plotsetup(texpath)
    
    fig, axs = plt.subplots(
        1, 4,
        figsize=(7.2, 1.9),
        constrained_layout=True
    )

    for j, model in enumerate(models):
        
        mean_err = [[] for i in range(3)]
        lo_err = [[] for i in range(3)]
        hi_err = [[] for i in range(3)]
        
        for n in ns:
            print(model, n)
            theory = np.load(f'./deterministic/{model}/dense/deterministic_n={n}.npz')
            mean_err[0].append(np.mean(theory['errors_1']))
            lo_err[0].append(np.percentile(theory['errors_1'], q[0]))
            hi_err[0].append(np.percentile(theory['errors_1'], q[1]))
            
            mean_err[1].append(np.mean(theory['errors_2']))
            lo_err[1].append(np.percentile(theory['errors_2'], q[0]))
            hi_err[1].append(np.percentile(theory['errors_2'], q[1]))
            
            mean_err[2].append(np.mean(theory['errors_3']))
            lo_err[2].append(np.percentile(theory['errors_3'], q[0]))
            hi_err[2].append(np.percentile(theory['errors_3'], q[1]))

        for i in range(3):
            mean_err[i] = np.asarray(mean_err[i])
            lo_err[i] = np.asarray(lo_err[i])
            hi_err[i] = np.asarray(hi_err[i])


        axs[j].set_title(title[model])
        axs[j].text(0.03, 0.87, letter[model], transform=axs[j].transAxes)

        for i in range(3):

            axs[j].errorbar(
                ns,
                mean_err[i],
                yerr=[mean_err[i] - lo_err[i], hi_err[i] - mean_err[i]],
                fmt='o-',
                markersize=5.0,
                markerfacecolor=color[i],
                markeredgecolor='#516572',
                markeredgewidth=0.5,
                color='#516572',
                ecolor='#516572',
                lw=0.9,
                elinewidth=1.0,
                capsize=2.5
            )
        
        axs[j].set_xscale('log')
        axs[j].set_yscale('log')
        axs[j].set_xlabel(r'$n$', fontsize=labelsize)
        axs[j].set_ylabel(r'$||\bar{x}_i^{(\ell)}-x_i^{(\ell)}||_2$', fontsize=labelsize)
        axs[j].set_ylim(ylim[model])
        axs[j].set_xlim(9e1, 3e4)


    # legend
    error_handles = [
        axs[j].errorbar(
            [], [],
            yerr=[],
            fmt='o-',
            markersize=5.0,
            markerfacecolor=color[i],
            markeredgecolor='#516572',
            markeredgewidth=0.5,
            color='#516572',
            ecolor='#516572',
            lw=0.9,
            elinewidth=1.0,
            capsize=2.5,
        )
        for i in range(3)
    ]
    
    fig.legend(
        error_handles,
        [r'$\ell = 1$', r'$\ell = 2$', r'$\ell = 3$'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        fontsize=8
    )

    fig.savefig('./deterministic-dense.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()
            

        
        
    

    



