import matplotlib.pyplot as plt
import numpy as np

from utils import plotsetup, funcbin


# parameters
datasets = [
    'cornell',
    'wisconsin',
    'citeseer',
    'pubmed',
    'chameleon',
    'computers',
    'actor'
]

q = (5.0, 95.0)
labelsize = 11
nbins = 10

title = {
    'actor': r'\texttt{actor}',
    'chameleon': r'\texttt{chameleon}',
    'citeseer': r'\texttt{citeseer}',
    'computers': r'\texttt{computers}',
    'cornell': r'\texttt{cornell}',
    'pubmed': r'\texttt{pubmed}',
    'wisconsin': r'\texttt{wisconsin}',
}

letters = [
    r"\textbf{\textsf{(a)}}",
    r"\textbf{\textsf{(b)}}",
    r"\textbf{\textsf{(c)}}",
    r"\textbf{\textsf{(d)}}",
    r"\textbf{\textsf{(e)}}",
    r"\textbf{\textsf{(f)}}",
    r"\textbf{\textsf{(g)}}",
]

ylim = {
    'actor' : (9e-2, 50),
    'chameleon' : (9e-1, 1.1e2),
    'citeseer' : (9e-1, 50),
    'computers' : (9e-1, 50),
    'cornell' : (9e-1, 50),
    'pubmed' : (9e-1, 50),
    'wisconsin' : (9e-1, 50)
}

texpath = "./"


def main():

    plotsetup(texpath)

    fig = plt.figure(
        figsize=(7.2, 3.6),
        constrained_layout=False
    )

    gs = fig.add_gridspec(2, 8, hspace=0.60, wspace=2.5)

    axs = [
        # top row
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[0, 6:8]),

        # bottom row, centered
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
        fig.add_subplot(gs[1, 5:7]),
    ]

    for j, dataset in enumerate(datasets):

        print(dataset)

        theory = np.load(
            f'./theory/realworld/theory_and_errors_{dataset}.npz'
        )

        kappa = (
            theory['kappas'][None]
            * np.ones_like(theory['pre_error_1'])
        ).ravel()

        pre = theory['pre_error_1'].ravel()

        # bin errors by expected degree
        mask = kappa > 1

        center, ymean, ylo, yhi = funcbin(
            kappa[mask],
            pre[mask],
            nbins=nbins,
            q=q,
        )

        center = np.asarray(center)
        ymean = np.asarray(ymean)
        ylo = np.asarray(ylo)
        yhi = np.asarray(yhi)

        valid = (
            np.isfinite(center)
            & np.isfinite(ymean)
            & np.isfinite(ylo)
            & np.isfinite(yhi)
        )

        axs[j].errorbar(
            center[valid],
            ymean[valid],
            yerr=[
                ymean[valid] - ylo[valid],
                yhi[valid] - ymean[valid],
            ],
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

        axs[j].set_xscale('log')
        axs[j].set_yscale('log')

        axs[j].set_title(title[dataset])
        axs[j].set_xlabel(
            r'$\kappa$',
            fontsize=labelsize
        )

        if j == 0 or j == 4:
            axs[j].set_ylabel(
                r'$||\hat{h}_i^{(1)} - h_i^{(1)}||_2$',
                fontsize=labelsize
            )

        axs[j].text(
            0.03,
            0.85,
            letters[j],
            transform=axs[j].transAxes
        )

        axs[j].set_ylim(ylim[dataset])

    fig.savefig(
        './realworld_remaining.pdf',
        bbox_inches='tight'
    )


if __name__ == "__main__":
    main()