import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

from utils import plotsetup

# parameters
ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
densities = ['sparse', 'dense']

f = 1
q = (2.5, 97.5)
nplt = 16384
vplt = 4440 #1512

labelsize = 11
nbins = 12

# adjust these as needed
ylim_error = {
    'sparse': (9e-3, 2.5),
    'dense':  (5e-4, 2e-1),
}

xlim_distribution = {
    'sparse': (-0.45, 0.25),
    'dense':  (-0.26, 0.16),
}

ylim_distribution = {
    'sparse': (-0.01, 8.0),
    'dense':  (-0.01, 15.0),
}

letter = {
    'sparse_dist':  r"\textbf{\textsf{(a)}}",
    'dense_dist':   r"\textbf{\textsf{(b)}}",
    'sparse_error': r"\textbf{\textsf{(c)}}",
    'dense_error':  r"\textbf{\textsf{(d)}}",
}

texpath = "./"


def main():

    plotsetup(texpath)

    fig, axs = plt.subplots(
        2, 2,
        figsize=(7.2, 5.0),
        constrained_layout=True
    )

    for j, density in enumerate(densities):

        # -------------------------------------------------------------
        # Error decay
        # -------------------------------------------------------------
        mean_w1 = []
        lo_w1 = []
        hi_w1 = []

        for n in ns:
            print(density, n)

            theory = np.load(
                f'./theory/srgg/{density}/theory_and_errors_n={n}.npz'
            )

            errors = theory['errors_wass']

            mean_w1.append(np.mean(errors))
            lo_w1.append(np.percentile(errors, q[0]))
            hi_w1.append(np.percentile(errors, q[1]))

        mean_w1 = np.asarray(mean_w1)
        lo_w1 = np.asarray(lo_w1)
        hi_w1 = np.asarray(hi_w1)

        # -------------------------------------------------------------
        # Distribution at n = nplt
        # -------------------------------------------------------------
        result = np.load(
            f'./results/srgg/{density}/results_n={nplt}.npz'
        )
        h = result['preactivations_1']

        theory = np.load(
            f'./theory/srgg/{density}/theory_and_errors_n={nplt}.npz'
        )
        mu = theory['means_theory']
        sig = theory['sigs_theory']

        # -------------------------------------------------------------
        # Top row: marginal distribution
        # -------------------------------------------------------------
        ax = axs[0, j]

        ax.set_title(
            r'$\mathrm{SRGG}$, sparse'
            if density == 'sparse'
            else r'$\mathrm{SRGG}$, dense'
        )

        ax.text(
            0.03,
            0.90,
            letter[f'{density}_dist'],
            transform=ax.transAxes,
            fontsize=11
        )

        ax.set_ylabel(
            r'$\mathbb{P}(h^{(1)})$',
            fontsize=labelsize
        )
        ax.set_xlabel(
            r'$h^{(1)}$',
            fontsize=labelsize
        )

        # theoretical Gaussian
        hplt = np.linspace(
            mu[vplt, f] - 7 * sig[vplt, f],
            mu[vplt, f] + 7 * sig[vplt, f],
            num=250
        )

        ax.plot(
            hplt,
            stats.norm.pdf(
                hplt,
                mu[vplt, f],
                sig[vplt, f]
            ),
            ls='-',
            lw=1.25,
            color='black'
        )

        # empirical distribution
        bins = np.linspace(
            mu[vplt, f] - 3 * sig[vplt, f],
            mu[vplt, f] + 3 * sig[vplt, f],
            num=nbins + 1
        )

        centers = 0.5 * (bins[1:] + bins[:-1])

        ph, _ = np.histogram(
            h[:, vplt, f],
            bins=bins,
            density=True
        )

        finite = np.isfinite(ph)

        ax.plot(
            centers[finite],
            ph[finite],
            marker='o',
            markersize=5.0,
            markerfacecolor='#E3A34F',
            markeredgecolor='#516572',
            markeredgewidth=0.5,
            color='#516572',
            lw=0.0
        )

        ax.set_xlim(xlim_distribution[density])
        ax.set_ylim(ylim_distribution[density])

        # -------------------------------------------------------------
        # Bottom row: Wasserstein error
        # -------------------------------------------------------------
        ax = axs[1, j]

        ax.errorbar(
            ns,
            mean_w1,
            yerr=[
                mean_w1 - lo_w1,
                hi_w1 - mean_w1
            ],
            fmt='none',
            ecolor='#516572',
            elinewidth=1.0,
            capsize=1.5
        )

        ax.plot(
            ns,
            mean_w1,
            marker='o',
            markersize=5.0,
            markerfacecolor='#8A6BBE',
            markeredgecolor='#516572',
            markeredgewidth=0.5,
            color='#516572',
            ls='-',
            lw=0.9
        )

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(
            r'$n$',
            fontsize=labelsize
        )
        ax.set_ylabel(
            r'$W_1$',
            fontsize=labelsize
        )

        ax.set_xlim(9e1, 3e4)
        ax.set_ylim(ylim_error[density])

        ax.text(
            0.03,
            0.90,
            letter[f'{density}_error'],
            transform=ax.transAxes,
            fontsize=11
        )

    simulation_handle = Line2D(
        [],
        [],
        marker='o',
        markersize=5.0,
        markerfacecolor='#E3A34F',
        markeredgecolor='#516572',
        markeredgewidth=0.5,
        linestyle='none'
    )

    theory_handle = Line2D(
        [],
        [],
        color='black',
        lw=1.25,
        ls='-'
    )

    error_handle = axs[1, 0].errorbar(
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
        [
            simulation_handle,
            theory_handle,
            error_handle
        ],
        [
            'simulation',
            'theory',
            r'$W_1$ distance ($95\%$ interval)'
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        fontsize=8
    )

    fig.savefig(
        './distribution-srgg.pdf',
        bbox_inches='tight'
    )


if __name__ == "__main__":
    main()