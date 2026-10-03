import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
from utils import plotsetup

texpath = "./"

datasets = [
    'actor',
    'chameleon',
    'citeseer',
    'computers',
    'cora',
    'cornell',
    'photo',
    'pubmed',
    'squirrel',
    'texas',
    'wisconsin'
]



plotsetup(texpath)

test_rw = []
train_rw = []
test_ergm_mean = []
train_ergm_mean = []
test_ergm_lo = []
train_ergm_lo = []
test_ergm_hi = []
train_ergm_hi = []
for dataset in datasets:
    x = np.load(f'./results/realworld/results_{dataset}.npz')
    train_rw.append(x['accuracy_train_realworld'])
    test_rw.append(x['accuracy_test_realworld'])
    train_ergm_mean.append(x['accuracy_train'].mean())
    test_ergm_mean.append(x['accuracy_test'].mean())
    train_ergm_lo.append(np.percentile(x['accuracy_train'],2.5))
    train_ergm_hi.append(np.percentile(x['accuracy_train'], 97.5))
    test_ergm_lo.append(np.percentile(x['accuracy_test'], 2.5))
    test_ergm_hi.append(np.percentile(x['accuracy_test'], 97.5))

train_rw = np.asarray(train_rw)
test_rw = np.asarray(test_rw)
train_ergm_mean = np.asarray(train_ergm_mean)
test_ergm_mean = np.asarray(test_ergm_mean)
train_ergm_lo = np.asarray(train_ergm_lo)
train_ergm_hi = np.asarray(train_ergm_hi)
test_ergm_lo = np.asarray(test_ergm_lo)
test_ergm_hi = np.asarray(test_ergm_hi)

colors = list({
    'actor':     '#9A9FE8',
    'chameleon': '#E98C8C',
    'citeseer':  '#9ED89E',
    'computers': '#F2B06B',
    'cora':      '#B8A3E6',
    'cornell':   '#F0E68A',
    'photo':     '#8FD3C7',
    'pubmed':    '#E9A0B5',
    'squirrel':  '#B6DD8A',
    'texas':     '#A98BE0',
    'wisconsin': '#6FB7D6',
}.values())

fig = plt.figure(figsize=(3.2, 1.9))
ax = fig.add_subplot(111)

for d, c, t_rw, t_ergm, t_ergm_hi, t_ergm_lo in zip(datasets, colors, test_rw, test_ergm_mean, test_ergm_hi, test_ergm_lo):
    ax.errorbar(t_rw, t_ergm, yerr=[[t_ergm - t_ergm_lo], [t_ergm_hi - t_ergm]], capsize=2, markerfacecolor=c, alpha=0.90, color='#516572', marker='s', ls='', label=rf'$\texttt{{d}}$')

for d, c, t_rw, t_ergm, t_ergm_hi, t_ergm_lo in zip(datasets, colors, train_rw, train_ergm_mean, train_ergm_hi, train_ergm_lo):
    ax.errorbar(t_rw, t_ergm, yerr=[[t_ergm - t_ergm_lo], [t_ergm_hi - t_ergm]], capsize=2, markerfacecolor=c, alpha=0.90, color='#516572', marker='o', ls='')

ax.plot(np.linspace(-0.02, 1.03), np.linspace(-0.02, 1.03), lw=1.0, ls='--', color='black')

ax.set_xlim(-0.01, 1.02)
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_ylim(-0.01, 1.02)
ax.set_xlabel(r'$a_{\mathrm{real}}$', fontsize=11)
ax.set_ylabel(r'$a_{\mathrm{DC-SBM}}$', fontsize=11)
ax.tick_params(labelsize=9)


# dataset/color handles
dataset_handles = [
    Line2D(
        [0], [0],
        marker='D',
        linestyle='',
        markersize=5,
        markerfacecolor=c,
        alpha=0.95,
        markeredgecolor='#516572',
        markeredgewidth=0.7,
        label=d,
    )
    for d, c in zip(datasets, colors)
]

# train/test marker handles
split_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        markersize=5,
        markerfacecolor='lightgrey',
        markeredgecolor='#516572',
        markeredgewidth=0.7,
        label='train',
    ),
    Line2D(
        [0], [0],
        marker='s',
        linestyle='',
        markersize=5,
        markerfacecolor='lightgrey',
        markeredgecolor='#516572',
        markeredgewidth=0.7,
        label='test',
    ),
]

handles = dataset_handles + split_handles

ax.legend(
    handles=handles,
    loc='center left',
    bbox_to_anchor=(1.13, 0.4),
    ncol=1,
    borderaxespad=0.0,
    frameon=True,
    fontsize=7,
    handlelength=1.0,
    handletextpad=0.4,
    labelspacing=0.4,
)

fig.subplots_adjust(
    left=0.20,
    right=0.70,
    bottom=0.25,
    top=0.95,
)

fig.savefig('./acccuracy.pdf')