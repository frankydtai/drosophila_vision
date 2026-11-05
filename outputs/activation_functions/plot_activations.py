"""Common ANN activations, with explicit parameters and a softmax example."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
x = np.linspace(-6, 6, 2401)
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(x, 0)
alpha_leaky, alpha_prelu, alpha_elu = .1, .25, 1.
cdf = .5 * (1 + np.array([math.erf(float(v) / math.sqrt(2)) for v in x]))
functions = [
    ('Linear', r'$f(x)=x$', x),
    ('ReLU', r'$f(x)=\max(0,x)$', relu),
    ('Leaky ReLU', r'$\max(0,x)+0.1\min(0,x)$', np.maximum(x, 0) + alpha_leaky * np.minimum(x, 0)),
    ('PReLU', r'$\max(0,x)+a\min(0,x)$; example $a=0.25$', np.maximum(x, 0) + alpha_prelu * np.minimum(x, 0)),
    ('Sigmoid', r'$f(x)=1/(1+e^{-x})$', sigmoid),
    ('Tanh', r'$f(x)=\tanh(x)$', np.tanh(x)),
    ('Softplus', r'$f(x)=\ln(1+e^x)$', np.logaddexp(0, x)),
    ('ELU', r'$x$ if $x>0$; $e^x-1$ otherwise ($\alpha=1$)', np.where(x > 0, x, np.expm1(x))),
    ('GELU', r'$f(x)=x\Phi(x)$', x * cdf),
    ('SiLU / Swish', r'$f(x)=x\,\mathrm{sigmoid}(x)$', x * sigmoid),
    ('Clipped ReLU', r'$f(x)=\min(1,\max(0,x))$', np.clip(x, 0, 1)),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(3, 4, figsize=(16, 10.8), sharex=True)
fig.subplots_adjust(left=.065, right=.975, top=.80, bottom=.105, hspace=.65, wspace=.30)
fig.suptitle('Common ANN activation functions', x=.065, y=.97,
             ha='left', fontsize=23, fontweight='bold')
fig.text(.065, .92, 'Same input range in every panel. Output scales vary so that bounded and negative responses stay visible.',
         fontsize=12, color='#475569')
fig.text(.065, .882, 'PReLU slope is learned (one example shown). Swish uses beta = 1. Softmax is a vector-valued function.',
         fontsize=11, color='#64748b')

for letter, ax, (name, formula, y) in zip('ABCDEFGHIJK', axes.flat, functions):
    color = '#b66a16' if name in ('Sigmoid', 'Tanh', 'Clipped ReLU') else '#196e9c'
    ax.plot(x, y, color=color, lw=2.5, zorder=3)
    ax.set_title(f'{letter}   {name}', loc='left', fontsize=13, fontweight='bold', pad=25)
    ax.text(0, 1.045, formula, transform=ax.transAxes, fontsize=9.3, color='#475569', va='bottom')
    if name in ('Sigmoid', 'Clipped ReLU'):
        ax.set_ylim(-.15, 1.15)
        ax.set_yticks([0, .5, 1])
        ax.axhline(1, color='#b9c1ca', lw=.9, ls='--')
    elif name == 'Tanh':
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        for limit in [-1, 1]:
            ax.axhline(limit, color='#b9c1ca', lw=.9, ls='--')
    elif name == 'Linear':
        ax.set_ylim(-6.5, 6.5)
        ax.set_yticks([-6, -3, 0, 3, 6])
    else:
        ax.set_ylim(-1.8, 6.5)
        ax.set_yticks([-1, 0, 3, 6])
        # A light negative band makes small sub-zero responses easier to find.
        ax.axhspan(-1.8, 0, color='#f1f5f9', zorder=0)
    ax.set_ylabel('f(x)')

ax = axes.flat[-1]
logits = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=1)
exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
prob = exponents / exponents.sum(axis=1, keepdims=True)
assert np.allclose(prob.sum(axis=1), 1.)
for idx, (color, label) in enumerate(zip(['#196e9c', '#c77712', '#7c5ea5'],
                                       [r'$p_1$ (logit $x$)', r'$p_2$ (logit $0$)', r'$p_3$ (logit $1$)'])):
    ax.plot(x, prob[:, idx], color=color, lw=2.2, label=label)
ax.set_title('L   Softmax', loc='left', fontsize=13, fontweight='bold', pad=25)
ax.text(0, 1.045, r'$\mathbf{z}=(x,0,1)$; $p_i=\exp(z_i)/\mathrm{sum}_j\exp(z_j)$',
        transform=ax.transAxes, fontsize=9.3, color='#475569', va='bottom')
ax.set_ylim(-.15, 1.15)
ax.set_yticks([0, .5, 1])
ax.set_ylabel('Class probability')
ax.axhline(1, color='#b9c1ca', lw=.9, ls='--')
ax.legend(loc='center left', frameon=False, fontsize=8)

for ax in axes.flat:
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.axhline(0, color='#9aa5b1', lw=.8)
    ax.axvline(0, color='#b9c1ca', lw=.8, ls=':')
    ax.grid(color='#e8edf2', lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelbottom=True)
    ax.set_xlabel('Input x')
fig.text(.065, .045, 'Orange single curves have an upper bound. GELU uses the exact Gaussian CDF; no tanh approximation.',
         fontsize=11, color='#475569')
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'ann_activation_functions.{ext}', dpi=220, facecolor='white')
plt.close(fig)
np.savetxt(OUT / 'ann_activation_functions.csv',
           np.column_stack([x] + [v[2] for v in functions] + [prob[:, i] for i in range(3)]),
           delimiter=',', header=','.join(['x'] + [v[0] for v in functions] + ['softmax_p1', 'softmax_p2', 'softmax_p3']), comments='')
assert np.allclose(functions[0][2], x)
assert np.all((sigmoid > 0) & (sigmoid < 1))
assert np.all((functions[-1][2] >= 0) & (functions[-1][2] <= 1))
print(OUT / 'ann_activation_functions.png')
