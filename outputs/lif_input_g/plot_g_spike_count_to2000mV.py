"""Spike count versus prescribed rectangular-drive amplitude for the 160 ms LIF pulse.

Counts use exact constant-drive threshold crossing times with voltage reset and
2.2 ms absolute refractoriness. The prescribed g is not reset by spikes.
Compare marked amplitudes against the numerical time traces exported separately.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
REST, RESET, THRESHOLD = -52., -52., -45.
TAU_M, REFRACTORY, PULSE_WIDTH = 20., 2.2, 160.
G_THRESHOLD = THRESHOLD - REST
AMPLITUDES = np.linspace(0., 2000., 40001)


def spike_count(g):
    if g <= G_THRESHOLD:
        return 0
    first = -TAU_M * np.log1p(-G_THRESHOLD / g)
    if first >= PULSE_WIDTH:
        return 0
    # Spike times: first + k * (first + refractory). Pulse excludes its end.
    return int(np.count_nonzero(first + np.arange(1000) * (first + REFRACTORY) < PULSE_WIDTH))


counts = np.array([spike_count(g) for g in AMPLITUDES])
marked_g = np.array([0., 200., 500., 1000., 1500., 2000.])
marked_counts = np.array([spike_count(g) for g in marked_g])
# The values must match the six-panel numerical simulation.
assert spike_count(10.) == 6
assert spike_count(20.) == 14
assert spike_count(40.) == 26
assert spike_count(60.) == 34
assert spike_count(100.) == 44
assert spike_count(200.) == 55
assert np.all(np.diff(counts) >= 0)
# Exact thresholds at which the n-th spike first fits into the 160 ms pulse.
n = np.arange(1, 73)
rise_budget = (PULSE_WIDTH - (n - 1) * REFRACTORY) / n
valid = rise_budget > 0
transition_g = G_THRESHOLD / (-np.expm1(-rise_budget[valid] / TAU_M))
transition_n = n[valid]
inside = transition_g <= 2000.
transition_g, transition_n = transition_g[inside], transition_n[inside]
# Represent exact transition locations (the equality itself is a boundary case).
step_x = np.r_[0., transition_g, 2000.]
step_y = np.r_[0, transition_n, transition_n[-1]]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(8.5, 6.2))
fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.79)
fig.suptitle('Spike count versus input amplitude', x=.12, y=.96,
             ha='left', fontsize=19, fontweight='bold')
fig.text(.12, .885, '160 ms rectangular pulse; voltage reset retained; input g unaffected by spikes',
         fontsize=11, color='#475569')
ax.axhline(np.ceil(PULSE_WIDTH / REFRACTORY), color='#94a3b8', lw=1.1, ls=':', label='Large-g limit: 73 spikes')
ax.step(step_x, step_y, where='post', color='#196e9c', lw=2,
        label='LIF response')
ax.scatter(marked_g, marked_counts, s=45, color='#c77712',
           edgecolor='white', linewidth=.8, zorder=4,
           label='Selected input amplitudes', clip_on=False)
for g, count in zip(marked_g, marked_counts):
    ax.annotate(f'({g:g}, {count})', (g, count),
                xytext=((8, -17) if g == 200 else ((-7, -17) if g >= 600 else (7 if g == 0 else -7, 12))), textcoords='offset points',
                ha='left' if g in (0, 200) else 'right', fontsize=10, color='#92510b')
ax.set_xlim(0, 2000)
ax.set_ylim(-1, 79)
ax.set_xlabel('Input amplitude g (mV)', fontsize=13, labelpad=9)
ax.set_ylabel('Spike count during the 160 ms pulse', fontsize=13, labelpad=9)
ax.xaxis.set_major_locator(MultipleLocator(400))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(color='#e8edf2', lw=.8)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False, fontsize=10)
fig.text(.12, .10,
         r'$\tau_m=20$ ms; $V_{\mathrm{rest}}=V_{\mathrm{reset}}=-52$ mV; $V_{\mathrm{th}}=-45$ mV; refractory = 2.2 ms',
         fontsize=10, color='#475569')
fig.text(.12, .045, 'Blue staircase: integer spike count. Orange points: selected input amplitudes.',
         fontsize=10, color='#64748b')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'g_vs_spike_count_160ms_to2000mV.{ext}', dpi=240, facecolor='white')
plt.close(fig)
with (OUT / 'g_vs_spike_count_160ms_to2000mV.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['g_mV', 'spike_count_160ms'])
    writer.writerows(zip(AMPLITUDES, counts))
print(list(zip(marked_g.tolist(), marked_counts.tolist())))
print(OUT / 'g_vs_spike_count_160ms_to2000mV.png')
