"""Illustrate homogeneous Poisson input using exponential inter-arrival times.

Illustrative realizations, not data reproduced from Shiu et al.
Each event is an idealized arrival; event heights do not specify synaptic weights.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
T = 1.0


def arrivals(rate, rng):
    t, times = 0., []
    while True:
        t += rng.exponential(1. / rate)
        if t >= T:
            break
        times.append(t)
    return 1000. * np.array(times)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
rng = np.random.default_rng(20261104)
fig = plt.figure(figsize=(13.5, 9.0))
grid = fig.add_gridspec(3, 2, left=.09, right=.97, top=.79, bottom=.15,
                      height_ratios=[1, 1, 1.32], hspace=.82, wspace=.32)
fig.suptitle('Poisson input: random event times at a fixed average rate',
             x=.09, y=.97, ha='left', fontsize=18, fontweight='bold')
fig.text(.09, .915, 'A rate of 100 Hz means 100 events per second on average; individual intervals and event counts vary.',
         fontsize=12, color='#475569')
fig.text(.09, .867,
         r'$N(T)\sim\mathrm{Poisson}(rT)$;  '
         r'$\Delta t\sim\mathrm{Exponential}(r)$;  '
         r'$E[\Delta t]=1/r$   (T in seconds, r in Hz)',
         fontsize=12, color='#475569')

rows = []
for letter, rate, slot in [('A', 20, grid[0, 0]), ('B', 50, grid[0, 1]),
                           ('C', 100, grid[1, 0]), ('D', 200, grid[1, 1])]:
    ax = fig.add_subplot(slot)
    times = arrivals(rate, rng)
    ax.vlines(times, 0, 1, color='#c77712', lw=1.05)
    ax.axhline(0, lw=.8, color='#cbd5e1')
    ax.set_xlim(0, 1000)
    ax.set_ylim(-.04, 1.2)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(f'{letter}   Average rate r = {rate} Hz', loc='left', fontsize=13, pad=12)
    ax.text(1, 1.08, f'This trial: {len(times)} events', transform=ax.transAxes,
            ha='right', fontsize=10, color='#64748b')
    ax.set_xlabel('Time (ms)', fontsize=11)
    rows.extend((rate, 1, float(t)) for t in times)

ax = fig.add_subplot(grid[2, :])
rate = 100
for k in range(5):
    times = arrivals(rate, rng)
    ax.vlines(times, k + .18, k + .82, color='#196e9c', lw=.9)
    ax.text(1010, k + .5, f'{len(times)} events', va='center', fontsize=10, color='#475569')
    rows.extend((rate, k + 2, float(t)) for t in times)
# Regular control has exactly 10 ms between events; all trials shown over 1 s.
regular = np.arange(5, 1000, 10)
ax.vlines(regular, 5.18, 5.82, color='#64748b', lw=.9)
ax.text(1010, 5.5, '100 events', va='center', fontsize=10, color='#475569')
ax.axhline(5, color='#d8dee7', lw=1)
ax.set_xlim(0, 1000)
ax.set_ylim(6, 0)
ax.set_yticks(np.arange(6) + .5)
ax.set_yticklabels([f'Trial {i}' for i in range(1, 6)] + ['Regular'])
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_title('E   Same average rate (100 Hz), different random realizations', loc='left', fontsize=13, pad=12)
fig.text(.09, .065, 'Each vertical tick is one event arrival. Tick height is symbolic and does not represent g or voltage.',
         fontsize=11, color='#475569')
fig.text(.09, .027, 'Poisson at 100 Hz: mean interval = 10 ms. Regular at 100 Hz: every interval = 10 ms. Seed: 20261104.',
         fontsize=10, color='#64748b')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'poisson_input.{ext}', dpi=220, facecolor='white', bbox_inches='tight')
plt.close(fig)
np.savetxt(OUT / 'poisson_input_events.csv', np.array(rows), delimiter=',',
           header='rate_Hz,realization_id,event_time_ms', comments='')
print(OUT / 'poisson_input.png')
