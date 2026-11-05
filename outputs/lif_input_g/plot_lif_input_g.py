"""LIF responses to single synaptic events using Shiu repository defaults.

Source: https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py
Here g is the voltage-valued synaptic drive, not a conductance.
At 20 ms, g += g_peak. Thereafter dg/dt = -g / 5 ms.
At a spike, both v and g reset; v is held during the 2.2 ms refractory period.
The action-potential waveform is not part of this model.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
REST = RESET = -52.0
THRESHOLD = -45.0
TAU_M, TAU_S, REFRACTORY = 20.0, 5.0, 2.2
DT, DURATION, ONSET = 0.01, 120.0, 20.0
AMPLITUDES = [0., 10., 20., 40., 60., 100.]
TIME = np.arange(round(DURATION / DT) + 1) * DT


def simulate(peak):
    voltage = np.full(TIME.size, REST)
    drive = np.zeros_like(TIME)
    spikes = []
    refractory_until = -np.inf
    decay_m, decay_s = np.exp(-DT / TAU_M), np.exp(-DT / TAU_S)
    # Exact solution of the coupled linear ODEs within each time step.
    coupling = TAU_S / (TAU_M - TAU_S) * (decay_m - decay_s)
    onset_index = round(ONSET / DT)
    for k in range(TIME.size - 1):
        if k == onset_index:
            drive[k] += peak
        if TIME[k] < refractory_until:
            voltage[k + 1] = RESET
            drive[k + 1] = drive[k]
            continue
        next_voltage = REST + (voltage[k] - REST) * decay_m + drive[k] * coupling
        next_drive = drive[k] * decay_s
        if next_voltage > THRESHOLD:
            spikes.append(TIME[k + 1])
            voltage[k + 1], drive[k + 1] = RESET, 0.
            refractory_until = TIME[k + 1] + REFRACTORY
        else:
            voltage[k + 1], drive[k + 1] = next_voltage, next_drive
    return voltage, drive, spikes


plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.075, right=.94, bottom=.18, top=.75, hspace=.43, wspace=.43)
fig.suptitle('Membrane voltage and synaptic input over time',
             x=.075, y=.97, ha='left', fontsize=17, fontweight='bold')
fig.text(.075, .908,
         r'Single event at 20 ms: $g \leftarrow g + g_{\mathrm{peak}}$, then $\tau_s\,dg/dt=-g$',
         fontsize=12, color='#475569')
fig.text(.075, .858,
         r'$\tau_m\,dV/dt=V_{\mathrm{rest}}-V+g$;  '
         r'$\tau_m=20$ ms, $\tau_s=5$ ms, $V_{\mathrm{rest}}=V_{\mathrm{reset}}=-52$ mV',
         fontsize=11, color='#475569')

fig.legend(handles=[
    Line2D([], [], color='#196e9c', lw=2.1, label='Voltage V (left axis)'),
    Line2D([], [], color='#c77712', lw=1.9, ls='--', label='Input g (right axis)'),
    Line2D([], [], color='#c25a49', lw=1.1, ls=(0, (5, 3)), label='Spike threshold'),
    Line2D([], [], color='#c25a49', marker='^', ls='none', label='Spike event'),
], loc='lower left', bbox_to_anchor=(.069, .782), ncol=4, frameon=False, fontsize=11)

records = []
summary = []
for letter, ax, peak in zip('ABCDEF', axes.flat, AMPLITUDES):
    v, g, spikes = simulate(peak)
    ax.axvspan(0, ONSET, color='#f1f5f9', zorder=0)
    ax.axvline(ONSET, color='#94a3b8', lw=1, ls=':', zorder=1)
    ax.axhline(THRESHOLD, color='#c25a49', lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.axhline(REST, color='#d8dee7', lw=.8, zorder=1)
    ax.plot(TIME, v, color='#196e9c', lw=2.1, zorder=3)
    g_ax = ax.twinx()
    g_ax.plot(TIME, g, color='#c77712', lw=1.9, ls='--', zorder=2)
    g_ax.set_ylim(-5, 110)
    g_ax.set_yticks([0, 25, 50, 75, 100])
    g_ax.set_ylabel('Input g (mV)', color='#c77712', fontsize=11)
    g_ax.tick_params(axis='y', colors='#c77712', labelsize=10)
    g_ax.spines['right'].set_visible(True)
    g_ax.spines['right'].set_color('#c77712')
    g_ax.spines['left'].set_visible(False)
    ax.spines['left'].set_color('#196e9c')
    ax.tick_params(axis='y', colors='#196e9c')
    if spikes:
        ax.scatter(spikes, np.full(len(spikes), THRESHOLD), marker='^', s=50,
                   color='#c25a49', zorder=5)
    ax.set_title(f'{letter}   ' + r'$g_{\mathrm{peak}} = $' + f'{peak:g} mV', loc='left', pad=9)
    ax.text(.97, .70, '1 spike' if spikes else 'No spike', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            color='#c25a49' if spikes else '#64748b')
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-53, -43.8)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([-52, -50, -48, -46, -45])
    ax.grid(axis='y', color='#edf0f4', lw=.7)
    ax.tick_params(labelsize=10)
    summary.append((peak, float(v.max()), spikes))
    records.extend((peak, float(t), float(vv), float(gg)) for t, vv, gg in zip(TIME, v, g))

for ax in axes[:, 0]:
    ax.set_ylabel('Membrane voltage (mV)', color='#196e9c')
for ax in axes[-1, :]:
    ax.set_xlabel('Time (ms)')
fig.text(.075, .075, 'All panels use the same axis scales. Input g decays with a 5 ms time constant and resets to 0 at each spike.',
         fontsize=11, color='#475569')
fig.text(.075, .035, 'Spike threshold: −45 mV; voltage reset: −52 mV; refractory period: 2.2 ms. Action-potential waveforms are not modeled.',
         fontsize=10, color='#64748b')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'lif_input_g_overlay.{ext}', dpi=220, facecolor='white')
plt.close(fig)

with (OUT / 'lif_input_g_traces.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['g_peak_mV', 'time_ms', 'voltage_mV', 'synaptic_drive_mV'])
    writer.writerows(records)

# Check the non-spiking solution against its closed-form expression.
v20, _, sp20 = simulate(20.)
post = TIME >= ONSET
elapsed = TIME[post] - ONSET
expected = REST + 20. * TAU_S / (TAU_M - TAU_S) * (np.exp(-elapsed / TAU_M) - np.exp(-elapsed / TAU_S))
assert not sp20 and np.allclose(v20[post], expected, atol=1e-10)
assert np.all(simulate(0.)[0] == REST)
assert all(len(simulate(p)[2]) == (1 if p >= 60 else 0) for p in AMPLITUDES)
for peak, vmax, spikes in summary:
    print(f'g_peak={peak:g} mV; max recorded V={vmax:.4f} mV; spikes (ms)={spikes}')
print(OUT / 'lif_input_g_overlay.png')
