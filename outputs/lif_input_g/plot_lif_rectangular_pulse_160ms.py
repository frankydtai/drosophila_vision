"""LIF responses to a prescribed 160 ms rectangular, voltage-valued drive g.

Retain V reset and the 2.2 ms refractory period. The externally prescribed g
has no decay and is unaffected by postsynaptic spikes or refractoriness.
This is a modified input/reset model, not the Shiu paper's synaptic model.
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
TAU_M, REFRACTORY = 20.0, 2.2
PULSE_WIDTH = 160.0
DT, DURATION, ONSET = 0.01, 280.0, 20.0
AMPLITUDES = [0., 10., 20., 40., 60., 100.]
TIME = np.arange(round(DURATION / DT) + 1) * DT


def simulate(amplitude):
    voltage = np.full(TIME.size, REST)
    # Rectangular input on [20, 180) ms, independent of the spike history.
    # Duration matches vision/simulation/conf/config.yaml: ms_sti = 160.0.
    drive = np.zeros_like(TIME)
    drive[round(ONSET / DT):round((ONSET + PULSE_WIDTH) / DT)] = amplitude
    spikes = []
    refractory_end_index = -1
    refractory_steps = round(REFRACTORY / DT)
    decay_m = np.exp(-DT / TAU_M)
    for k in range(TIME.size - 1):
        if k < refractory_end_index:
            voltage[k + 1] = RESET
            continue
        # Exact subthreshold solution for the constant input during this step.
        equilibrium = REST + drive[k]
        next_voltage = equilibrium + (voltage[k] - equilibrium) * decay_m
        if next_voltage > THRESHOLD:
            spikes.append(float(TIME[k + 1]))
            voltage[k + 1] = RESET
            refractory_end_index = k + 1 + refractory_steps
        else:
            voltage[k + 1] = next_voltage
    return voltage, drive, spikes


plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.075, right=.94, bottom=.18, top=.75, hspace=.43, wspace=.43)
fig.suptitle('LIF responses to a 160 ms rectangular input pulse',
             x=.075, y=.97, ha='left', fontsize=17, fontweight='bold')
fig.text(.075, .908,
         r'$g(t)=g_0$ from 20 to 180 ms; $g(t)=0$ otherwise. Input g is independent of spike resets.',
         fontsize=12, color='#475569')
fig.text(.075, .858,
         r'$\tau_m\,dV/dt=V_{\mathrm{rest}}-V+g$;  '
         r'$\tau_m=20$ ms, $V_{\mathrm{rest}}=V_{\mathrm{reset}}=-52$ mV',
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
    ax.axvspan(ONSET, ONSET + PULSE_WIDTH, color='#fff3df', zorder=0)
    ax.axvline(ONSET, color='#94a3b8', lw=1, ls=':', zorder=1)
    ax.axvline(ONSET + PULSE_WIDTH, color='#94a3b8', lw=1, ls=':', zorder=1)
    ax.axhline(THRESHOLD, color='#c25a49', lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.axhline(REST, color='#d8dee7', lw=.8, zorder=1)
    ax.plot(TIME, v, color='#196e9c', lw=2.1, zorder=3)
    g_ax = ax.twinx()
    g_ax.step(TIME, g, where='post', color='#c77712', lw=1.9, ls='--', zorder=2)
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
        ax.scatter(spikes, np.full(len(spikes), THRESHOLD), marker='^', s=13,
                   color='#c25a49', zorder=5)
    ax.set_title(f'{letter}   ' + r'$g_0 = $' + f'{peak:g} mV', loc='left', pad=9)
    ax.text(.97, .70, f'{len(spikes)} spike' + ('s' if len(spikes) != 1 else ''), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            color='#c25a49' if spikes else '#64748b')
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-53, -43.8)
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240, 280])
    ax.set_yticks([-52, -50, -48, -46, -45])
    ax.grid(axis='y', color='#edf0f4', lw=.7)
    ax.tick_params(labelsize=10)
    summary.append((peak, float(v.max()), spikes))
    records.extend((peak, float(t), float(vv), float(gg)) for t, vv, gg in zip(TIME, v, g))

for ax in axes[:, 0]:
    ax.set_ylabel('Membrane voltage (mV)', color='#196e9c')
for ax in axes[-1, :]:
    ax.set_xlabel('Time (ms)')
fig.text(.075, .075, 'Orange shading: 160 ms pulse (sbar duration). Input g stays constant throughout the pulse, including during spikes and refractoriness.',
         fontsize=11, color='#475569')
fig.text(.075, .035, 'Spike threshold: −45 mV; voltage reset: −52 mV; refractory period: 2.2 ms. Action-potential waveforms are not modeled.',
         fontsize=10, color='#64748b')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'lif_rectangular_pulse_160ms.{ext}', dpi=220, facecolor='white')
plt.close(fig)

with (OUT / 'lif_rectangular_pulse_160ms_traces.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['g_amplitude_mV', 'time_ms', 'voltage_mV', 'input_drive_mV'])
    writer.writerows(records)

# Verify a subthreshold pulse against the continuous closed-form response.
v5, g5, spikes5 = simulate(5.)
elapsed = np.maximum(TIME - ONSET, 0.)
active_elapsed = np.minimum(elapsed, PULSE_WIDTH)
expected = REST + 5. * (1. - np.exp(-active_elapsed / TAU_M)) * np.exp(-np.maximum(elapsed - PULSE_WIDTH, 0.) / TAU_M)
assert not spikes5 and np.allclose(v5, expected, atol=1e-10)
assert np.all(simulate(0.)[0] == REST)
# Compare spike counts/times with the constant-drive analytical firing period.
for amplitude, vmax, spikes in summary:
    if amplitude <= THRESHOLD - REST:
        expected_spikes = np.array([])
    else:
        rise_time = -TAU_M * np.log(1. - (THRESHOLD - REST) / amplitude)
        expected_spikes = ONSET + rise_time + np.arange(100) * (rise_time + REFRACTORY)
        expected_spikes = expected_spikes[expected_spikes < ONSET + PULSE_WIDTH]
    assert len(spikes) == len(expected_spikes), (amplitude, spikes, expected_spikes)
    # A threshold detected at the next sample can add up to DT per spike.
    timing_error = np.array(spikes) - expected_spikes
    assert np.all(timing_error >= -1e-8)
    assert np.all(timing_error <= DT * np.arange(1, len(spikes) + 1) + 1e-8)
    _, drive, _ = simulate(amplitude)
    assert np.all(drive[round(ONSET / DT):round((ONSET + PULSE_WIDTH) / DT)] == amplitude)
    print(f'g0={amplitude:g} mV; {len(spikes)} spikes; first/last (ms)={([round(spikes[0], 2), round(spikes[-1], 2)] if spikes else [])}')
print(OUT / 'lif_rectangular_pulse_160ms.png')
