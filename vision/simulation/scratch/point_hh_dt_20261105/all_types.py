"""Same-current isolated StableHH responses for every curated neuron type."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

from sweep import ionic, rates, reference


def equilibrium(p):
    def rhs(y):
        a, b = rates(y[0])
        return np.r_[-ionic(y[0], y[1:], p), a*(1-y[1:])-b*y[1:]]
    def balance(v):
        a, b = rates(v)
        return ionic(v, a/(a+b), p)
    grid = np.arange(-120., 81.)
    candidates = []
    for left, right in zip(grid[:-1], grid[1:]):
        if balance(left)*balance(right) > 0:
            continue
        v = brentq(balance, left, right, xtol=1e-13)
        a, b = rates(v)
        state = np.r_[v, a/(a+b)]
        eps = 1e-6
        jacobian = np.column_stack([(rhs(state+np.eye(4)[i]*eps)-rhs(state-np.eye(4)[i]*eps))/(2*eps)
                                    for i in range(4)])
        max_real = float(np.max(np.linalg.eigvals(jacobian).real))
        if max_real < 0:
            candidates.append((abs(v-p['eLeak']), state, max_real))
    assert candidates, ('No stable intrinsic equilibrium', p)
    _, state, max_real = min(candidates, key=lambda item: item[0])
    assert np.max(np.abs(rhs(state))) < 1e-8
    return state, max_real


def main(args):
    sys.path.insert(0, str(args.ripple/'src'))
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from ripple.biophysics.channels import StableHH
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert jax.default_backend() == 'cpu'
    assert jax.config.read('jax_enable_x64')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = args.ripple/'configs/parameters/t4t5/curated/v1/ion_channel_rules.csv'
    with source.open() as f:
        table = list(csv.DictReader(f))
    names = [r['neuron_type'] for r in table]
    params = [{k: float(v) for k, v in r.items() if k != 'neuron_type'} for r in table]
    assert len(names) == len(set(names)) == 12
    dt, end, amplitude = .025, 100., 20.
    t = np.arange(round(end/dt)+1)*dt
    drive = np.where((t >= 20.) & (t < 60.), amplitude, 0.)
    network = jx.Network([jx.Cell() for _ in names])
    network.insert(StableHH(name='HH'))
    network.set('capacitance', 1.)
    rows, initial_states, stability = [], [], []
    for i, p in enumerate(params):
        state, eigmax = equilibrium(p)
        initial_states.append(state)
        stability.append(eigmax)
        cell = network.cell(i)
        for key, value in p.items():
            cell.set('HH_'+key, value)
        cell.set('v', state[0])
        for gate, value in zip(('m', 'h', 'n'), state[1:]):
            cell.set('HH_'+gate, value)
        area_um2 = float(2*np.pi*cell.nodes.iloc[0]['radius']*cell.nodes.iloc[0]['length'])
        cell.stimulate(jnp.asarray(drive*area_um2*1e-5), verbose=False)
        cell.record('v', verbose=False)
    assert len(network.edges) == 0
    voltage = np.asarray(jx.integrate(network, delta_t=dt, solver='bwd_euler'))[:, :len(t)]
    assert voltage.shape == (12, len(t)) and np.isfinite(voltage).all()
    baseline = np.array([s[0] for s in initial_states])
    np.testing.assert_allclose(voltage[:, 0], baseline, rtol=0, atol=1e-12)
    assert np.max(abs(voltage[:, t < 20.]-baseline[:, None])) < 1e-8
    delta = voltage-baseline[:, None]
    during = (t >= 20.) & (t < 60.)
    tail = (t >= 55.) & (t < 60.)
    for i, (name, p, state) in enumerate(zip(names, params, initial_states)):
        ref = reference(p, amplitude, state)
        ref_v = ref(t)[0]
        rmse = float(np.sqrt(np.mean((voltage[i]-ref_v)**2)))
        peak_idx = np.flatnonzero(during)[np.argmax(voltage[i, during])]
        row = dict(neuron_type=name, rest_mV=float(baseline[i]),
                   peak_mV=float(voltage[i, peak_idx]), peak_delta_mV=float(delta[i, peak_idx]),
                   peak_time_ms=float(t[peak_idx]), plateau_55_60_delta_mV=float(np.mean(delta[i, tail])),
                   min_post_stimulus_mV=float(np.min(voltage[i, t >= 60.])),
                   rmse_vs_DOP853_mV=rmse, equilibrium_max_eigenvalue_per_ms=stability[i])
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Equal y limits make absolute response comparisons honest across panels.
    ymin, ymax = np.floor(voltage.min()/5)*5-2, np.ceil(voltage.max()/5)*5+2
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, .45])
    for i, (name, row) in enumerate(zip(names, rows)):
        ax = fig.add_subplot(gs[i//3, i%3])
        ax.axvspan(20, 60, color='gray', alpha=.12)
        ax.axhline(baseline[i], color='gray', ls='--', lw=.8)
        ax.plot(t, voltage[i], color=colors[i], lw=1.7)
        ax.set(xlim=(0, 100), ylim=(ymin, ymax), title=name,
               xlabel='Time (ms)', ylabel='Voltage (mV)')
        ax.text(.97, .05, f"Rest {row['rest_mV']:.2f} mV\nPeak delta {row['peak_delta_mV']:.2f} mV\nTail delta {row['plateau_55_60_delta_mV']:.2f} mV",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        ax.grid(alpha=.2)
    stimulus_ax = fig.add_subplot(gs[4, :])
    stimulus_ax.plot(t, drive, color='black', lw=1.6)
    stimulus_ax.set(xlim=(0, 100), ylim=(-1, 23), xlabel='Time (ms)',
                    ylabel='Input\n(uA/cm2)', yticks=[0, 20])
    stimulus_ax.grid(alpha=.2)
    fig.suptitle('All 12 curated HH types | Identical 20 uA/cm2 current, 20-60 ms\n'
                 'Isolated single compartments; no synapses | StableHH / Jaxley, dt=0.025 ms, float64\n'
                 'HH is used for every type here, including photoreceptors; manual model priors, not measured physiology.', fontsize=12)
    fig.savefig(out/'all_types_traces.png', dpi=170)
    fig.savefig(out/'all_types_traces.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, .35]}, constrained_layout=True)
    for i, name in enumerate(names):
        axes[0].plot(t, voltage[i], color=colors[i], label=name, lw=1.5)
        axes[1].plot(t, delta[i], color=colors[i], label=name, lw=1.5)
    for ax in axes[:2]:
        ax.axvspan(20, 60, color='gray', alpha=.12)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Absolute voltage (mV)')
    axes[0].legend(ncol=6, fontsize=9, loc='upper right')
    axes[1].set_ylabel('Voltage minus own rest (mV)')
    axes[2].plot(t, drive, color='black')
    axes[2].set(xlabel='Time (ms)', ylabel='Input\n(uA/cm2)', xlim=(0, 100), ylim=(-1, 23), yticks=[0, 20])
    axes[2].grid(alpha=.2)
    fig.suptitle('Same stimulus, different curated HH parameters | No connections | dt=0.025 ms')
    fig.savefig(out/'all_types_overlay.png', dpi=170)
    fig.savefig(out/'all_types_overlay.pdf')
    plt.close(fig)
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out/'traces.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ms', 'input_uA_per_cm2']+names)
        writer.writerows(np.column_stack([t, drive, voltage.T]))
    np.savez_compressed(out/'traces.npz', time_ms=t, input_uA_per_cm2=drive,
                        neuron_types=np.array(names), voltage_mV=voltage, delta_mV=delta)
    paths = [Path(__file__), Path(__file__).with_name('sweep.py'), source,
             args.ripple/'src/ripple/biophysics/channels.py']
    meta = dict(scope='isolated intrinsic responses of manual curated priors; no biological validation',
                pulse=dict(on_ms=20., off_ms=60., density_uA_per_cm2=20.),
                dt_ms=dt, duration_ms=end, capacitance_uF_per_cm2=1., synapse_count=0,
                initialization='stable zero-input equilibrium closest to eLeak; gates at equilibrium',
                photoreceptor_policy='HH for every curated row; no passive substitution or clamp',
                parameters=dict(zip(names, params)),
                reference='independent piecewise SciPy DOP853 comparison for all 12 traces',
                versions={k: importlib.metadata.version(k) for k in ['jax', 'jaxlib', 'jaxley', 'numpy', 'scipy']},
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
    (out/'provenance.json').write_text(json.dumps(meta, indent=2))
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ripple', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
