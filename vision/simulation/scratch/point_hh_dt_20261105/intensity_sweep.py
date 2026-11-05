"""Three pulse amplitudes per panel for all 12 curated point-HH types."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from all_types import equilibrium


def main(args):
    sys.path.insert(0, str(args.ripple/'src'))
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from ripple.biophysics.channels import StableHH
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert jax.default_backend() == 'cpu' and jax.config.read('jax_enable_x64')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = args.ripple/'configs/parameters/t4t5/curated/v1/ion_channel_rules.csv'
    with source.open() as f:
        table = list(csv.DictReader(f))
    names = [r['neuron_type'] for r in table]
    parameters = [{k: float(v) for k, v in r.items() if k != 'neuron_type'} for r in table]
    states = [equilibrium(p)[0] for p in parameters]
    factors = [.5, 1., 2.]
    dt = .025
    t = np.arange(4001)*dt
    pulse = ((t >= 20.) & (t < 60.)).astype(float)
    network = jx.Network([jx.Cell() for _ in range(36)])
    network.insert(StableHH(name='HH'))
    network.set('capacitance', 1.)
    for i, (p, state) in enumerate(zip(parameters, states)):
        for j, factor in enumerate(factors):
            cell = network.cell(3*i+j)
            for key, value in p.items():
                cell.set('HH_'+key, value)
            cell.set('v', state[0])
            for gate, value in zip(('m', 'h', 'n'), state[1:]):
                cell.set('HH_'+gate, value)
            area = float(2*np.pi*cell.nodes.iloc[0]['radius']*cell.nodes.iloc[0]['length'])
            cell.stimulate(jnp.asarray(pulse*20.*factor*area*1e-5), verbose=False)
            for key in ['v', 'HH_m', 'HH_h', 'HH_n']:
                cell.record(key, verbose=False)
    assert len(network.edges) == 0
    recordings = np.asarray(jx.integrate(network, delta_t=dt, solver='bwd_euler'))[:, :len(t)]
    recordings = recordings.reshape(12, 3, 4, len(t))
    assert np.isfinite(recordings).all()
    gates = recordings[:, :, 1:]
    assert gates.min() >= -1e-10 and gates.max() <= 1+1e-10
    voltage = recordings[:, :, 0]
    baseline = np.array([s[0] for s in states])
    assert np.max(abs(voltage[:, :, t < 20.]-baseline[:, None, None])) < 1e-8
    old_path = Path(__file__).parent/'all-types-results-34388803/traces.npz'
    with np.load(old_path) as old:
        assert list(old['neuron_types']) == names
        max_original_difference = float(np.max(abs(voltage[:, 1]-old['voltage_mV'])))
        assert max_original_difference < 1e-9

    colors = ['#377eb8', '#333333', '#e41a1c']
    labels = ['0.5x: 10 uA/cm2', '1x: 20 uA/cm2', '2x: 40 uA/cm2']
    ymin = np.floor(voltage.min()/5)*5-2
    ymax = np.ceil(voltage.max()/5)*5+2
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    rows = []
    during = (t >= 20.) & (t < 60.)
    tail = (t >= 55.) & (t < 60.)
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        ax.axvspan(20, 60, color='gray', alpha=.12)
        for j, factor in enumerate(factors):
            ax.plot(t, voltage[i, j], color=colors[j], lw=1.6, label=labels[j])
            local_peak = np.flatnonzero(during)[np.argmax(voltage[i, j, during])]
            rows.append(dict(neuron_type=name, stimulus_factor=factor,
                             current_density_uA_per_cm2=20.*factor,
                             rest_mV=float(baseline[i]), peak_mV=float(voltage[i, j, local_peak]),
                             peak_delta_mV=float(voltage[i, j, local_peak]-baseline[i]),
                             peak_time_ms=float(t[local_peak]),
                             tail_delta_mV=float(np.mean(voltage[i, j, tail])-baseline[i])))
        ax.set(title=name, xlim=(0, 100), ylim=(ymin, ymax), xlabel='Time (ms)', ylabel='Voltage (mV)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper right', framealpha=.85)
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle('Curated point HH: 0.5x / 1x / 2x stimulus in every panel\n'
                 'Pulse: 20-60 ms (gray band) | Same intrinsic initial state per type | No connections | dt=0.025 ms', fontsize=14)
    fig.savefig(out/'all_types_three_intensities.png', dpi=180)
    fig.savefig(out/'all_types_three_intensities.pdf')
    plt.close(fig)
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out/'traces.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ms']+[f'{name}_{factor:g}x_mV' for name in names for factor in factors])
        writer.writerows(np.column_stack([t, voltage.reshape(36, len(t)).T]))
    np.savez_compressed(out/'traces.npz', time_ms=t, neuron_types=np.array(names),
                        stimulus_factors=np.array(factors), voltage_mV=voltage,
                        input_density_uA_per_cm2=np.array(factors)[:, None]*20*pulse)
    sources = [Path(__file__), Path(__file__).with_name('all_types.py'),
               Path(__file__).with_name('sweep.py'), source, old_path,
               args.ripple/'src/ripple/biophysics/channels.py']
    metadata = dict(scope='manual-curated HH intrinsic-current diagnostic; not measured physiology',
                    amplitude_uA_per_cm2=[10., 20., 40.], pulse_on_off_ms=[20., 60.],
                    dt_ms=dt, duration_ms=100., capacitance_uF_per_cm2=1., synapse_count=0,
                    max_difference_original_1x_mV=max_original_difference,
                    all_voltages_and_gates_finite=True, gates_in_0_1=True,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    job_id=os.environ.get('SLURM_JOB_ID'))
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ripple', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
