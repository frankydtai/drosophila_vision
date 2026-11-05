"""Isolated CPU diagnostic of Paper 1 StableHH; no network/training integration.

Run with the existing comp_neuro interpreter on an allocated CPU node.
All dt values are actual integration steps, without hidden substeps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


DT_VALUES = [.1, .2, .3]
END_MS = 100.
ON_MS, OFF_MS = 20., 60.
REF_STEP_MS = .005


def rates(v):
    def vtrap(x, scale):
        z = x / scale
        return scale * (1 - z / 2 + z*z / 12) if abs(z) < 1e-7 else x / np.expm1(z)
    alpha = np.array([.1*vtrap(-(v+40), 10), .07*np.exp(-(v+65)/20),
                      .01*vtrap(-(v+55), 10)])
    beta = np.array([4*np.exp(-(v+65)/18), 1/(np.exp(-(v+35)/10)+1),
                     .125*np.exp(-(v+65)/80)])
    return alpha, beta


def ionic(v, gates, p):
    m, h, n = gates
    # S/cm2 * 1000 -> mS/cm2; current in uA/cm2.
    return 1000 * (p['gNa']*m**3*h*(v-p['eNa'])
                   + p['gK']*n**4*(v-p['eK']) + p['gLeak']*(v-p['eLeak']))


def initial_state(p):
    def residual(v):
        a, b = rates(v)
        return ionic(v, a/(a+b), p)
    v = brentq(residual, -90., -45., xtol=1e-13)
    a, b = rates(v)
    y = np.r_[v, a/(a+b)]
    assert abs(residual(v)) < 1e-9
    return y


def reference(p, amplitude, y0, rtol=1e-10, atol=1e-12, max_step=.02):
    """Split exactly at pulse discontinuities, with continuous carried state."""
    pieces = []
    state = y0.copy()
    for start, end, drive in [(0., ON_MS, 0.), (ON_MS, OFF_MS, amplitude),
                              (OFF_MS, END_MS, 0.)]:
        def rhs(t, y):
            a, b = rates(y[0])
            return np.r_[(drive-ionic(y[0], y[1:], p)), a*(1-y[1:])-b*y[1:]]
        sol = solve_ivp(rhs, (start, end), state, method='DOP853',
                        rtol=rtol, atol=atol, max_step=max_step, dense_output=True)
        assert sol.success, sol.message
        state = sol.y[:, -1]
        pieces.append((start, end, sol))
    def evaluate(ts):
        ts = np.asarray(ts)
        result = np.empty((4, len(ts)))
        for i, (start, end, sol) in enumerate(pieces):
            mask = (ts >= start) & ((ts < end) if i < 2 else (ts <= end + 1e-9))
            result[:, mask] = sol.sol(ts[mask])
        return result
    return evaluate


def crossings(t, v):
    inds = np.flatnonzero((v[:-1] < 0.) & (v[1:] >= 0.))
    return t[inds] - v[inds]*(t[inds+1]-t[inds])/(v[inds+1]-v[inds])


def run(args):
    sys.path.insert(0, str(args.ripple / 'src'))
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from ripple.biophysics.channels import StableHH
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert jax.default_backend() == 'cpu'
    assert jax.config.read('jax_enable_x64')
    table_path = args.ripple / 'configs/parameters/t4t5/curated/v1/ion_channel_rules.csv'
    with table_path.open() as handle:
        table = {r['neuron_type']: {k: float(v) for k, v in r.items() if k != 'neuron_type'}
                 for r in csv.DictReader(handle)}
    classical = dict(gNa=.12, gK=.036, gLeak=.0003, eNa=50., eK=-77., eLeak=-54.3)
    cases = [('paper1_T4a', table['T4a'], 20.), ('paper1_Mi1', table['Mi1'], 20.),
             ('classical_HH_stress', classical, 10.)]
    # Verify independent reference rate equations against the actual source class.
    for v in [-90., -65., -55., -40., 0., 40.]:
        actual = np.asarray([np.asarray(g(jnp.asarray(v))) for g in
                             (StableHH.m_gate, StableHH.h_gate, StableHH.n_gate)])
        a, b = rates(v)
        np.testing.assert_allclose(actual[:, 0], a, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[:, 1], b, rtol=1e-12, atol=1e-12)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fine_t = np.linspace(0., END_MS, round(END_MS/REF_STEP_MS)+1)
    rows, all_arrays, reference_checks = [], {'reference_time_ms': fine_t}, {}
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(.05, .95, len(DT_VALUES)))
    selected = set(DT_VALUES)
    for case_i, (name, p, amplitude) in enumerate(cases):
        y0 = initial_state(p)
        ref = reference(p, amplitude, y0)
        ref_tight = reference(p, amplitude, y0, rtol=2e-12, atol=2e-14, max_step=.01)
        ref_v = ref(fine_t)[0]
        ref_disagreement = float(np.max(np.abs(ref_v-ref_tight(fine_t)[0])))
        assert ref_disagreement < 1e-5, ref_disagreement
        ref_spikes = crossings(fine_t, ref_v)
        reference_checks[name] = dict(max_tight_reference_difference_mV=ref_disagreement,
                                     initial_state=y0.tolist(), parameters=p,
                                     input_uA_per_cm2=amplitude,
                                     reference_spike_times_ms=ref_spikes.tolist())
        all_arrays[name+'_reference_v_mV'] = ref_v
        axes[case_i, 0].plot(fine_t, ref_v, color='black', lw=2, label='DOP853 reference')
        ref_peak_t = float(fine_t[np.argmax(ref_v)])
        zoom_start = max(ON_MS-2., ref_peak_t-2.)
        zoom_end = min(OFF_MS+4., ref_peak_t+8.)
        axes[case_i, 1].plot(fine_t, ref_v, color='black', lw=2, label='Reference')
        for dt, color in zip(DT_VALUES, colors):
            cell = jx.Cell()
            cell.insert(StableHH(name='HH'))
            for key, val in p.items():
                cell.set('HH_'+key, val)
            cell.set('capacitance', 1.)
            cell.set('v', y0[0])
            for key, val in zip(('m', 'h', 'n'), y0[1:]):
                cell.set('HH_'+key, val)
            # Physical cylinder membrane area; uA/cm2 -> nA.
            area_um2 = float(2*np.pi*cell.nodes.iloc[0]['radius']*cell.nodes.iloc[0]['length'])
            for state in ['v', 'HH_m', 'HH_h', 'HH_n']:
                cell.record(state, verbose=False)
            started = time.perf_counter()
            # Preserve exact physical pulse edges at 20 and 60 ms. For .3 ms,
            # shorten only the last step in each segment; never shift the pulse.
            times, records, carried = [0.], [], None
            for start, end, density in [(0., ON_MS, 0.), (ON_MS, OFF_MS, amplitude),
                                        (OFF_MS, END_MS, 0.)]:
                steps = int(np.floor((end-start)/dt + 1e-9))
                remainder = (end-start) - steps*dt
                chunks = [(dt, steps)]
                if remainder > 1e-9:
                    chunks.append((remainder, 1))
                elapsed_segment = 0.
                for step_size, n_steps in chunks:
                    cell.delete_stimuli()
                    cell.stimulate(jnp.full(n_steps, density*area_um2*1e-5), verbose=False)
                    chunk, carried = jx.integrate(cell, delta_t=step_size, solver='bwd_euler',
                                                  all_states=carried, return_states=True)
                    chunk = np.asarray(chunk)
                    assert chunk.shape[1] == n_steps+1
                    if not records:
                        records.append(chunk[:, :1])
                    records.append(chunk[:, 1:])
                    times.extend(start+elapsed_segment+np.arange(1, n_steps+1)*step_size)
                    elapsed_segment += n_steps*step_size
            rec = np.concatenate(records, axis=1)
            t = np.asarray(times)
            assert abs(t[-1]-END_MS) < 1e-9
            assert np.all(np.diff(t) > 0.)
            assert np.min(abs(t-ON_MS)) < 1e-9 and np.min(abs(t-OFF_MS)) < 1e-9
            elapsed = time.perf_counter()-started
            v = rec[0]
            finite = bool(np.isfinite(rec).all())
            gates_ok = bool(finite and np.min(rec[1:]) >= -1e-10 and np.max(rec[1:]) <= 1+1e-10)
            assert np.isclose(v[0], y0[0], atol=1e-12)
            dense_v = np.interp(fine_t, t, v)
            error = dense_v-ref_v
            node_error = v-ref(t)[0]
            spikes = crossings(t, v) if finite else np.array([])
            timing = (float(np.mean(np.abs(spikes-ref_spikes)))
                      if len(spikes) == len(ref_spikes) and len(spikes) else None)
            row = dict(case=name, dt_ms=dt, finite=finite, gates_in_0_1=gates_ok,
                       rmse_common_grid_mV=float(np.sqrt(np.mean(error**2))),
                       rmse_at_solver_nodes_mV=float(np.sqrt(np.mean(node_error**2))),
                       max_abs_error_mV=float(np.max(np.abs(error))),
                       peak_voltage_mV=float(np.max(v)), reference_peak_mV=float(np.max(ref_v)),
                       peak_error_mV=float(np.max(v)-np.max(ref_v)),
                       peak_time_ms=float(t[np.argmax(v)]), reference_peak_time_ms=ref_peak_t,
                       peak_time_error_ms=float(t[np.argmax(v)]-ref_peak_t),
                       nrmse_percent_of_reference_range=float(100*np.sqrt(np.mean(error**2))/np.ptp(ref_v)),
                       spike_count=len(spikes), reference_spike_count=len(ref_spikes),
                       mean_spike_time_error_ms=timing,
                       runtime_including_compile_s=elapsed)
            rows.append(row)
            token = name+'_dt_'+str(dt).replace('.', 'p')
            all_arrays[token+'_time_ms'] = t
            all_arrays[token+'_voltage_mV'] = v
            if dt in selected:
                label = f'{dt:g} ms'
                axes[case_i, 0].plot(t, v, lw=1, color=color, alpha=.85, label=label)
                axes[case_i, 1].plot(t, v, '.-', ms=3, lw=1, color=color, label=label)
            print(json.dumps(row), flush=True)
        case_rows = [r for r in rows if r['case'] == name]
        axes[case_i, 2].loglog(DT_VALUES, [r['rmse_common_grid_mV'] for r in case_rows], 'o-', label='RMSE, common grid')
        axes[case_i, 2].loglog(DT_VALUES, [r['rmse_at_solver_nodes_mV'] for r in case_rows], 's--', label='RMSE, solver nodes')
        axes[case_i, 2].loglog(DT_VALUES, [r['max_abs_error_mV'] for r in case_rows], '^:', label='Max absolute error')
        axes[case_i, 0].set_title(f'{name}: {amplitude:g} uA/cm2, 20-60 ms')
        axes[case_i, 1].set_title('Peak / onset detail')
        axes[case_i, 1].set_xlim(zoom_start, zoom_end)
        axes[case_i, 2].set_title('Error against adaptive reference')
        for ax in axes[case_i, :2]:
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('Voltage (mV)')
            ax.axvspan(ON_MS, OFF_MS, color='gray', alpha=.08)
        axes[case_i, 2].set_xlabel('Actual integration step (ms)')
        axes[case_i, 2].set_ylabel('Voltage error (mV)')
        for ax in axes[case_i]:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Point HH: 0.1 / 0.2 / 0.3 ms | Jaxley backward Euler voltage + exponential gates | float64\n'
                 'Exact pulse edges at 20/60 ms; only boundary steps shortened for 0.3 ms. Reference: DOP853.', fontsize=13)
    fig.savefig(out/'waveforms_and_accuracy.png', dpi=170)
    fig.savefig(out/'waveforms_and_accuracy.pdf')
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'traces.npz', **all_arrays)
    # Original .1/.2 ms runs must agree despite segmented continuation.
    original = Path(__file__).parent/'results-34382548/metrics.csv'
    with original.open() as f:
        previous = {(r['case'], float(r['dt_ms'])): r for r in csv.DictReader(f)}
    max_old_difference = 0.
    for row in rows:
        if row['dt_ms'] in (.1, .2):
            old = previous[(row['case'], row['dt_ms'])]
            for metric in ['rmse_common_grid_mV', 'peak_voltage_mV', 'peak_time_ms']:
                difference = abs(row[metric]-float(old[metric]))
                max_old_difference = max(max_old_difference, difference)
                assert difference < 1e-8, (row['case'], row['dt_ms'], metric, difference)
    paths = [Path(__file__), table_path, args.ripple/'src/ripple/biophysics/channels.py']
    meta = dict(scope='isolated single-cell numerical diagnostic, not biological/task accuracy',
                solver='actual Jaxley bwd_euler, exponential gates; nominal .1/.2/.3 ms; boundary-clipped final steps only',
                pulse_timing='20-60 ms exactly for every nominal dt; continuous full-state carry between segments',
                max_difference_from_original_01_02=max_old_difference,
                reference='SciPy DOP853, piecewise at 20/60 ms, rtol=1e-10 atol=1e-12 max_step=.02 ms',
                common_grid_error='linear interpolation of coarse output to uniform .005 ms grid; includes undersampling',
                node_error='coarse voltages compared to dense adaptive solution at identical solver times',
                reference_checks=reference_checks, backend=jax.default_backend(),
                versions={k: importlib.metadata.version(k) for k in ['jax', 'jaxlib', 'jaxley', 'numpy', 'scipy']},
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
    (out/'provenance.json').write_text(json.dumps(meta, indent=2))
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ripple', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
