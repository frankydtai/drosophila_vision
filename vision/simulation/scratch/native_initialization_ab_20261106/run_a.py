"""A: call the project's unmodified forward_v initialization, then native step."""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch

ROOT = Path('/home/yitai/drosophila_vision/vision/simulation')
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals, session_z_from_node_vals
from neuron import point_hh
from neuron.forward import forward_v, pack_t_onset

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources = [ROOT/'1_neuron/1_4_point_hh.py', ROOT/'1_neuron/4_forward.py',
           ROOT/'4_train/1_param.py', ROOT/'4_train/2_session.py', ROOT/'conf/config.yaml']
before = {str(p): sha(p) for p in sources}
for run_id in ('34400581', '34401154'):
    started = time.monotonic()
    run = next((ROOT/'0_runs/point_hh').glob(run_id+'-*'))
    dest = OUT/run_id
    dest.mkdir(exist_ok=True)
    opts = json.loads((run/'data/train_opts.json').read_text())
    # Device only; retain saved task, pre_steady, precision and all scientific options.
    opts['device'] = 'cpu'
    assert opts['tasks'] == ['mbar'] and opts['contrasts'] == ['bright']
    assert opts['mbar_sti_opts']['delta_ms'] == opts['mbar_sti_opts']['delta_ms_pre'] == 1.
    with torch.no_grad():
        session = train.resolve_session(opts)
        session, z = session_z_from_node_vals(*load_best_node_vals(str(run)), session)
        params = train.params_from_z(z, session)
        pack = session.primary_pack
        onset = pack_t_onset(pack)
        drive = session.pack_i_sti(pack)
        if drive.ndim == 2:
            drive = drive.unsqueeze(0)
        assert onset == 324
        print('A START', run_id, 'input', tuple(drive.shape), 'onset', onset, flush=True)
        captured, calls = {}, []
        def profile(frame, event, arg):
            if event != 'return':
                return
            if frame.f_code is point_hh.pre_steady.__code__:
                captured['seed'] = {k:v[0:1].clone() for k,v in arg.items()}
                calls.append('neuron.point_hh.pre_steady')
            elif frame.f_code is point_hh.step.__code__:
                current = frame.f_locals['i_sti'][0:1]
                if 'background' not in captured:
                    captured['background'] = current.clone()
                assert torch.equal(current, captured['background']), 'pre-period is not constant background'
                captured['state'] = {k:v[0:1].clone() for k,v in arg.items()}
                captured['steps'] = captured.get('steps', 0) + 1
        # Observe returns only: no monkeypatch, replacement, or copied solver.
        sys.setprofile(profile)
        try:
            pre_trace = forward_v(session, params, drive[:, :onset+1], pack=pack)
        finally:
            sys.setprofile(None)
        assert captured['steps'] == onset
        assert torch.equal(pre_trace[0,-1], captured['state']['v'][0])
        initial = captured['state']
        background = captured['background']
        pp = point_hh.prepare_params(params, background, session)
        N = int(session.connectome.n_node)
        src = session.connectome.conn.source_idxs.cpu().numpy()
        post = session.connectome.conn.target_idxs.cpu().numpy()
        E = len(src)
        inputs = {}
        for key in ('cap','gNa','gK','gLeak','eNa','eK','eLeak'):
            inputs[key] = np.broadcast_to(pp[key].cpu().numpy().reshape(-1), (N,)).copy()
        for key in ('gS','e_syn','v_th_syn','delta_syn','k_minus_syn'):
            inputs[key] = np.broadcast_to(pp[key].cpu().numpy().reshape(-1), (E,)).copy()
        nc = session.connectome.node_cells.cpu().numpy()
        inputs.update(background_pA=background[0].cpu().numpy(), source=src, target=post,
                      node_cell=nc, cells=np.array(session.connectome.cells),
                      u=np.asarray(session.connectome.us), v_coordinate=np.asarray(session.connectome.vs))
        # Contains parameters/topology/input only. B may read this file; no A states.
        np.savez_compressed(dest/'matched_inputs.npz', **inputs)
        np.savez_compressed(dest/'A_seed.npz', **{k:v[0].cpu().numpy() for k,v in captured['seed'].items()})
        np.savez_compressed(dest/'A_initial.npz', **{k:v[0].cpu().numpy() for k,v in initial.items()})
        state = initial
        voltage = np.empty((1201,N), dtype=np.float32)
        voltage[0] = state['v'][0].cpu().numpy()
        for i in range(1,1201):
            state = point_hh.step(state, pp, background, session, delta_ms=1.)
            voltage[i] = state['v'][0].cpu().numpy()
        assert np.isfinite(voltage).all()
        np.savez_compressed(dest/'A_observation.npz', time_ms=np.arange(1201), voltage_mV=voltage)
        manifest = dict(run=str(run), method='saved-project pre_steady -> original forward_v pre-period',
            pre_steady=opts['pre_steady'], n_iter=opts['pre_steady_n_iter'], damping=opts['pre_steady_damp'],
            dt_ms=1., physical_pre_steps=onset, physical_pre_ms=onset, observation_steps=1200,
            fp=opts['fp'], nodes=N, edges=E, native_calls=calls+['neuron.forward.forward_v','neuron.point_hh.step'],
            no_function_replacement=True, source_sha256=before,
            saved_inputs_sha256={str(p):sha(p) for p in (run/'data/train_opts.json',run/'data/best_param.npz')},
            source_unchanged=all(sha(p)==h for p,h in before.items()), elapsed_s=time.monotonic()-started)
        assert manifest['source_unchanged']
        (dest/'A_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('A DONE', run_id, 'nodes',N,'edges',E,'seconds',manifest['elapsed_s'],flush=True)
