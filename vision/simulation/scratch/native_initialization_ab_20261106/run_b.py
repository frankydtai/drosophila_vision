"""B: native Ripple cold start and native fixed-threshold settling; no A states."""
from pathlib import Path
import sys, json, hashlib, time, inspect
import numpy as np
import pandas as pd

REPO = Path('/home/yitai/Ripple')
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO/'src')]
from ripple.core.runtime import configure_jax_runtime
configure_jax_runtime()
import jax
import jax.numpy as jnp
import jaxley as jx
from ripple.models.whole_cns.point import build_point_network
from ripple.biophysics.channels import StableHH
from experiments.paper3_gene_mapping.point_hh_shared25 import SharedPointGeneMap
from experiments.paper3_gene_mapping.full_optic_shared25_physiology import native_engines
from experiments.paper3_gene_mapping.fixed_threshold_settle import fixed_threshold_settle
from experiments.paper3_gene_mapping.shared25_memory_engine import execution_plan

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
functions = [build_point_network, StableHH, SharedPointGeneMap, native_engines,
             fixed_threshold_settle, execution_plan, jx.integrate]
sources = {inspect.getsourcefile(f) for f in functions}
sources.add(str(REPO/'experiments/paper3_gene_mapping/shared25_memory_engine.py'))
sources.add(str(Path(inspect.getsourcefile(jx.integrate)).parent/'modules/base.py'))
before = {p:sha(p) for p in sorted(sources)}
for run_id in ('34400581','34401154'):
    started = time.monotonic()
    dest = OUT/run_id
    # Deliberately never read A_seed, A_initial, or A_observation.
    with np.load(dest/'matched_inputs.npz',allow_pickle=False) as f:
        p = {k:f[k] for k in f.files}
    N, E = len(p['cap']), len(p['source'])
    print('B START',run_id,'backend',jax.default_backend(),'nodes',N,'edges',E,flush=True)
    edges = pd.DataFrame(dict(body_pre=p['source'],body_post=p['target'],weight=np.ones(E)))
    net, _ = build_point_network(body_ids=list(range(N)),edges=edges,scale_gs_by_weight=False)
    net.insert(StableHH(name='HH'))
    # Convert total nS/pF/pA to Jaxley's S/cm2, uF/cm2, uS and nA.
    # Keep native point geometry; conversion uses its actual membrane area.
    area = (2*np.pi*net.nodes.radius*net.nodes.length).to_numpy(float)
    net.nodes['capacitance'] = p['cap']/(area*.01)
    for key in ('gNa','gK','gLeak'):
        net.nodes['HH_'+key] = p[key]/(area*10.)
    for key in ('eNa','eK','eLeak'):
        net.nodes['HH_'+key] = p[key].astype(float)
    net.nodes['v'] = net.nodes.HH_eLeak.to_numpy(float)
    for target, source in [('gS','gS'),('e_syn','e_syn'),('v_th','v_th_syn'),
                           ('delta','delta_syn'),('k_minus','k_minus_syn')]:
        net.edges['Syn_'+target] = p[source].astype(float)/(1000. if target=='gS' else 1.)
    np.testing.assert_array_equal(net.edges.body_pre,p['source'])
    np.testing.assert_array_equal(net.edges.body_post,p['target'])
    audit = {}
    for key in ('gNa','gK','gLeak'):
        back = net.nodes['HH_'+key].to_numpy()*area*10.
        np.testing.assert_allclose(back,p[key],rtol=1e-14,atol=1e-14)
        audit[key+'_roundtrip_max_abs'] = float(np.max(abs(back-p[key])))
    np.testing.assert_allclose(net.nodes.capacitance.to_numpy()*area*.01,p['cap'],rtol=1e-14)
    np.testing.assert_allclose(net.edges.Syn_gS.to_numpy()*1000.,p['gS'],rtol=1e-14)
    # Original residual mapper at beta == reference: physical parameters are
    # exactly the supplied references. No gene intervention or fitting is run.
    # Designs are inactive input arrays, not a substitute solver or controller.
    mapper = SharedPointGeneMap(reference=np.zeros(25),
        hh_reference=net.nodes[['HH_gNa','HH_gK','HH_gLeak']].to_numpy(float),
        gs_reference=net.edges.Syn_gS.to_numpy(float),
        ion_design=np.zeros((1,14)),node_type=np.zeros(N,dtype=int),
        edge_design=np.zeros((1,11)),edge_group=np.zeros(E,dtype=int),
        release_design=np.zeros((1,3)),role_logit=np.zeros(1),protected_p=np.full(E,np.nan),
        delta=net.edges.Syn_delta.to_numpy(float),pre_index=p['source'],post_index=p['target'],
        e_syn=net.edges.Syn_e_syn.to_numpy(float),synapse_name='Syn',threshold_policy='fixed_absolute')
    beta = jnp.zeros(25)
    theta = jnp.asarray(net.edges.Syn_v_th.to_numpy(float))
    hh,gs,th,_ = mapper.parameters(beta,theta,beta,xp=np)
    np.testing.assert_array_equal(hh,mapper.hh_reference)
    np.testing.assert_array_equal(gs,mapper.gs_reference)
    np.testing.assert_array_equal(th,p['v_th_syn'])
    # Exactly the cold-start call used by train_c0_legacy_types.run.
    # In this installed Jaxley t_max=0 still performs one dt update; retain it.
    net.record('v',verbose=False)
    np.savez_compressed(dest/'B_builder_seed.npz',v=net.nodes.v.to_numpy(float),
        HH_m=net.nodes.HH_m.to_numpy(float),HH_h=net.nodes.HH_h.to_numpy(float),
        HH_n=net.nodes.HH_n.to_numpy(float),Syn_s=net.edges.Syn_s.to_numpy(float))
    _, cold = jx.integrate(net,delta_t=1.,t_max=0.,return_states=True)
    np.savez_compressed(dest/'B_cold_after_native_bootstrap.npz',**{k:np.asarray(v) for k,v in cold.items()})
    config = dict(dt_ms=1.,settle_duration_ms=100.,duration_ms=1200.)
    record = np.arange(N)
    release_edges = np.unique(np.linspace(0,E-1,min(E,32),dtype=int))
    forward,_,settle = native_engines(net,mapper,np.arange(N),record,release_edges,
        config,{**config,'long_training':{'endpoint_policy':'visible'}})
    background = jnp.asarray(p['background_pA'].astype(float)/1000.)
    current = jnp.repeat(background[:,None],101,axis=1)
    state, diagnostic = fixed_threshold_settle(beta=beta,states=cold,threshold=theta,mapper=mapper,
        settle=settle,current=current,max_blocks=80,voltage_min=-120.,voltage_max=50.)
    jax.block_until_ready(state)
    np.savez_compressed(dest/'B_initial.npz',**{k:np.asarray(v) for k,v in state.items()})
    (dest/'B_settle_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print('B SETTLE',run_id,diagnostic['status'],'blocks',diagnostic['blocks'],flush=True)
    # Diagnostic observation of the returned state; do not relabel a failed
    # settle as a valid equilibrated state or proceed to training.
    current_obs = jnp.repeat(background[:,None],1201,axis=1)
    raw,safety,final = forward(beta,state,theta,beta,current_obs)
    raw = np.asarray(raw)
    assert np.isfinite(raw).all()
    np.savez_compressed(dest/'B_observation.npz',time_ms=np.arange(raw.shape[1]),
        voltage_mV=raw[:N].T,safety=np.asarray(safety))
    hh2,gs2,th2,_ = mapper.parameters(beta,theta,beta,xp=np)
    for a,b in ((hh,hh2),(gs,gs2),(th,th2)):
        np.testing.assert_array_equal(a,b)
    plan = execution_plan(100.,1.,'visible')
    manifest = dict(method='Ripple native cold-start -> native_engines -> fixed_threshold_settle',
        initialization_callsite=str(REPO/'experiments/paper3_gene_mapping/train_c0_legacy_types.py')+':200',
        no_A_state_read=True,no_Point_HH_import=True,no_function_replacement=True,
        native_functions={f.__name__:inspect.getsourcefile(f) for f in functions},
        mapper='original SharedPointGeneMap; inactive residual designs; beta == reference; parameters audited unchanged',
        dt_ms=1.,fp=64,backend=jax.default_backend(),bootstrap_steps=1,settle_execution_plan=plan,
        physical_pre_steps=1+diagnostic['blocks']*plan['physical_steps'],
        physical_pre_ms=1+diagnostic['blocks']*plan['physical_final_time_ms'],
        status=diagnostic['status'],converged=diagnostic['converged'],blocks=diagnostic['blocks'],
        nodes=N,edges=E,observation_steps=raw.shape[1]-1,unit_audit=audit,
        matched_inputs_sha256=sha(dest/'matched_inputs.npz'),source_sha256=before,
        source_unchanged=all(sha(p)==h for p,h in before.items()),elapsed_s=time.monotonic()-started)
    assert manifest['source_unchanged']
    (dest/'B_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('B DONE',run_id,'seconds',manifest['elapsed_s'],flush=True)
