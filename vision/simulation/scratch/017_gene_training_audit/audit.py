from pathlib import Path
import sys, os, json, hashlib, time
from dataclasses import replace
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
COPY = OUT.parent / '012_adam_resume_diagnosis/copy/vision'
RUN = ROOT / 'vision/simulation/0_runs/point_hh/34437673-continued_optimization-loss-282967'
sys.path[:0] = [str(COPY / 'simulation'), str(COPY)]
sys.modules.pop('import_bootstrap', None)
import import_bootstrap
import numpy as np
import torch
import train
import train.implementation as impl
import train.param as parammod
import train.cost as costmod
import neuron.point_hh as hh
import task.mbar.pack as packmod

torch.set_num_threads(1)
report = {'run': str(RUN), 'checks': {}, 'started_at_unix': time.time()}
def note(name, value):
    report['checks'][name] = value
    (OUT / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(name, json.dumps(value, ensure_ascii=False), flush=True)
def check(name, condition, details=None):
    assert condition, (name, details)
    note(name, details if details is not None else True)

protected_paths = [ROOT / 'vision/simulation/1_neuron/1_4_point_hh.py', ROOT / 'vision/simulation/4_train/1_param.py', ROOT / 'vision/simulation/4_train/2_session.py', ROOT / 'vision/simulation/conf/config.yaml']
protected_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
check('actual_training_model_import', Path(hh.__file__).resolve() == (COPY / 'simulation/1_neuron/1_4_point_hh.py').resolve(), hh.__file__)
check('current_and_training_model_code_match', hashlib.sha256(Path(hh.__file__).read_bytes()).hexdigest() == protected_before[str(protected_paths[0])])
opts = json.loads((RUN / 'data/train_opts.json').read_text())
opts['device'] = 'cpu'
original_build_gt = packmod.build_mbar_gt
packmod.build_mbar_gt = lambda *a, **kw: original_build_gt(*a, **dict(kw, use_cache=False))
session = train.resolve_session(opts)
saved_node_vals, saved_cells, saved_pairs = impl.load_best_node_vals(str(RUN))
session, z = impl.session_z_from_node_vals(saved_node_vals, saved_cells, saved_pairs, session)
check('all_gene_coefficients_shared', all(s['shared'] == list(range(14)) and parammod.param_n_z(s) == 1 for n, s in session.schema.items() if n.startswith('gene_')))
check('direct_conductances_not_trainable', not {'gNa','gK','gLeak','gS'}.intersection(session.schema))
note('trainable_slots', {n: parammod.param_n_z(s) for n,s in session.schema.items() if parammod.param_n_z(s)})
check('62_parameters_49_gene_coefficients', z.numel() == 62 and len([n for n in session.schema if n.startswith('gene_')]) == 49, {'n_z': z.numel(), 'n_gene': 49})
with np.load(RUN / 'data/best_param.npz', allow_pickle=True) as saved:
    check('checkpoint_has_coefficients_not_effective_conductances', all(n in saved.files for n in hh.GENE_PARAM_NAMES) and not {'gNa','gK','gLeak','gS'}.intersection(saved.files))
    check('saved_gene_values_all_finite_and_shared', all(np.isfinite(saved[n]).all() and np.ptp(saved[n]) == 0 for n in hh.GENE_PARAM_NAMES))
    checkpoint_coef = np.array([saved[n][0] for n in hh.GENE_PARAM_NAMES])
provenance = json.loads((RUN / 'data/provenance.json').read_text())
source = Path(provenance['source_run'])
session_source, z_source, adam = impl.load_init_z(str(source), session, resume_adam=True)
check('resume_preserves_context_object', session_source.model_context is session.model_context)
with np.load(source / 'data/best_param.npz', allow_pickle=True) as saved:
    decoded_source = parammod.node_vals_from_z(z_source, session_source.schema)
    check('resume_loads_absolute_gene_weights', all(np.array_equal(decoded_source[n], saved[n]) for n in hh.GENE_PARAM_NAMES))
    note('gene_weights_change_in_final_continuation', {'max_absolute_change': float(np.max(np.abs(checkpoint_coef - np.array([saved[n][0] for n in hh.GENE_PARAM_NAMES])))), 'inherited_adam_iter': adam['iter']})
check('source_and_final_fixed_context_identical', (source / 'data/point_hh_context.json').read_bytes() == (RUN / 'data/point_hh_context.json').read_bytes())
context_json = json.loads((RUN / 'data/point_hh_context.json').read_text())
for name, spec in list(context_json['sources'].items()) + [('expression', context_json['gene_mapping']['expression'])]:
    if isinstance(spec, dict) and 'sha256' in spec:
        check('source_hash_matches_' + name, hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() == spec['sha256'])
ctx = session.model_context
base_before = {p: ctx.params[p].clone() for p in ('gNa','gK','gLeak','gS')}
decoded = parammod.assign_params(z, session.schema, session.connectome)
physical = hh.prepare_training_params(decoded, session)
repeated = hh.prepare_training_params(physical, session)
check('repeated_forward_does_not_compound', all(torch.equal(repeated[p], physical[p]) for p in base_before))
check('context_baseline_not_mutated', all(torch.equal(ctx.params[p], base_before[p]) for p in base_before))
zero = dict(decoded)
zero.update({n: torch.zeros_like(decoded[n]) for n in hh.GENE_PARAM_NAMES})
zero_physical = hh.prepare_training_params(zero, session)
check('zero_gene_weights_recover_curated_baseline', all(torch.equal(zero_physical[p], base_before[p]) for p in base_before))
changed_direct = dict(decoded)
changed_direct.update({p: torch.full_like(base_before[p], 12345) for p in base_before})
ignored = hh.prepare_training_params(changed_direct, session)
check('direct_G_values_ignored_in_gene_mode', all(torch.equal(ignored[p], physical[p]) for p in base_before))
for p in base_before:
    fold_limit = context_json['gene_mapping']['max_fold_changes'][p]
    mask = base_before[p] > 0
    folds = physical[p][mask] / base_before[p][mask]
    check('fold_bounds_' + p, bool(((folds >= 1/fold_limit - 1e-6) & (folds <= fold_limit + 1e-6)).all()), {'min': float(folds.min()), 'max': float(folds.max()), 'allowed': [1/fold_limit, fold_limit]})
check('zero_baseline_edges_remain_zero', bool((physical['gS'][base_before['gS'] == 0] == 0).all()), {'n_zero_edges': int((base_before['gS'] == 0).sum())})

# Exercise the exact training mapping function in float64 on a small independent context.
gen = torch.Generator().manual_seed(17)
incidence = torch.zeros((19,3), dtype=torch.float64)
for i, (p, gene) in enumerate(hh.ION_GENE_LINKS):
    incidence[i, ('gNa','gK','gLeak').index(p)] = 1
small_design = {'ion_features': torch.randn((3,19), generator=gen, dtype=torch.float64), 'ion_to_parameter': incidence, 'edge_features': torch.randn((4,30), generator=gen, dtype=torch.float64), 'log_folds': torch.log(torch.tensor([4.,3.5,3.,6.], dtype=torch.float64))}
small_base = {p: torch.tensor([1.,2.,3.], dtype=torch.float64) for p in ('gNa','gK','gLeak')}
small_base['gS'] = torch.tensor([0.,1.,2.,3.], dtype=torch.float64)
small_ctx = replace(ctx, params=small_base, gene_design=small_design)
small_session = SimpleNamespace(model_context=small_ctx, train_opts=opts, connectome=SimpleNamespace(node_cells=torch.arange(3)))
def evaluate(beta, sess=small_session):
    dec = {n: beta[i:i+1].expand(3) for i,n in enumerate(hh.GENE_PARAM_NAMES)}
    out = hh.prepare_training_params(dec, sess)
    return torch.cat([out[p] for p in ('gNa','gK','gLeak','gS')])
beta = torch.tensor(checkpoint_coef, dtype=torch.float64, requires_grad=True)
check('all_49_mapping_gradients_match_finite_difference', torch.autograd.gradcheck(evaluate, (beta,), eps=1e-6, atol=1e-5, rtol=1e-4))
for value in (-2., 0., 2., -1e4, 1e4):
    v = torch.full((49,), value, dtype=torch.float64, requires_grad=True)
    val = evaluate(v)
    grad = torch.autograd.grad(val.sum(), v)[0]
    check('finite_mapping_output_gradient_beta_' + str(value), bool(torch.isfinite(val).all() and torch.isfinite(grad).all() and (val >= 0).all()))
doubled_ctx = replace(small_ctx, params={p: v*2 for p,v in small_base.items()})
doubled_session = SimpleNamespace(model_context=doubled_ctx, train_opts=opts, connectome=small_session.connectome)
check('doubling_curated_baseline_doubles_output', torch.equal(evaluate(beta, doubled_session), evaluate(beta)*2))
identical_design = dict(small_design, ion_features=torch.zeros_like(small_design['ion_features']))
identical_session = SimpleNamespace(model_context=replace(small_ctx, gene_design=identical_design), train_opts=opts, connectome=small_session.connectome)
identical = evaluate(beta, identical_session)[:3].detach().tolist()
check('identical_expression_different_baselines_produce_different_G', identical == [1.,2.,3.], identical)

note('full_saved_objective_evaluation_started', True)
with torch.no_grad():
    parts = costmod.calc_cost_parts(z, session)
    value = float(costmod.calc_cost(z, session, parts=parts))
expected = provenance['selected_cost']
check('full_saved_loss_reproduced', abs(value - expected) < 0.1, {'reproduced': value, 'saved': expected, 'absolute_difference': abs(value-expected)})
note('full_saved_objective_backward_started', True)
z_grad = z.detach().clone().requires_grad_(True)
backward_cost, backward_parts = costmod.backward_part_sums(z_grad, session)
check('full_objective_gradient_finite', z_grad.grad is not None and bool(torch.isfinite(z_grad.grad).all()), {'cost': backward_cost, 'grad_norm': float(z_grad.grad.double().norm()) if z_grad.grad is not None else None})
check('forward_backward_cost_consistent', abs(backward_cost - value) < max(0.2, abs(value)*1e-6), {'forward': value, 'backward': backward_cost})
gene_grads = {}
for name,spec,start,stop in parammod.schema_params(session.schema):
    if name in hh.GENE_PARAM_NAMES:
        gene_grads[name] = float(z_grad.grad[start])
(OUT / 'gene_loss_gradients.json').write_text(json.dumps(gene_grads, indent=2)+'\n')
note('gene_loss_gradient_summary', {'finite': all(np.isfinite(v) for v in gene_grads.values()), 'n_nonzero': sum(v != 0 for v in gene_grads.values()), 'n_total': len(gene_grads)})
check('protected_formal_files_unchanged', all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in protected_before.items()))
report['completed'] = True
report['elapsed_seconds'] = time.time() - report['started_at_unix']
(OUT / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print('COMPLETE', report['elapsed_seconds'], flush=True)
