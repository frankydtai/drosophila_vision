from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
target = ROOT / 'vision/simulation/4_train/4_optimization.py'
assert target.read_bytes() == (HERE / '4_optimization.candidate.py').read_bytes()
protected = json.loads((HERE / 'protected_hashes.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in protected.items())
assert '105 passed' in (HERE / 'final_verification.log').read_text()
timing = json.loads((HERE / 'benchmark_results.json').read_text())
assert len(timing['records']) == 12 and 'summary' in timing
control = json.loads((HERE / 'deterministic_results.json').read_text())
assert all(control[k] for k in ('same_cost', 'same_gradients', 'same_adam_update'))
assert all(r['grad_finite'] for r in timing['records'] + control['records'])
result = {
    'production_change': str(target),
    'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'preserves_existing_optimizer_edits': True,
    'protected_code_and_config_unchanged': True,
    'correctness_tests_passed': 105,
    'correctness_log': str(HERE / 'final_verification.log'),
    'dsi_definition': 'Arithmetic mean over complete cell/contrast groups of raw PD-minus-ND voltage sums; independent of loss weights',
    'display': 'Cost: ... [weighted cost parts]  DSI: ...',
    'timing': timing['summary'],
    'timing_design': {
        'source_run': timing['source_run'],
        'device': timing['device'],
        'hostname': timing['hostname'],
        'warmups': 2,
        'measurements_each': 6,
        'alternating_order': True,
        'cuda_synchronized': True,
        'same_initial_parameters': True,
        'includes': ['forward', 'backward', 'optional DSI monitor', 'Adam step', 'clamp', 'description formatting'],
        'excludes': ['session load', 'initial/final evaluation', 'checkpoint and file export'],
    },
    'deterministic_gpu_control': {k: control[k] for k in ('same_cost', 'same_gradients', 'same_adam_update')},
    'normal_cuda_note': 'Repeated runs contain floating-point variations; no bitwise equality asserted outside deterministic control.',
    'artifacts': str(HERE),
}
(HERE / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
