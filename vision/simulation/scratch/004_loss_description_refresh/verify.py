import ast
import hashlib
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/yitai/drosophila_vision')
HERE = Path(__file__).resolve().parent
TARGET = ROOT / 'vision/simulation/4_train/4_optimization.py'
sys.path.insert(0, str(ROOT / 'vision/.venv/lib/python3.9/site-packages'))
from tqdm import tqdm
from tqdm._monitor import TMonitor

tqdm.monitor_interval = 0
before = ast.parse((HERE / '4_optimization.before.py').read_text())
after = ast.parse(TARGET.read_text())

def gradient(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gradient_network')

old_loop = next(n for n in gradient(before).body if isinstance(n, ast.For))
new_loop = next(n for n in gradient(after).body if isinstance(n, ast.For))
old_condition = old_loop.body[-1]
assert isinstance(old_condition, ast.If)
assert len(old_condition.body) == 1
old_loop.body[-1:] = old_condition.body
assert ast.dump(before) == ast.dump(after), 'Unrelated source changes'
assert TARGET.read_bytes() == (HERE / '4_optimization.candidate.py').read_bytes()
compile(TARGET.read_bytes(), str(TARGET), 'exec')

# Execute the actual production formatting helpers, tqdm constructor, and
# description-update statement; omit model computation and optimizer work.
namespace = {}
helpers = [n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == '_fmt_cost_parts']
exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])), '<production formatter>', 'exec'), namespace)
constructor = next(n for n in gradient(after).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'progress_bar' for t in n.targets))
constructor_code = compile(ast.Expression(body=constructor.value), '<production tqdm constructor>', 'eval')
update_code = compile(ast.fix_missing_locations(ast.Module(body=[new_loop.body[-1]], type_ignores=[])), '<production description update>', 'exec')
records = []

class RecordingTqdm(tqdm):
    def display(self, *args, **kwargs):
        records.append((self.n, self.desc))
        return super().display(*args, **kwargs)

class MonitorEvent:
    def __init__(self):
        self.waits = 0
    def wait(self, interval):
        self.waits += 1
    def is_set(self):
        return self.waits > 1

summaries = []

def run_case(name, n_iter, duration, monitor=False, parts=True):
    records.clear()
    stream = io.StringIO()
    now = [time.time()]
    cached = {'mbar_test': 1000.0, 'spot_mid+0': 2000.0, 'hidden': 3000.0}
    env = dict(namespace, tqdm=RecordingTqdm, sys=SimpleNamespace(stderr=stream), n_iter=n_iter, cost=999.0, initial_parts=cached if parts else None, _TQDM_REFRESH_INTERVAL=10)
    bar = eval(constructor_code, env)
    bar._time = lambda: now[0]
    bar.start_t = bar.last_print_t = now[0]
    if monitor:
        fake_monitor = SimpleNamespace(_time=lambda: now[0], was_killed=None, sleep_interval=10, tqdm_cls=SimpleNamespace(get_lock=lambda: threading.RLock()), get_instances=lambda: [bar])
    switched = False
    for i in bar:
        now[0] += duration
        if monitor:
            fake_monitor.was_killed = MonitorEvent()
            TMonitor.run(fake_monitor)
            switched = switched or bar.miniters == 1
        cached.update(mbar_test=1000.0 + i + 1, **{'spot_mid+0': 2000.0 + i + 1})
        env.update(progress_bar=bar, cost=float(i + 1), float_last_parts=(lambda order: dict(cached)) if parts else None, task_order=None)
        size = len(stream.getvalue())
        exec(update_code, env)
        assert len(stream.getvalue()) == size, 'Description update wrote an extra log line'
        assert bar.desc.startswith('Cost: %.4f' % (i + 1))
    positive = [(n, desc) for n, desc in records if n > 0]
    for n, desc in positive:
        assert desc.startswith('Cost: %.4f' % n), (name, n, desc)
        if parts:
            assert 'mbar_test=%.4f' % (1000 + n) in desc
            assert 'spot_mid+0=%.4f' % (2000 + n) in desc
            assert 'hidden' not in desc
    if monitor:
        assert switched
        assert {n for n, desc in positive} == set(range(5, n_iter + 1))
        assert bar.miniters == 1
    else:
        expected = set(range(10, n_iter + 1, 10))
        if n_iter:
            expected.add(n_iter)
        assert {n for n, desc in positive} == expected, (name, positive)
        assert bar.miniters == 10
    summaries.append({'case': name, 'iterations': n_iter, 'emitted_iteration_numbers': sorted(set(n for n, desc in positive)), 'passed': True})
    (HERE / (name + '.log')).write_text(stream.getvalue())

for count in (0, 1, 9, 10, 11, 25):
    run_case('fast_%d' % count, count, 0.2)
run_case('slow_monitor_switch', 15, 12.0, monitor=True)
run_case('without_parts', 11, 0.2, parts=False)
original_hashes = json.loads((HERE / 'existing_file_hashes.json').read_text())
for path, expected in original_hashes.items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
result = {'passed': True, 'checks': ['Only removed the description-update iteration condition and its indentation', 'Production source compiles', 'No extra writes from refresh=False', 'Printed totals and visible parts match completed iterations', '10-iteration schedule preserved', 'Actual tqdm monitor switches miniters to 1 after 60 seconds', 'Zero, short, boundary, and missing-parts cases', 'Existing modified config and run files unchanged'], 'cases': summaries}
(HERE / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
