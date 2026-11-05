from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
expected=json.loads((root/'code_hashes.json').read_text())
changed=[p for p,h in expected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
if changed:raise SystemExit('Code changed since experiment setup: '+', '.join(changed))
print('Experiment source hashes verified.',flush=True)
