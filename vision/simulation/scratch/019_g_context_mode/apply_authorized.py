from pathlib import Path
import json,difflib,hashlib,xml.etree.ElementTree as ET
O=Path(__file__).resolve().parent
R=O.parents[3]
paths=['vision/simulation/1_neuron/1_4_point_hh.py','vision/simulation/conf/config.yaml']
checks=ET.parse(O/'junit_candidate.xml').getroot()
assert all(int(s.get('failures',0))==0 and int(s.get('errors',0))==0 for s in checks.iter('testsuite'))
integration=json.loads((O/'integration_candidate.json').read_text())
assert integration['all_edges_use_init_times_contacts'] and integration['previously_unresolved_edges_enabled']==3309
for name in paths:
    assert (R/name).read_bytes()==(O/'before'/name).read_bytes(),f'File changed since snapshot: {name}'
patch=''
for name in paths:
    source=O/'copy'/name
    old=(R/name).read_text()
    new=source.read_text()
    patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name,tofile=name))
(O/'candidate.patch').write_text(patch)
for name in paths:
    (R/name).write_bytes((O/'copy'/name).read_bytes())
print(patch)
