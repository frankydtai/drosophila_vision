from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,difflib
O=Path(__file__).resolve().parent
R=O.parents[3]
before=json.loads((O/'source_hashes_before.json').read_text())
changed=[p for p,h in before.items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
expected=['vision/simulation/1_neuron/1_4_point_hh.py','vision/simulation/conf/config.yaml']
assert sorted(changed)==sorted(expected),(changed,expected)
for p in expected: assert (R/p).read_bytes()==(O/'copy'/p).read_bytes()
root=ET.parse(O/'junit_formal.xml').getroot()
suites=list(root.iter('testsuite'))
assert all(int(s.get('failures',0))==0 and int(s.get('errors',0))==0 for s in suites)
integration=json.loads((O/'integration_formal.json').read_text())
assert integration['all_edges_use_init_times_contacts'] and integration['cache_init_change']
patch=''
for p in expected:
    patch+=''.join(difflib.unified_diff((O/'before'/p).read_text().splitlines(True),(R/p).read_text().splitlines(True),fromfile=p,tofile=p))
assert patch==(O/'candidate.patch').read_text()
report={'changed_formal_files':changed,'other_source_files_unchanged':True,'applied_patch_matches_tested_copy':True,'tests_passed':sum(int(s.get('tests',0)) for s in suites),'default_mode':'rule','init_gS_includes_previously_unmatched_connections':True,'real_network':integration,'training_started':False}
(O/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
