from pathlib import Path
import os,sys,json
O=Path(__file__).resolve().parent
C=O/'copy/vision'
if os.environ.get('G_CONTEXT_TEST_TARGET')=='formal':
    C=O.parents[3]/'vision'
sys.path[:0]=[str(C/'simulation'),str(C)]
sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import neuron.point_hh as hh
assert Path(hh.__file__).resolve()==(C/'simulation/1_neuron/1_4_point_hh.py').resolve(),hh.__file__
print('TEST_MODEL',hh.__file__,flush=True)
import argparse
parser = argparse.ArgumentParser.parse_known_args
while parser.__name__ == "_parse_known_args_standardize":
    parser = parser.__globals__["_ORIG_PARSE_KNOWN_ARGS"]
argparse.ArgumentParser.parse_known_args = parser
import pytest
phase=os.environ.get('G_CONTEXT_TEST_PHASE','candidate')
suite = C/'simulation/tests' if os.environ.get('G_CONTEXT_FULL_SUITE') else C/'simulation/tests/test_point_hh_neuron.py'
args=[str(O/'test_g_context_mode.py'),str(suite),'-q','-p','no:cacheprovider','--basetemp',str(O/('pytest_tmp_'+phase)),'--junitxml',str(O/('junit_'+phase+'.xml'))]
raise SystemExit(pytest.main(args))
