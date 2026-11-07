from pathlib import Path
import importlib.util
import os
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'vision'))
import import_bootstrap
import task.mbar.pack as original_pack

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

fixtures=load('peak_restore_fixtures',ROOT/'vision/simulation/tests/test_mbar_dsi_cost.py')
before=load('peak_restore_before',HERE/'optimization.before.py')
if os.environ.get('DSI_TEST_PRODUCTION')=='1':
    import train.optimization as optimization
    import figure.mbar as mbar
    pack=original_pack
else:
    pack=load('peak_restore_pack',HERE/'pack.candidate.py')
    original_pack.dsi_peak_values=pack.dsi_peak_values
    optimization=load('peak_restore_optimization',HERE/'optimization.candidate.py')
    mbar=load('peak_restore_mbar',HERE/'mbar.candidate.py')
