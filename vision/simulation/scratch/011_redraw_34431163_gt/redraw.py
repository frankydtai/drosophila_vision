from pathlib import Path
import hashlib
import json
import os
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'vision/simulation/0_runs/point_hh/34431163-run_init_from-n_iter-100-tasks-mbar-contrasts-bright-part_cost_scales-mbar-1,mbar_DSI-0'
sys.path.insert(0,str(ROOT/'vision'))
import import_bootstrap
import torch
import train
import figure.plot as plot
import figure.mbar as mbar
import task.mbar.pack as pack_module

torch.set_num_threads(1)
device_label=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
original_gt=pack_module.build_mbar_gt
pack_module.build_mbar_gt=lambda *a,**kw:original_gt(*a,**dict(kw,use_cache=False))
start=time.monotonic()
session,z,best_cost=plot.load_best(str(RUN),verbose=True)
print('LOADED',session.model,session.device,z.numel(),flush=True)
assert session.model=='point_hh'
assert session.train_opts['part_cost_scales']['mbar_DSI']==0
suffix=f'trained, cost {best_cost:.2f} ({session.train_opts["cost_norm"]})'
readouts={contrast:mbar.build_readout(session,z,contrast) for contrast in session.contrasts}
assert set(readouts)=={'bright'}
readout=readouts['bright']
assert set(readout.dsi_by_cell)==set(readout.dsi_peak_by_cell)=={'T4a','T4b'}
assert all(-1<=x<=1 for x in readout.dsi_peak_by_cell.values())
output=HERE/'mbar_gt_v.png'
original_save=mbar.save_figure
panel_titles=[]
def save_and_verify(fig,path,**kwargs):
    titles=[ax.get_title() for ax in fig.axes if ax.axison]
    assert len(titles)==4,titles
    assert all(t.count('DSI=')==1 and t.count('DSI_peak=')==1 for t in titles),titles
    panel_titles.extend(titles)
    original_save(fig,path,**kwargs)
mbar.save_figure=save_and_verify
title,_=mbar.figure_titles(session,suffix,plot.session_filter_figure_token(session),contrast='bright')
mbar.plot_gt(str(output),readouts=readouts,title=title,cost_parts=plot._stored_cost_parts(str(RUN)))
assert output.is_file()
protected=json.loads((HERE/'run_hashes.before.json').read_text())
for path,digest in protected.items():
    assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==digest,path
result={
    'source_run':str(RUN),'saved_cost':best_cost,
    'device':device_label,'hostname':os.uname().nodename,
    'DSI':readout.dsi_by_cell,'DSI_peak':readout.dsi_peak_by_cell,
    'panel_titles':panel_titles,'elapsed_s':time.monotonic()-start,
    'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
    'saved_training_files_and_other_figures_unchanged':True,
}
(HERE/'redraw_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
