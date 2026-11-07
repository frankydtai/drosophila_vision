from pathlib import Path
import difflib
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
files={
    'pack':ROOT/'vision/simulation/3_task/4_mbar/4_pack.py',
    'optimization':ROOT/'vision/simulation/4_train/4_optimization.py',
    'mbar':ROOT/'vision/simulation/5_figure/2_4_mbar.py',
}
sources={name:p.read_text() for name,p in files.items()}
for name,source in sources.items():
    (HERE/f'{name}.before.py').write_text(source)
historical=subprocess.check_output(['git','show','b0f21fb9:vision/simulation/3_task/4_mbar/4_pack.py'],cwd=ROOT,text=True)
old=historical[historical.index('def dsi_cost_parts('):historical.index('\ndef _mbar_cost_part_plot_specs(')]
(HERE/'historical_peak_loss.py').write_text(old)
peak=old.replace('def dsi_cost_parts(', 'def dsi_peak_values(',1)
start=peak.index('    """')
end=peak.index('    """',start+7)+7
peak=peak[:start]+'''    """Normalized peaks of baseline-relative, aligned PD/ND mean traces.

    Average each direction over selected entries on its common transit window,
    take its nonnegative peak, then return (R_PD-R_ND)/(R_PD+R_ND+eps).
    """'''+peak[end:]
peak=peak.replace('    # hex_dsi_window has one transit duration before first stimulus and two\n    # after it: length = 3 * duration + 1 (both endpoints included).\n    before = (pack.dsi_n_ts - 1) // 3', '    # Current hex_dsi_window spans [-T, 3T], including both endpoints.\n    before = (pack.dsi_n_ts - 1) // 4')
peak=peak.replace('        parts[key] = (dsi - 1.0).square()', '        parts[key] = dsi')
assert 'square()' not in peak and '// 3' not in peak
peak=peak.rstrip()+'\n\n\n'
sources['pack']=sources['pack'].replace('\ndef _mbar_cost_part_plot_specs(', '\n'+peak+'def _mbar_cost_part_plot_specs(',1)
s=sources['optimization']
s=s.replace('from task.mbar.pack import dsi_groups, dsi_values', 'from task.mbar.pack import dsi_groups, dsi_peak_values, dsi_values',1)
s=s.replace('from train.cost import forward_pack, session_cost_part_keys', 'from train.cost import forward_pack, gt_affine_from_pack, session_cost_part_keys',1)
s=s.replace('    """Mean raw moving-bar DSI across cell/contrast groups, independent of loss scales."""\n    values = []', '    """Mean raw DSI and normalized DSI_peak, independent of loss scales."""\n    values = []\n    peak_values = []',1)
s=s.replace('            values.extend(float(value.item()) for value in dsi_values(pack, trace).values())\n    return sum(values) / len(values) if values else None', '''            values.extend(float(value.item()) for value in dsi_values(pack, trace).values())
            _, bias_gt = gt_affine_from_pack(params, pack, session)
            peak_values.extend(
                float(value.item())
                for value in dsi_peak_values(pack, trace, bias_gt).values()
            )
    return (
        (sum(values) / len(values), sum(peak_values) / len(peak_values))
        if values else None
    )''',1)
s=s.replace('    return "" if dsi is None else f"  DSI: {dsi:.4f}"', '''    if dsi is None:
        return ""
    value, peak = dsi
    return f"  DSI: {value:.4f}  DSI_peak: {peak:.4f}"''',1)
sources['optimization']=s
s=sources['mbar']
s=s.replace('import train\n', 'import train\nfrom train.cost import gt_affine_from_pack\n',1)
s=s.replace('    dsi_groups,\n    dsi_values,', '    dsi_groups,\n    dsi_peak_values,\n    dsi_values,',1)
s=s.replace('    dsi_window_ts: dict = field(default_factory=dict)\n', '    dsi_window_ts: dict = field(default_factory=dict)\n    dsi_peak_by_cell: dict = field(default_factory=dict)\n',1)
s=s.replace('    trace = as_numpy(trace)\n    specs = bar_specs_from_task(session, task, contrast)', '''    _, bias_gt = gt_affine_from_pack(params, pack, session)
    dsi_peak_by_cell = {
        key.removeprefix(f"mbar_DSI_{contrast}_"): float(value)
        for key, value in dsi_peak_values(pack, trace, bias_gt).items()
    }
    trace = as_numpy(trace)
    specs = bar_specs_from_task(session, task, contrast)''',1)
s=s.replace('        dsi_by_cell=dsi_by_cell,\n', '        dsi_by_cell=dsi_by_cell,\n        dsi_peak_by_cell=dsi_peak_by_cell,\n',1)
s=s.replace("                cell_title += f'\\nDSI={row_readout.dsi_by_cell[cell]:.3f}'", "                cell_title += f'\\nDSI={row_readout.dsi_by_cell[cell]:.3f}'\n            if cell in row_readout.dsi_peak_by_cell:\n                cell_title += f'  DSI_peak={row_readout.dsi_peak_by_cell[cell]:.3f}'",1)
sources['mbar']=s
for name,source in sources.items():
    before=(HERE/f'{name}.before.py').read_text()
    assert source!=before,name
    compile(source,f'{name}.candidate.py','exec')
    (HERE/f'{name}.candidate.py').write_text(source)
    (HERE/f'{name}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),source.splitlines(True),fromfile=str(files[name]),tofile=f'{name}.candidate.py')))
protected=[ROOT/'vision/simulation/4_train/3_cost.py',ROOT/'vision/simulation/conf/config.yaml',ROOT/'vision/simulation/run.slurm',ROOT/'vision/simulation/run_init_from.slurm']
(HERE/'protected_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2)+'\n')
print('Prepared three candidate files; production files untouched.')
