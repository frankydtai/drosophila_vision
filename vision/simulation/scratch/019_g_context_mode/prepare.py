from pathlib import Path
import shutil, hashlib, json, difflib, subprocess
O=Path(__file__).resolve().parent
R=O.parents[3]
S=R/'vision/simulation'
C=O/'copy/vision'
C.mkdir(parents=True)
shutil.copy2(R/'vision/import_bootstrap.py',C/'import_bootstrap.py')
for name in ('1_neuron','2_network','3_task','4_train','5_figure','analyze','conf','tests'):
    shutil.copytree(S/name,C/'simulation'/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copy2(S/'config.py',C/'simulation/config.py')
files=['vision/simulation/1_neuron/1_4_point_hh.py','vision/simulation/conf/config.yaml']
original={name:(R/name).read_text() for name in files}
for name in files:
    target=O/'before'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(R/name,target)
protected={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in ('1_neuron','2_network','3_task','4_train','5_figure','analyze','conf') for p in (S/parent).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')}
(O/'source_hashes_before.json').write_text(json.dumps(protected,indent=2)+'\n')
(O/'git_diff_before.patch').write_text(subprocess.check_output(['git','diff','--'],cwd=R,text=True))
p=C/'simulation/1_neuron/1_4_point_hh.py'
s=p.read_text()
def edit(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
edit('    capacitance_uF_cm2: float = 1.0,\n) -> PointHHContext:', '    capacitance_uF_cm2: float = 1.0,\n    g_context_mode: str = "rule",\n    g_inits: Mapping[str, float] | None = None,\n) -> PointHHContext:')
edit('    ion_path = Path(ion_rules_path).expanduser().resolve()', '''    g_context_mode = str(g_context_mode).lower()
    if g_context_mode not in ("rule", "init"):
        raise ValueError("point_hh.g_context_mode must be rule|init")
    if g_context_mode == "init":
        try:
            g_inits = {
                name: float(g_inits[name]) for name in ("gNa", "gK", "gLeak", "gS")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("init G context requires gNa/gK/gLeak/gS init values") from exc
        if any(not np.isfinite(value) or value < 0 for value in g_inits.values()):
            raise ValueError("init G context conductances must be finite and non-negative")
    ion_path = Path(ion_rules_path).expanduser().resolve()''')
edit('            value = density * area_um2 * 10.0 if column.startswith("g") else density', '''            value = density * area_um2 * 10.0 if column.startswith("g") else density
            if g_context_mode == "init" and column.startswith("g"):
                value = g_inits[column]''')
edit('        g_s_nS = float(values["gS"]) * 1000.0 * float(edge_n_syn[edge_index])', '''        g_s_nS = float(values["gS"]) * 1000.0 * float(edge_n_syn[edge_index])
        if g_context_mode == "init":
            g_s_nS = g_inits["gS"] * float(edge_n_syn[edge_index])''')
edit('        "mapping_version": "point_hh_curated_v1",', '        "mapping_version": "point_hh_curated_v1",\n        "g_context_mode": g_context_mode,')
edit('    return PointHHContext(\n        params=params,', '''    if g_context_mode == "init":
        provenance["g_inits"] = dict(g_inits)
        provenance["contact_scaling"] = "gS_nS = params.gS.init * network_n_syn"
        provenance["unresolved_synapse_policy"] = "gS=params.gS.init*network_n_syn"
    return PointHHContext(
        params=params,''')
edit('        "baseline": "curated_conductances_with_contact_counts_already_applied",', '''        "baseline": (
            "init_conductances_with_contact_counts_already_applied"
            if context.provenance.get("g_context_mode") == "init" else
            "curated_conductances_with_contact_counts_already_applied"
        ),''')
edit('    cache_key = json.dumps(point_hh, sort_keys=True)', '''    g_inits = None
    if str(point_hh.get("g_context_mode", "rule")).lower() == "init":
        entries = opts.get("params") or {}
        g_inits = {
            name: (entries.get(name) or {}).get("init")
            for name in ("gNa", "gK", "gLeak", "gS")
        }
    cache_key = json.dumps((point_hh, g_inits), sort_keys=True)''')
edit('            capacitance_uF_cm2=point_hh.get("capacitance_uF_cm2", 1.0),\n        )', '''            capacitance_uF_cm2=point_hh.get("capacitance_uF_cm2", 1.0),
            g_context_mode=point_hh.get("g_context_mode", "rule"),
            g_inits=g_inits,
        )''')
p.write_text(s)
p=C/'simulation/conf/config.yaml'
s=p.read_text()
assert s.count('  gene_mode: "on"\n')==1
s=s.replace('  gene_mode: "on"\n','  gene_mode: "on"\n  # G baseline: curated rule tables or params.*.init (nS; gS per contact).\n  g_context_mode: "rule"\n')
p.write_text(s)
patch=''
for name in files:
    new=(C/'simulation'/Path(name).relative_to('vision/simulation')).read_text()
    patch+=''.join(difflib.unified_diff(original[name].splitlines(True),new.splitlines(True),fromfile=name,tofile=name))
(O/'candidate.patch').write_text(patch)
print(patch)
