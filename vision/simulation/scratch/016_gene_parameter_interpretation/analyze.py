import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

root = Path('/home/yitai/drosophila_vision')
run = root / 'vision/simulation/0_runs/point_hh/34437673-continued_optimization-loss-282967'
out = Path(__file__).resolve().parent
context = json.loads((run / 'data/point_hh_context.json').read_text())
gm = context['gene_mapping']
expr_path = Path(gm['expression']['path'])
assert hashlib.sha256(expr_path.read_bytes()).hexdigest() == gm['expression']['sha256']

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (out / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

param = read_csv(run / 'param.csv')
coeff = {k: float(param[0][k]) for k in gm['coefficient_names']}
assert all(float(row[k]) == coeff[k] for row in param for k in coeff)
expression = {row['']: row for row in read_csv(expr_path)}
scaler = gm['feature_scaler']
centers = dict(zip(scaler['genes'], scaler['centers']))
scales = dict(zip(scaler['genes'], scaler['scales']))
features = {}
for row in param:
    cell = row['cell']
    source = 'R1.6' if cell == 'R1-6' else cell
    for gene in scaler['genes']:
        raw = float(expression[gene][source])
        features[cell, gene] = (math.log1p(raw) - centers[gene]) / scales[gene]

ion_links = [(name, name.split('_', 2)[1], name.split('_', 2)[2]) for name in coeff if not name.startswith('gene_gS_')]
ion_rows, contribution_rows = [], []
for base in read_csv(run / 'data/point_hh_type_params.csv'):
    cell = base['neuron_type']
    result = {'cell': cell}
    for p in ('gNa', 'gK', 'gLeak'):
        terms = []
        for name, target, gene in ion_links:
            if target != p:
                continue
            x = features[cell, gene]
            contribution = coeff[name] * x
            terms.append(contribution)
            contribution_rows.append({'cell': cell, 'parameter': p, 'gene': gene, 'coefficient': coeff[name], 'feature': x, 'signal_contribution': contribution})
        signal = math.fsum(terms)
        multiplier = math.exp(math.log(gm['max_fold_changes'][p]) * math.tanh(signal))
        result.update({p + '_baseline_nS': float(base[p]), p + '_signal': signal, p + '_fold': multiplier, p + '_effective_nS': float(base[p]) * multiplier})
    ion_rows.append(result)
write_csv('ion_effective_conductances.csv', ion_rows)
write_csv('ion_gene_contributions.csv', contribution_rows)

common = ('brp', 'Rim', 'Liprin-alpha', 'Syt1', 'Syt7', 'Sytbeta', 'nSyb', 'Syx1A', 'Snap25', 'Csp', 'cpx', 'shi', 'Syn', 'Syp', 'Vap33')
specific = {
    'ACH': (('ChAT', 'VAChT'), ('nAChRalpha1', 'nAChRalpha2', 'nAChRbeta1')),
    'GABA': (('Gad1', 'VGAT'), ('Rdl', 'Lcch3', 'Grd')),
    'Histamine': (('Hdc',), ('ort', 'HisCl1')),
    'GLUT': (('VGlut',), ('GluClalpha',)),
}
syn_rows = []
groups = defaultdict(list)
for edge in read_csv(run / 'data/point_hh_edge_params.csv'):
    pre, post, nt = edge['pre_type'], edge['post_type'], edge['model_nt_type']
    signal = 0.0
    if nt in specific:
        pre_genes, post_genes = specific[nt]
        signal = math.fsum(coeff['gene_gS_' + g] * features[pre, g] for g in common + pre_genes)
        signal += math.fsum(coeff['gene_gS_' + g] * features[post, g] for g in post_genes)
    fold = math.exp(math.log(gm['max_fold_changes']['gS']) * math.tanh(signal))
    base = float(edge['gS_nS'])
    syn_rows.append({'edge_index': edge['edge_index'], 'pre_type': pre, 'post_type': post, 'nt': nt, 'n_syn': edge['n_syn'], 'baseline_nS': base, 'signal': signal, 'fold': fold, 'effective_nS': base * fold})
    groups[pre, post, nt, base > 0].append((fold, base, base * fold))
write_csv('synapse_effective_conductances.csv', syn_rows)
pairs = []
for (pre, post, nt, active), vals in groups.items():
    folds = [v[0] for v in vals]
    assert max(folds) - min(folds) < 1e-12
    pairs.append({'pre_type': pre, 'post_type': post, 'nt': nt, 'nonzero_baseline': active, 'n_edges': len(vals), 'fold': folds[0], 'baseline_sum_nS': sum(v[1] for v in vals), 'effective_sum_nS': sum(v[2] for v in vals)})
write_csv('synapse_pair_folds.csv', pairs)
report = {'source_run': str(run), 'coefficient_precision': 'param.csv rounded to six decimals; reconstructed folds are approximate', 'expression_sha256_verified': True, 'shared_coefficients_verified': True, 'ion': ion_rows, 'selected_synapse_pairs': [r for r in pairs if r['post_type'].startswith('T4') and r['pre_type'] in ('Mi1','Tm3','Mi4','Mi9') and r['nonzero_baseline']], 'coefficients': coeff}
(out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
