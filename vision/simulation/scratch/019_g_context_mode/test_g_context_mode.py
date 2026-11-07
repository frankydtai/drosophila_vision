import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import neuron.point_hh as hh
import config

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
spec=importlib.util.spec_from_file_location('point_hh_before_g_context',OUT/'before/vision/simulation/1_neuron/1_4_point_hh.py')
old=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=old
spec.loader.exec_module(old)
G=('gNa','gK','gLeak','gS')
INIT={'gNa':3.,'gK':1.,'gLeak':1.2,'gS':.3}

@pytest.fixture
def fixture(tmp_path):
    ion=tmp_path/'ion.csv'
    ion.write_text('neuron_type,gNa,gK,gLeak,eLeak,eNa,eK\nR1-6,.003,.0024,.0018,-68,50,-80\nL1,.002,.0009,.0012,-33,50,-80\nL3,.0015,.0008,.001,-35,50,-80\nT4a,.0065,.002,.0016,-60,50,-80\n')
    syn=tmp_path/'syn.csv'
    syn.write_text('pre_type,post_type,nt_type,gS,e_syn,v_th,k_minus,delta\nR1-6,L1,Histamine,.0006,-72,-60,.1,3\nT4a,L1,GABA,.008,-72,-55,.03,3.2\n')
    conn_file=tmp_path/'connections.csv'
    conn_file.write_text('pre_root_id,post_root_id,syn_count,nt_type\n20,10,4,ACH\n20,30,2,ACH\n10,40,3,GLUT\n50,60,5,GABA\n')
    conn=SimpleNamespace(source_idxs=torch.tensor([1,1,0,4]),target_idxs=torch.tensor([0,2,3,5]),w_signed=torch.tensor([-1.,-1.,-1.,-1.],dtype=torch.float64),n_edge=4)
    network=SimpleNamespace(conn=conn,cells=['R1-6','L1','L2','T4b','T4a'],node_cells=torch.tensor([1,0,2,3,4,1]),node_ids=[10,20,30,40,50,60],edge_n_syn=torch.tensor([4.,2.,3.,5.],dtype=torch.float64),edge_source_cells=('R1-6','R1-6','L1','T4a'),edge_target_cells=('L1','L2','T4b','L1'),device='cpu',source_json=tmp_path/'network.json')
    paths=dict(ion_rules_path=ion,synapse_rules_path=syn,connections_path=conn_file)
    opts={'point_hh':{'ion_rules':str(ion),'synapse_rules':str(syn),'connections':str(conn_file),'gene_mode':'off'},'params':{p:{'init':v,'jit':9.,'mode':'indi','inits':{'L1':999.}} for p,v in INIT.items()}}
    return network,paths,opts

def assert_equal(a,b):
    assert a.params.keys()==b.params.keys()
    for p in a.params: torch.testing.assert_close(a.params[p],b.params[p],rtol=0,atol=0)
    assert a.type_rows==b.type_rows
    assert a.edge_rows==b.edge_rows


def test_default_and_explicit_rule_preserve_original_exactly(fixture):
    n,paths,opts=fixture
    original=old.build_point_hh_context(n,**paths)
    default=hh.build_point_hh_context(n,**paths)
    explicit=hh.build_point_hh_context(n,**paths,g_context_mode='rule',g_inits={'gNa':float('nan')})
    assert_equal(original,default)
    assert_equal(original,explicit)
    metadata=dict(default.provenance)
    assert metadata.pop('g_context_mode')=='rule'
    assert metadata==original.provenance
    assert default.params['gS'].tolist()==pytest.approx([2.4,1.2,0.,40.])
    assert [r['assignment'] for r in default.type_rows]==['exact','exact','historical_L2_proxy','direction_proxy','exact']

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_init_uses_shared_nS_values_and_all_contacts(fixture,dtype):
    n,paths,opts=fixture
    n.edge_n_syn=n.edge_n_syn.to(dtype)
    baseline=hh.build_point_hh_context(n,**paths)
    result=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=INIT)
    for p in G[:3]:
        torch.testing.assert_close(result.params[p],torch.full((6,),INIT[p],dtype=dtype))
        assert all(r[p]==INIT[p] for r in result.type_rows)
    torch.testing.assert_close(result.params['gS'],n.edge_n_syn*INIT['gS'])
    assert result.params['gS'][2]>0
    assert result.edge_rows[2]['resolution']=='unresolved_init'
    assert result.edge_rows[2]['gS_nS']==pytest.approx(.9)
    assert result.edge_rows[2]['model_sign']==1
    for p in ('eNa','eK','eLeak','cap','e_syn','v_th_syn','delta_syn','k_minus_syn'):
        torch.testing.assert_close(result.params[p],baseline.params[p],rtol=0,atol=0)
    assert result.provenance['g_inits']==INIT
    assert result.provenance['unresolved_synapse_policy']=='gS=params.gS.init*network_n_syn'


def test_init_conductances_do_not_scale_with_area_or_density(fixture):
    n,paths,opts=fixture
    a=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=INIT)
    b=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=INIT,area_um2=250)
    for p in G: torch.testing.assert_close(a.params[p],b.params[p],rtol=0,atol=0)
    torch.testing.assert_close(b.params['cap'],a.params['cap']*2.5)
    rule=hh.build_point_hh_context(n,**paths,area_um2=250)
    assert rule.params['gNa'][1].item()==pytest.approx(7.5)


def test_cache_tracks_mode_and_init_values_but_not_jitter_or_modes(fixture):
    n,paths,opts=fixture
    rule=hh.build_context(n,opts)
    assert hh.build_context(n,opts) is rule
    opts['point_hh']['g_context_mode']='init'
    init=hh.build_context(n,opts)
    assert init is not rule
    assert hh.build_context(n,opts) is init
    opts['params']['gNa']['init']=7.
    changed=hh.build_context(n,opts)
    assert changed is not init
    assert (changed.params['gNa']==7.).all()
    opts['params']['gNa']['jit']=100.
    opts['params']['gNa']['inits']['L1']=123.
    opts['params']['gNa']['mode']='shared'
    assert hh.build_context(n,opts) is changed
    opts['point_hh']['g_context_mode']='rule'
    assert_equal(hh.build_context(n,opts),rule)

@pytest.mark.parametrize('value',['bad','',None,False,3])
def test_invalid_mode_rejected(fixture,value):
    n,paths,opts=fixture
    with pytest.raises(ValueError,match='g_context_mode'):
        hh.build_point_hh_context(n,**paths,g_context_mode=value)
    opts['point_hh']['g_context_mode']=value
    with pytest.raises(ValueError,match='g_context_mode'): hh.build_context(n,opts)

@pytest.mark.parametrize('parameter',G)
@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),float('-inf'),'invalid',None])
def test_invalid_init_rejected(fixture,parameter,value):
    n,paths,opts=fixture
    values=dict(INIT,**{parameter:value})
    with pytest.raises(ValueError): hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=values)
    opts['point_hh']['g_context_mode']='init'
    opts['params'][parameter]['init']=value
    with pytest.raises(ValueError): hh.build_context(n,opts)

@pytest.mark.parametrize('parameter',G)
def test_missing_init_rejected(fixture,parameter):
    n,paths,opts=fixture
    values=dict(INIT);values.pop(parameter)
    with pytest.raises(ValueError,match='init values'): hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=values)
    opts['point_hh']['g_context_mode']='init'
    del opts['params'][parameter]
    with pytest.raises(ValueError,match='init values'): hh.build_context(n,opts)


def test_all_zero_init_and_zero_contact_count(fixture):
    n,paths,opts=fixture
    zero=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=dict.fromkeys(G,0.))
    assert all((zero.params[p]==0).all() for p in G)
    path=paths['connections_path']
    path.write_text(path.read_text().replace('20,10,4,ACH','20,10,0,ACH'))
    n.edge_n_syn[0]=0
    zero_contact=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=INIT)
    assert zero_contact.params['gS'][0]==0


def test_context_save_records_effective_init_values(fixture,tmp_path):
    n,paths,opts=fixture
    ctx=hh.build_point_hh_context(n,**paths,g_context_mode='init',g_inits=INIT)
    dest=tmp_path/'export'
    hh.save_context(ctx,dest)
    provenance=json.loads((dest/'point_hh_context.json').read_text())
    assert provenance['g_context_mode']=='init' and provenance['g_inits']==INIT
    with (dest/'point_hh_type_params.csv').open() as f: rows=list(csv.DictReader(f))
    assert all(float(r['gNa'])==3 for r in rows)
    with (dest/'point_hh_edge_params.csv').open() as f: rows=list(csv.DictReader(f))
    assert [float(r['gS_nS']) for r in rows]==pytest.approx([1.2,.6,.9,1.5])


def enable_genes(n,opts,tmp_path):
    gene_names=[name.removeprefix('gene_gS_') if name.startswith('gene_gS_') else name.split('_',2)[2] for name in hh.GENE_PARAM_NAMES]
    expr=tmp_path/'expression.csv'
    with expr.open('w') as f:
        f.write(',R1.6,L1,L2,T4b,T4a\n')
        for i,gene in enumerate(gene_names): f.write(gene+','+','.join(str(1+(i+1)*(k+1)/10) for k in range(5))+'\n')
    opts['point_hh']['gene_mode']='on'
    opts['point_hh']['gene']={'version':'v6','expression':str(expr),'max_fold_changes':dict(zip(G,[4.,3.5,3.,6.])),'params':{name:{'lo':-2.,'hi':2.,'init':0.,'jit':0.,'mode':'shared'} for name in hh.GENE_PARAM_NAMES}}

@pytest.mark.parametrize('mode',['rule','init'])
def test_gene_mapping_uses_selected_context_and_gradients(fixture,tmp_path,mode):
    n,paths,opts=fixture
    opts['point_hh']['g_context_mode']=mode
    enable_genes(n,opts,tmp_path)
    ctx=hh.build_context(n,opts)
    sess=SimpleNamespace(model_context=ctx,train_opts=opts,connectome=n)
    def evaluate(coeff):
        dec={name:coeff[i:i+1].expand(5) for i,name in enumerate(hh.GENE_PARAM_NAMES)}
        return hh.prepare_training_params(dec,sess)
    zero=evaluate(torch.zeros(49,dtype=torch.float64))
    for p in G: torch.testing.assert_close(zero[p],ctx.params[p],rtol=0,atol=0)
    coeff=torch.linspace(-.1,.1,49,dtype=torch.float64,requires_grad=True)
    actual=evaluate(coeff)
    design=ctx.gene_design
    for i,p in enumerate(G[:3]):
        terms=[design['ion_features'][:,j]*coeff[j] for j,(target,gene) in enumerate(hh.ION_GENE_LINKS) if target==p]
        fold=torch.exp(torch.log(torch.tensor(opts['point_hh']['gene']['max_fold_changes'][p]))*torch.tanh(sum(terms)))
        torch.testing.assert_close(actual[p],ctx.params[p]*fold[n.node_cells],rtol=1e-7,atol=1e-9)
    signal=design['edge_features']@coeff[19:]
    torch.testing.assert_close(actual['gS'],ctx.params['gS']*torch.exp(np.log(6)*torch.tanh(signal)))
    def flat(c):
        vals=evaluate(c)
        return torch.cat([vals[p] for p in G])
    assert torch.autograd.gradcheck(flat,(coeff,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert ctx.provenance['gene_mapping']['baseline'].startswith('init_' if mode=='init' else 'curated_')
    if mode=='rule':
        original=old.build_context(copy.deepcopy(n),opts)
        before=old.prepare_training_params({name:coeff[i:i+1].expand(5) for i,name in enumerate(hh.GENE_PARAM_NAMES)},SimpleNamespace(model_context=original,train_opts=opts,connectome=n))
        for p in G: torch.testing.assert_close(actual[p],before[p],rtol=0,atol=0)


def test_current_yaml_default_and_config_binding():
    cfg=OmegaConf.to_container(OmegaConf.load(Path(config.__file__).parent/'conf/config.yaml'),resolve=True)
    assert cfg['point_hh']['g_context_mode']=='rule'
    config._bind_config(cfg)
    assert config.POINT_HH['g_context_mode']=='rule'
    cfg['point_hh']['g_context_mode']='init'
    config._bind_config(cfg)
    assert config.POINT_HH['g_context_mode']=='init'
