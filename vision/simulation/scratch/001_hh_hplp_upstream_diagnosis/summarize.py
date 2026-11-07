from pathlib import Path
import os,json,csv,hashlib
import numpy as np
OUT=Path(__file__).resolve().parent
CONTROL=OUT.parent/'002_t4_pathway_runtime_controls'
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
models=[('HH','HH 34427799','#cf493e','-'),('HPLP100','HPLP 100 mbar iterations','#65965a','--'),('HPLP','HPLP reference 34381872','#296bb1','-')]
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,c in zip(axes.flat,['L1','Mi1','Tm3','Mi9']):
 for key,label,color,ls in models:
  with np.load(OUT/(key+'_aligned.npz')) as a:
   t=a['time__right_bright_w1'];y=a[c+'__right_bright_w1'];base=y[(t>=-350)&(t<=-180)].mean()
   ax.plot(t,y-base,label=label,color=color,ls=ls,lw=1.6)
 ax.axhline(0,color='.5',lw=.7);ax.axvline(0,color='.5',lw=.7,ls=':');ax.set_xlim(-180,650);ax.grid(alpha=.16);ax.set_title(c);ax.set_ylabel('Change from baseline (mV)');ax.set_xlabel('Time from local first stimulus (ms)')
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Key missing HH signals: L1 rebound, Mi1/Tm3 undershoot, Mi9 modulation\nSame bright rightward moving-bar task, dt=1 ms; saved parameters, no retraining',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'key_upstream_waveforms.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,model,case,title in zip(axes.flat,['HPLP','HH','HPLP','HH'],['Mi9_frozen','Mi9_frozen','Mi1_Tm3_no_undershoot','Mi4_frozen'],['HPLP: freeze Mi9 output','HH: freeze Mi9 output','HPLP: remove Mi1/Tm3 undershoot','HH: freeze Mi4 output']):
 for casekey,name,color in [('baseline','Original','#296bb1'),(case,'Pathway control','#cf493e')]:
  with np.load(CONTROL/(model+'__'+casekey+'.npz')) as a:
   for token,pref,ls in [('right_bright_w1','PD','-'),('left_bright_w1','ND','--')]:ax.plot(a['time__'+token],a['T4a__'+token],color=color,ls=ls,lw=1.6,label=name+' '+pref)
 ax.axhline(0,color='.5',lw=.7);ax.set_xlim(-180,650);ax.grid(alpha=.16);ax.set_title(title);ax.set_ylabel('T4a response (mV)');ax.set_xlabel('Time from local first stimulus (ms)')
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Pathway controls onto T4a/T4b only, dt=1 ms\nOutput frozen at background or below-background output removed; no retraining',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(CONTROL/'causal_pathway_controls.png',dpi=160);plt.close(fig)
summary=json.loads((OUT/'summary.json').read_text());controls=json.loads((CONTROL/'summary.json').read_text())
lookup={}
for row in csv.DictReader((OUT/'waveform_metrics.csv').open()):lookup[row['model'],row['cell'],row['direction']]=row
fit=[]
for model in ['HH','HPLP100','HPLP']:
 row={'model':model,'dt_ms':1.,'cost':summary['models'][model]['cost']}
 for cell in ['T4a','T4b']:
  pdtoken='right_bright_w1' if cell=='T4a' else 'left_bright_w1';ndtoken='left_bright_w1' if cell=='T4a' else 'right_bright_w1'
  pd=float(lookup[model,cell,pdtoken]['peak_mV']);nd=float(lookup[model,cell,ndtoken]['peak_mV'])
  row[cell+'_PD_peak_mV']=pd;row[cell+'_ND_peak_mV']=nd;row[cell+'_peak_DSI']=(pd-nd)/(pd+nd)
 fit.append(row)
with (OUT/'matched_fit_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(fit[0]));w.writeheader();w.writerows(fit)
inputs=[]
with np.load(CONTROL/'HPLP_input_contributions.npz') as a,np.load(OUT/'HPLP_aligned.npz') as b:
 t=b['time__right_bright_w1'];pre=(t>=-350)&(t<=-180);window=(t>=-150)&(t<=576)
 for c in ['Mi1','Tm3','Mi4','Mi9']:
  y=a[c+'__T4a__right_bright_w1'];base=float(y[pre].mean());inputs.append(dict(source=c,units='mV equivalent drive',baseline=base,delta_min=float(y[window].min()-base),delta_max=float(y[window].max()-base)))
(CONTROL/'hplp_pathway_summary.json').write_text(json.dumps(inputs,indent=2)+'\n')
report='''结论：该 HH 结果最突出的问题是 Mi9→T4 的动态去抑制信号弱，以及 L1/L2 反弹和 Mi1/Tm3 的峰后负相缺失。不能用兴奋幅度普遍不足解释；Mi1/Tm3 的 HH 正峰与 HPLP 相当。

可比性：使用相同网络、亮条右/左两个方向、T4a/T4b 任务；统一 1 ms、读取已保存参数而不重训。HPLP 34381872 源自 300 次 mbar 训练，另加入只有 100 次 mbar 训练的 34381255，并用 HH 的完整 mbar 刺激/监督位置配置评估。100 次 HPLP 原训练仅监督 cost_mid=0，因此这仍不是等初始化、等训练条件的模型能力对照。HPLP 学到的刺激宽度增益 a_sti_mid 也不同。原生 HPLP 2 ms 的成本不能直接与 HH 1 ms 的时间求和成本比较。

波形：亮条右移，按本地刺激对齐及 -350~-180 ms 基线计算。HH/HPLP 参考的 Mi1 正峰分别 14.86/12.46 mV，负峰约 -0.10/-19.52 mV；Tm3 正峰 15.76/14.36 mV、负峰 -0.07/-14.74 mV；L1 反弹约 0.20/7.55 mV；Mi9 负峰 -0.49/-20.68 mV。100 次 HPLP 同样存在反弹与双相波形。HH Mi4 的局部对齐峰约116 ms，HPLP参考约214 ms，亦存在时序差异。

运行时对照：只改变指定来源传入 T4a/T4b 的输出，其余连接不变，均不重训。HPLP 冻结 Mi9 背景输出后，T4a PD/ND 峰由 15.08/7.28 变为4.87/5.10 mV，成本从584649.56升至5860761.0；HH 同样冻结Mi9后成本仅从1740471.88变至1737560.80。HPLP 移除 Mi1/Tm3 低于背景的输出后，成本升至2032584.22且T4负尾消失。HH冻结Mi4输出后成本升至5065325.31，说明当前HH仍依赖Mi4抑制，不能据此建议直接减弱或删除Mi4。HPLP基线重复得到逐点完全相同波形。

机制范围：该HH内核包含Na、K、Leak与电压门控化学突触；没有HPLP的显式慢高通状态。当前训练共用v_th_syn=-44.7447 mV、delta_syn=4 mV、k_minus_syn=0.0445319/ms，未独立为各上游设置释放时间尺度；因此Mi9低电位、Tm3较高背景电位对应不同的有效门控工作点。实际传入T4a的Mi9释放概率均值约0.0166，下降调制仅约0.00010；群体差异存在，不能把平均膜电位代入sigmoid当成平均释放概率。Mi9的HH门控几乎不产生HPLP中那样的有效去抑制调制。49个基因参数全局共享，也使细胞类型间调整相互耦合。

解释限制：HPLP路径冻结证明了其现有解对Mi9和双相兴奋输出的依赖，同时HH冻结实验支持其Mi9调制作用很小。这尚未证明只恢复Mi9就能修复HH；需要后续在独立临时副本中验证Mi9工作点、双相波形及各通路时序的恢复。图中现有DSI是未经基线扣除/归一化的PD电压样本和减ND电压样本和；matched_fit_summary.csv另计算了常用的峰值(PD-ND)/(PD+ND)，两者不能混用。

正式代码、主配置与已有结果均未由本次分析修改。只新增001和002两个编号scratch目录。两项运行均核对了运行期间源文件与已保存参数哈希；最终全局工作区另出现并发产生的run_name/日志/新任务文件变化，本次未修改或撤回。
'''
(OUT/'diagnosis_report.txt').write_text(report)
print(json.dumps(dict(matched_fit=fit,hplp_inputs=inputs),indent=2))
