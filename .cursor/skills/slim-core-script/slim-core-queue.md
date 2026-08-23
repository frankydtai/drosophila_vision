# Slim core queue

One file per agent turn. Skill: `slim-core-script`.

## Manual (one file, then you prompt again)

```text
用 slim-core-script skill，只處理 queue 裡下一個未完成的檔案：內聯單次使用 local（刪 assignment，把 RHS 放進唯一使用點）；禁止新造名詞（lexicon 以外）。做完就停。
```

## Auto (leave running)

Bind **one** conversation (file contents = conversation id):

```bash
echo '75e2cb58-9549-4af9-9438-df5b689fdee4' > .cursor/slim-core-auto.on
```

Then use **that** Agent chat only:

```text
用 slim-core-script skill，只處理 .cursor/skills/slim-core-script/slim-core-queue.md 裡下一個未完成的檔案：內聯單次使用 local；禁止新造名詞（僅 lexicon / 檔內既有名）。做完勾選後結束本輪（不要開下一檔；auto hook 會繼續）。
```

Other windows are ignored while auto is armed. Queue empty → hook deletes `slim-core-auto.on`.

提前停：

```bash
rm -f .cursor/slim-core-auto.on
```
## Queue
### 1_neuron

- [x] vision/simulation/1_neuron/1_1_borst.py
- [x] vision/simulation/1_neuron/1_2_hp_lp.py
- [x] vision/simulation/1_neuron/2_filter_ca.py
- [x] vision/simulation/1_neuron/3_schema.py
- [x] vision/simulation/1_neuron/4_forward.py
- [x] vision/simulation/1_neuron/5_readout.py

### 2_network

- [x] vision/simulation/2_network/1_path.py
- [x] vision/simulation/2_network/2_connectivity.py
- [x] vision/simulation/2_network/3_construction.py

### 3_task

- [x] vision/simulation/3_task/1_spread/1_sti_spec.py
- [x] vision/simulation/3_task/1_spread/2_gt.py
- [x] vision/simulation/3_task/1_spread/3_pack.py
- [x] vision/simulation/3_task/2_spot/1_sti_geo.py
- [x] vision/simulation/3_task/2_spot/2_sti_spec.py
- [x] vision/simulation/3_task/2_spot/3_gt.py
- [x] vision/simulation/3_task/2_spot/4_pack.py
- [x] vision/simulation/3_task/3_sbar/1_sti_geo.py
- [x] vision/simulation/3_task/4_mbar/1_sti_geo.py
- [x] vision/simulation/3_task/4_mbar/2_sti_spec.py
- [x] vision/simulation/3_task/4_mbar/3_gt.py
- [x] vision/simulation/3_task/4_mbar/4_pack.py

### 4_train

- [x] vision/simulation/4_train/1_param.py
- [x] vision/simulation/4_train/2_session.py
- [x] vision/simulation/4_train/3_cost.py
- [x] vision/simulation/4_train/4_optimization.py
- [x] vision/simulation/4_train/5_implementation.py

### 5_figure

- [x] vision/simulation/5_figure/1_panel.py
- [ ] vision/simulation/5_figure/2_1_spread.py
- [ ] vision/simulation/5_figure/2_2_spot.py
- [ ] vision/simulation/5_figure/2_3_mbar.py
- [ ] vision/simulation/5_figure/3_plot.py
- [ ] vision/simulation/5_figure/plot_sti/spot.py
- [ ] vision/simulation/5_figure/plot_sti/mbar.py

### 6_analyze

- [ ] vision/simulation/6_analyze/cell_dynamics.py
- [ ] vision/simulation/6_analyze/cost_part.py
- [ ] vision/simulation/6_analyze/trace.py
- [ ] vision/simulation/6_analyze/syn_sign.py
- [ ] vision/simulation/6_analyze/syn_strength.py
