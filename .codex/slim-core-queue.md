# Slim core queue

One file per agent turn. Skill: `slim-core-script`.

## Manual (one file, then you prompt again)

```text
用 slim-core-script skill，只處理 queue 裡下一個未完成的檔案。做完就停。
```

## Auto (leave running)

Arm the repository Codex Stop hook:

```bash
touch .codex/slim-core-auto.on
```

Then use the Codex chat that armed it:

```text
用 $slim-core-script，只處理 .codex/slim-core-queue.md 裡下一個未完成的檔案。做完勾選後結束本輪（不要開下一檔；Codex Stop hook 會繼續）。
```

Queue empty → hook deletes `slim-core-auto.on`. Use only one Codex session in this repository while auto mode is armed.

提前停：

```bash
rm -f .codex/slim-core-auto.on
```
## Queue

### 1_neuron

- [x] vision/simulation/1_neuron/1_params.py
- [x] vision/simulation/1_neuron/2_schema.py
- [x] vision/simulation/1_neuron/3_filter_ca.py
- [x] vision/simulation/1_neuron/4_model_borst.py
- [x] vision/simulation/1_neuron/4_model_hp_lp.py
- [x] vision/simulation/1_neuron/5_forward.py
- [x] vision/simulation/1_neuron/6_readout.py

### 2_network

- [x] vision/simulation/2_network/1_path.py
- [x] vision/simulation/2_network/2_connectivity.py
- [x] vision/simulation/2_network/3_construction.py

### 3_task

- [x] vision/simulation/3_task/spot/1_input.py
- [x] vision/simulation/3_task/spot/2_gt.py
- [x] vision/simulation/3_task/moving_bar/1_input.py
- [x] vision/simulation/3_task/moving_bar/2_gt.py

### 4_training

- [x] vision/simulation/4_training/1_config.py
- [x] vision/simulation/4_training/2_readout_pack.py
- [x] vision/simulation/4_training/3_params.py
- [x] vision/simulation/4_training/4_cost.py
- [x] vision/simulation/4_training/5_session.py
- [x] vision/simulation/4_training/6_implement.py
- [x] vision/simulation/4_training/7_experiment.py

### 5_figure

- [x] vision/simulation/5_figure/1_util.py
- [x] vision/simulation/5_figure/2_readout.py
- [x] vision/simulation/5_figure/3_spot.py
- [x] vision/simulation/5_figure/3_moving_bar.py
- [x] vision/simulation/5_figure/4_plot_run.py
- [x] vision/simulation/5_figure/plot_stimulus/spot.py
- [x] vision/simulation/5_figure/plot_stimulus/moving_bar.py

### 6_analyze

- [x] vision/simulation/6_analyze/cell_dynamics.py
- [x] vision/simulation/6_analyze/syn_strength.py
