# 如何在 Jaxley `flyvis` Branch 重現 FlyVis 訓練流程

> 對照 `flyvis/TRAINING_FLOW.md`，說明如何用 Jaxley flyvis branch 取代 FlyVis repo 執行完整訓練。

---

## 完整流程對照表

| FlyVis 元件 | FlyVis 檔案路徑 | Jaxley flyvis branch 對應 | Jaxley 檔案路徑 |
|------------|----------------|---------------------------|----------------|
| `ConnectomeFromAvgFilters` | `flyvis/connectome/connectome.py` | `jx.Network` + `jx.connect()` | `jaxley/modules/network.py` |
| `initialization.py` | `flyvis/network/initialization.py` | `net.set()` + `net.get_parameters()` | `jaxley/modules/base.py` |
| `Network.forward()` | `flyvis/network/network.py` | `step_fn` + `jax.lax.scan` | `jaxley/integrate.py` |
| `Stimulus.add_input()` | `flyvis/network/stimulus.py` | `net.stimulate()` | `jaxley/modules/base.py` |
| `PPNeuronIGRSynapses`（速率 ODE）| `flyvis/network/dynamics.py` | `build_dynamic_state_utils` 橋接層 | `jaxley/utils/dynamics.py` |
| `DecoderGAVP` | `flyvis/task/decoder.py` | 自訂 JAX 解碼層 | —（使用者自行實作） |
| `Task.loss()` | `flyvis/task/tasks.py` | `jax.value_and_grad(loss_fn)` | `jaxley/integrate.py` |
| `MultiTaskSolver.train()` | `flyvis/solver.py` | `optax.adam` + 訓練迴圈 | `optax`（外部套件） |
| `torch.save(chkpt)` | `flyvis/utils/chkpt_utils.py` | `orbax-checkpoint` 或 `pickle` | —（外部套件） |

---

## Step 0：建立網路結構

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/network.py` | `Network.__init__` 接收 `Compartment` 列表（點神經元）；`_append_synapses` 被 `jx.connect()` 呼叫來寫入 edges 表 |
> | `jaxley/synapses/tanh_rate.py` | FlyVis 速率突觸主體；`TanhRateSynapse.compute_current` 實作 ReLU 速率模型（目前用 `relu` 非 `tanh`，可依需求修改） |
> | `jaxley/synapses/tanh_conductance.py` | 含後突觸電導的速率突觸；`compute_current` 回傳 `(linear, constant)` 供 `gather_synapes` 使用 |
> | `jaxley/utils/syn_utils.py` | `gather_synapes` 將各突觸電流加總到對應 compartment；若突觸索引結構改變需同步修改 |

**FlyVis 做的事**：從 connectome JSON 展開 45,669 個點神經元與 1.5M 條突觸（`ConnectomeFromAvgFilters`）。

**Jaxley 對應**：用 `jx.Network` 手動建構，或從 FlyVis connectome 資料轉換。

```python
import jaxley as jx
from jaxley.synapses.tanh_rate import TanhRate  # flyvis branch 的速率突觸

# 每個 cell type 建一個 Compartment（點神經元）
# FlyVis 用點神經元 → 不需要多區室
neurons = [jx.Compartment() for _ in range(n_cell_types)]

# 用 connectome 的 edges 建立突觸連接
# source_index, target_index, n_syn, sign 來自 connectome
net = jx.Network(neurons)
for src, tar, n_syn, sign in zip(source_index, target_index, n_syn_arr, sign_arr):
    jx.connect(net[src], net[tar], TanhRate())
```

> **注意**：FlyVis 用 `tanh` 速率突觸，flyvis branch 已包含 `tanh_conductance.py` 與 `tanh_rate.py`。

---

## Step 1：參數初始化

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/base.py` | `Module.set()` 用來設定初始參數值；`Module.get_parameters()` 取出可訓練參數列表 |
> | `jaxley/synapses/tanh_rate.py` | `synapse_params` dict 定義 `gS`、`count`、`x_offset`、`slope` 的預設值；需對應 FlyVis 的 `syn_strength`、`n_syn`、`sign` 語意 |
> | `jaxley/synapses/tanh_conductance.py` | 同上，另含 `e_syn`（reversal potential） |

**FlyVis 做的事**：`initialization.py` 設定 `time_const`（τ）、`bias`（靜止電位）、`syn_strength`（α）。

```python
# 設定初始參數（對應 FlyVis 論文值）
net.set("time_const", 0.05)              # τ = 50 ms
net.set("bias", 0.5)                     # 靜止電位（可加 N(0, σ²) 雜訊）
net.set("syn_strength", 0.01 / mean_n_syn)  # α = 0.01 / <N>

# sign 和 syn_count 固定不訓練（來自 connectome）
# 透過 param_state 機制固定這些參數
params = net.get_parameters()            # 取得可訓練參數列表
```

| FlyVis 參數 | 初始值 | 是否可訓練 |
|------------|--------|------------|
| `time_const`（τ） | 0.05 s | 是 |
| `bias`（靜止電位） | N(0.5, σ²) | 是 |
| `syn_strength`（α） | 0.01 / ⟨N⟩ | 是 |
| `sign`（σ） | 來自 connectome | **否** |
| `syn_count`（N） | 來自 connectome | **否** |

---

## Step 2：暴露 ODE 介面（flyvis branch 核心）

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/utils/dynamics.py` | **flyvis branch 新增的核心檔案**；`build_dynamic_state_utils` 提供 `full_pytree_to_states`、`states_to_full_pytree`、`states_to_pytree` 三個橋接函數 |
> | `jaxley/integrate.py` | `build_init_and_step_fn` 回傳 `init_fn` 與 `step_fn`；flyvis branch 將 recording 簡化為只記錄電壓 `v` |
> | `jaxley/modules/base.py` | `append_channel_currents_to_states` 被 `states_to_full_pytree` 呼叫，重建電流冗餘量 |

這是 flyvis branch 新增的功能，讓外部 solver 可以接管 Jaxley 的內部狀態。

```python
from jaxley.integrate import build_init_and_step_fn
from jaxley.utils.dynamics import build_dynamic_state_utils

net.to_jax()

# 建立 init / step 函數
init_fn, step_fn = build_init_and_step_fn(
    net,
    voltage_solver="jaxley.dhs",  # 點神經元適用
    solver="bwd_euler"
)

# 建立狀態轉換函數
full_pytree_to_states, states_to_full_pytree, states_to_pytree = \
    build_dynamic_state_utils(net)
```

### 三個轉換函數的作用

| 函數 | 方向 | 說明 |
|------|------|------|
| `full_pytree_to_states(all_states)` | pytree → 扁平向量 | 移除電流、branchpoint 電壓等冗餘量 |
| `states_to_full_pytree(vec, params, dt)` | 扁平向量 → pytree | 重建冗餘量，讓 Jaxley 可執行 `_step()` |
| `states_to_pytree(vec)` | 扁平向量 → pytree | 不重建冗餘量（輕量版） |

---

## Step 3：注入視覺刺激

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/base.py` | `Module.stimulate()` 將外部電流寫入 `self.externals` 與 `self.external_inds`；輸入 shape 需符合 `(T, n_stimulated_comps)` |

**FlyVis 做的事**：`Stimulus.add_input()` 將視覺輸入注入到 R1–R8 光受器。

```python
# stimulus_array shape: (T, n_photoreceptors)
# 注入給 R1-R8 對應的 compartments
net.stimulate(stimulus_array)

externals = net.externals.copy()
external_inds = net.external_inds.copy()
```

---

## Step 4：執行 ODE（含暖機）

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `build_init_and_step_fn` 內的 `_step` 函數執行單步更新；flyvis branch 已將 recording 硬編碼為只記錄 `v`，減少追蹤 overhead |
> | `jaxley/modules/base.py` | `_step` 內部呼叫 `_step_channels`、`_step_synapses`、`_step_pumps`；flyvis branch 移除了 pumped ions / diffusion states 更新 |
> | `jaxley/solver_voltage.py` | flyvis branch 簡化為 `step_voltage_explicit`（`v += dt * constant_terms`），移除軸向電導計算，適合點神經元 |
> | `jaxley/utils/dynamics.py` | `states_to_full_pytree` 在每步開始前重建完整 pytree（由扁平向量還原） |

**FlyVis 做的事**：`Network.forward()` 用 Euler 積分跑 ODE，暖機 500 ms（`t_pre_train = 0.5`）。

```python
import jax
import jax.numpy as jnp

delta_t = 0.02      # 20 ms（對應 FlyVis 50 Hz）
t_pre_ms = 500      # 暖機 500 ms

# 初始化
all_states, all_params = init_fn(params, None, None, delta_t)

# 單步函數（供 scan 使用）
def step_dynamics(all_states, externals_now):
    all_states = step_fn(
        all_states, all_params, externals_now, external_inds, delta_t
    )
    activity = all_states["v"]   # 電壓即速率活動
    return all_states, activity

# 暖機階段（無梯度）
warmup_externals = jnp.zeros((int(t_pre_ms / (delta_t * 1000)), net.num_compartments))
all_states, _ = jax.lax.scan(step_dynamics, all_states, warmup_externals)

# 正式模擬（BPTT 梯度會通過此 scan）
all_states, activities = jax.lax.scan(
    step_dynamics,
    all_states,
    externals,          # shape: (T, n_neurons)
)
# activities shape: (T, n_neurons)
```

---

## Step 5：解碼光流

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `step_fn` 回傳的 `all_states["v"]` 即為各神經元活動；從中取出 T4/T5 index 的切片即可 |
> | —（使用者自行實作）| 解碼器本身不屬於 Jaxley；建議獨立寫成 JAX 函數，與 `loss_fn` 串接以支援 BPTT |

**FlyVis 做的事**：`DecoderGAVP` 對 T4/T5 神經元做 global average pooling 解碼光流。

```python
import jax.numpy as jnp

# 取出 T4/T5 神經元的活動
t4_t5_inds = [...]   # T4/T5 neuron 的 index（來自 connectome nodes）
t4_t5_activity = activities[:, t4_t5_inds]  # (T, n_t4t5)

# 加權解碼（對應 DecoderGAVP 的 global average pooling）
decoder_weights = jnp.array(...)   # 可訓練的解碼器權重
flow_pred = jnp.einsum("ti,ij->tj", t4_t5_activity, decoder_weights)
# flow_pred shape: (T, 2)  ← (x, y) 光流
```

---

## Step 6：損失函數 + BPTT 優化

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `build_init_and_step_fn` 回傳的 `step_fn` 需被包在 `jax.lax.scan` 內；整個 scan 對 `params` 可微，JAX 自動處理 BPTT |
> | `jaxley/utils/dynamics.py` | 若使用 dynamax 外部 solver，`full_pytree_to_states` / `states_to_full_pytree` 需在 `loss_fn` 內呼叫 |
> | —（外部套件 `optax`）| Adam optimizer、學習率 schedule 均用 `optax` 實作，不在 Jaxley 內 |

**FlyVis 做的事**：`MultiTaskSolver.train()` 用 Adam + L2 loss + BPTT（PyTorch `loss.backward()`）。

```python
import optax

# 損失函數（L2 norm，對應 FlyVis tasks.py）
def loss_fn(params, externals, flow_target):
    all_states, all_params = init_fn(params, None, None, delta_t)

    def _step(carry, ext):
        carry = step_fn(carry, all_params, ext, external_inds, delta_t)
        return carry, carry["v"]

    _, activities = jax.lax.scan(_step, all_states, externals)
    flow_pred = decode(activities)   # 你的解碼函數
    return jnp.mean((flow_pred - flow_target) ** 2)

# Adam optimizer（對應 FlyVis 論文設定）
# 學習率 5e-5 → 5e-6（10 步線性衰減）
lr_schedule = optax.linear_schedule(
    init_value=5e-5, end_value=5e-6, transition_steps=10
)
optimizer = optax.adam(
    learning_rate=lr_schedule,
    b1=0.9,
    b2=0.999,
)
opt_state = optimizer.init(params)

# 訓練迴圈（BPTT 由 JAX 自動微分自動處理）
@jax.jit
def train_step(params, opt_state, externals, flow_target):
    loss, grads = jax.value_and_grad(loss_fn)(params, externals, flow_target)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 訓練主迴圈
for iteration in range(n_iterations):
    batch = next(dataloader)   # 取一個 batch（batch_size=4）
    params, opt_state, loss = train_step(
        params, opt_state, batch["externals"], batch["flow"]
    )
    print(f"[{iteration}] loss: {loss:.4f}")
```

---

## Step 7：Checkpoint 儲存

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | —（不需修改 Jaxley）| Jaxley 的 `params` 是標準 JAX pytree，可直接用 `pickle` 或 `orbax-checkpoint` 序列化；無需修改任何 Jaxley 檔案 |

**FlyVis 做的事**：`torch.save(chkpt, path)`

**Jaxley（JAX）對應**：使用 `orbax-checkpoint` 或 `pickle`。

```python
import pickle
from pathlib import Path

chkpt_dir = Path("data/results/my_network/chkpts")
chkpt_dir.mkdir(parents=True, exist_ok=True)

# 儲存
with open(chkpt_dir / f"chkpt_{iteration:05d}.pkl", "wb") as f:
    pickle.dump({"params": params, "opt_state": opt_state, "loss": loss}, f)

# 讀取
with open(chkpt_dir / "chkpt_00100.pkl", "rb") as f:
    chkpt = pickle.load(f)
params = chkpt["params"]
```

---

## 關鍵差異提醒

| 面向 | FlyVis（PyTorch） | Jaxley flyvis branch（JAX） |
|------|-------------------|-----------------------------|
| 自動微分 | `loss.backward()` | `jax.value_and_grad()` |
| 時間迴圈 | Python for-loop | `jax.lax.scan`（JIT 友善） |
| 速率突觸 | `PPNeuronIGRSynapses` | `TanhRate` / `TanhConductance` |
| 電壓求解 | 手動 Euler（`dynamics.py`） | `step_voltage_explicit`（已簡化） |
| 狀態管理 | PyTorch tensor dict | Jaxley pytree ↔ 扁平向量 |
| 優化器 | `torch.optim.Adam` | `optax.adam` |
| Checkpoint | `torch.save` | `orbax-checkpoint` / `pickle` |

---

## 參考資源

- [Jaxley flyvis branch 說明](./FLYVIS_BRANCH.md)
- [FlyVis 訓練流程](../flyvis/TRAINING_FLOW.md)
- [Jaxley 官方文件](https://jaxley.readthedocs.io/en/latest/)
- [FlyVis GitHub](https://github.com/TuragaLab/flyvis)
- [dynamax](https://github.com/probml/dynamax)
- [optax](https://optax.readthedocs.io/)
