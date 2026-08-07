# Jaxley 通用建模工作流程

> 適用於**任意 connectome、任意神經元模型（生物物理 / 速率模型）、任意任務**。
> 步驟編號與格式對應 `JAXLEY_FLYVIS_INTEGRATION.md`。

---

## 完整流程對照表

| 步驟 | 任務 | 核心 Jaxley 檔案 |
|------|------|------------------|
| Step 0 | Connectivity（建立網路拓撲與突觸連接）| `modules/network.py`、`modules/compartment.py`、`connect.py` |
| Step 1 | 神經元模型定義 ODE（Channel / Synapse）| `channels/channel.py`、`synapses/synapse.py`、`utils/syn_utils.py` |
| Step 2 | 參數初始化（固定 / 可訓練 / 共享）| `modules/base.py`、`utils/cell_utils.py` |
| Step 3 | 注入外部輸入（刺激 / 感覺輸入 / clamp）| `modules/base.py`、`stimulus.py` |
| Step 4 | 執行 ODE（含暖機，`init_fn` / `step_fn` / `scan`）| `integrate.py`、`utils/dynamics.py`、`modules/base.py`、`solver_voltage.py`、`utils/jax_utils.py` |
| Step 5 | 任務輸出（解碼 / 讀出層）| `integrate.py`（取 `all_states["v"]`）、使用者自訂 |
| Step 6 | 損失函數 + BPTT 優化 | `integrate.py`、`utils/dynamics.py`、`optax`（外部）|
| Step 7 | Checkpoint 儲存 | —（不需修改 Jaxley）|

---

## Step 0：Connectivity（建立網路拓撲與突觸連接）

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/compartment.py` | 點神經元基本單元；預設 `length=10μm`、`radius=1μm`、`capacitance=1μF/cm²`、`v=-70mV` |
> | `jaxley/modules/branch.py` | 多個 compartment 串成一條 branch，用於多區室樹狀神經元 |
> | `jaxley/modules/cell.py` | 多條 branch 組成完整形態細胞 |
> | `jaxley/modules/network.py` | 多個 Cell 組成網路；`_append_synapses` 被 `jx.connect()` 呼叫寫入 edges 表；`vectorize_cells` 控制 GPU 向量化 |
> | `jaxley/modules/base.py` | `insert(Channel)` 將通道插入神經元 |
> | `jaxley/connect.py` | `jx.connect(pre, post, SynapseType())` 建立突觸連接，寫入 edges 表 |

**說明**：只處理網路拓撲與突觸連接，不涉及 ODE 內容。根據 connectome 資料選擇拓撲後組裝 Network。

**拓撲選擇指引**：

| 拓撲 | 建構元件 | 適用情境 |
|------|---------|----------|
| 點神經元 | `Compartment` | 大規模 connectome（萬級神經元）|
| 單分支多區室 | `Branch` | 簡單樹突結構 |
| 完整多區室樹 | `Cell`（含多 `Branch`）| 精確形態神經元 |
| 混合大網路 | `Network`（含多種 `Cell`）| 任意規模 |

```python
import jaxley as jx

# 點神經元網路（大規模 connectome）
neurons = [jx.Compartment() for _ in range(n_neurons)]
net = jx.Network(neurons)

# 從 connectome edges 建立突觸連接（Step 1 再 insert Channel/Synapse）
for src, tar in zip(edges["source_index"], edges["target_index"]):
    jx.connect(net[src], net[tar], MySynapse())  # MySynapse 在 Step 1 定義

# 多區室網路（精確形態）
cell_a = jx.Cell()
cell_b = jx.Cell()
net2 = jx.Network([cell_a, cell_b])
jx.connect(net2[0], net2[1], MySynapse())
```

---

## Step 1：神經元模型定義 ODE

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/channels/channel.py` | **Channel 基底類別**；`update_states` 定義通道閘控 ODE；`compute_current` 計算膜電流（`mA/cm²`）；需設 `current_is_in_mA_per_cm2 = True` |
> | `jaxley/channels/hh.py` | 現成 HH 通道（Na/K/Leak）；`update_states` 實作 Hodgkin-Huxley ODE |
> | `jaxley/channels/non_capacitive/rate.py` | 速率型通道（無電容，適合速率模型）|
> | `jaxley/channels/non_capacitive/izhikevich.py` | Izhikevich 棘波近似 ODE |
> | `jaxley/synapses/synapse.py` | **Synapse 基底類別**；`update_states` 定義突觸閘控 ODE；`compute_current` 計算突觸電流（`nA`）|
> | `jaxley/synapses/ionotropic.py` | 現成離子型突觸；`update_states` 實作閘控狀態 `s` 的 ODE |
> | `jaxley/synapses/tanh_rate.py` | 速率突觸（**無狀態**，無 ODE）；`compute_current` 用 ReLU 計算速率電流 |
> | `jaxley/synapses/tanh_conductance.py` | 含後突觸電導的速率突觸；`compute_current` 回傳 `(linear, constant)` 供電壓求解器使用 |
> | `jaxley/utils/syn_utils.py` | `gather_synapes` 在每個時間步將所有突觸電流加總到對應 compartment |

**說明**：Channel 的 `update_states` 與 `compute_current` 就是 ODE 右手邊（RHS），在 Step 5 每個時間步被呼叫。

**模型類型選擇指引**：

| 模型類型 | 適用情境 | 推薦 Channel | 推薦 Synapse |
|---------|---------|-------------|-------------|
| 生物物理（HH 型）| 棘波精確動力學 | `HH` | `IonotropicSynapse` |
| 速率模型 | 大規模 connectome 訓練 | `rate.py` | `TanhRateSynapse` / `TanhConductanceSynapse` |
| Izhikevich | 輕量棘波近似 | `IzhikevichChannel` | `IonotropicSynapse` |
| 自訂 | 論文新模型 | 繼承 `Channel` | 繼承 `Synapse` |

```python
from jaxley.channels import Channel
from jaxley.synapses import Synapse
import jax.numpy as jnp

# 自訂 Channel（ODE RHS）
class MyChannel(Channel):
    current_is_in_mA_per_cm2 = True
    channel_params = {"MyChannel_gbar": 0.1}   # S/cm²
    channel_states = {"MyChannel_m": 0.0}

    def update_states(self, states, dt, v, params):
        """通道閘控 ODE：dm/dt = (m_inf - m) / tau"""
        m_inf = 1.0 / (1.0 + jnp.exp(-(v + 40.0) / 10.0))
        return {"MyChannel_m": states["MyChannel_m"] + dt * (m_inf - states["MyChannel_m"]) / 5.0}

    def compute_current(self, states, v, params):
        """I = gbar * m * (v - E_rev) [mA/cm²]"""
        return params["MyChannel_gbar"] * states["MyChannel_m"] * (v - 50.0) / 1000.0

# 自訂 Synapse（conductance-based）
class MySynapse(Synapse):
    synapse_params = {"MySynapse_g": 1e-4, "MySynapse_e": 0.0}  # uS, mV
    synapse_states = {}

    def update_states(self, states, dt, pre_v, post_v, params):
        return {}  # 無狀態突觸

    def compute_current(self, states, pre_v, post_v, params):
        """回傳 (linear, constant) 供電壓求解器使用"""
        g = params["MySynapse_g"]
        return g, g * params["MySynapse_e"]

# 插入到網路（Step 0 建好的 net）
for neuron in neurons:
    neuron.insert(MyChannel())
```

---

## Step 2：參數初始化

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/base.py` | `set(key, val)` 設定初始值；`make_trainable(key)` 標記可訓練；`get_parameters()` 取出可訓練參數列表；`data_set()` 固定不可訓練的結構參數 |
> | `jaxley/utils/cell_utils.py` | `params_to_pstate` 將 params list 轉為 JAX-friendly 格式，供 `init_fn` 使用 |
> | Channel / Synapse 檔案 | `channel_params` / `synapse_params` dict 定義預設初始值；直接修改 dict 可改預設值 |

**三類參數區分**：

| 類型 | 說明 | Jaxley 做法 |
|------|------|-------------|
| **固定結構參數**（來自 connectome）| `n_syn`、`sign`、形態尺寸 | `net.set(key, val)`，不放入 `get_parameters()` |
| **可訓練參數** | 通道電導、突觸強度、時間常數 | `net.make_trainable(key)` → `net.get_parameters()` |
| **共享參數** | 同 cell type 共享同一值 | `net[type_mask].make_trainable(key)` |

```python
net.to_jax()  # 必須在 set/make_trainable 之後、integrate 之前呼叫

# 固定結構性參數（不訓練）
net.set("MySynapse_g", n_syn_array * base_conductance)

# 設定可訓練參數初始值
net.set("MyChannel_gbar", 0.1)

# 標記為可訓練
net.make_trainable("MyChannel_gbar")
net.make_trainable("MySynapse_g")

# 取出可訓練參數（傳給 optimizer）
params = net.get_parameters()
```

---

## Step 3：注入外部輸入

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/modules/base.py` | `stimulate(current_array)` 寫入 `self.externals` 與 `self.external_inds`；`clamp(voltage_array)` 強制固定電壓；`record(state)` 標記要錄製的狀態變數 |
> | `jaxley/stimulus.py` | `jx.step_current(i_delay, i_dur, i_amp, delta_t, t_max)` 產生標準電流波形陣列 |

```python
import jaxley as jx
import jax.numpy as jnp

delta_t = 0.025  # ms
t_max   = 100.0  # ms

# 方式一：step current（電流注入實驗）
current = jx.step_current(
    i_delay=10.0, i_dur=50.0, i_amp=0.1,
    delta_t=delta_t, t_max=t_max,
)
net[0].stimulate(current)

# 方式二：自訂感覺輸入（shape: T × n_stimulated）
visual_input = jnp.array(...)           # (T, n_input_neurons)
net[input_neuron_inds].stimulate(visual_input)

# 記錄電壓
net.record("v")

externals     = net.externals.copy()
external_inds = net.external_inds.copy()
```

---

## Step 4：執行 ODE（含暖機）

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `build_init_and_step_fn(module, voltage_solver, solver)` 回傳 `init_fn`（初始化狀態）與 `step_fn`（單步更新）；`jx.integrate()` 為一鍵模擬高階介面；flyvis branch 將 recording 硬編碼為只記錄 `v` |
> | `jaxley/utils/dynamics.py` | **flyvis branch 新增的橋接層**；`build_dynamic_state_utils(module)` 回傳三個函數，將 pytree 狀態與扁平向量互轉，供外部 ODE solver 使用 |
> | `jaxley/modules/base.py` | `_step_channels`（呼叫 Step 1 的 Channel ODE）、`_step_synapses`（呼叫 Step 1 的 Synapse ODE）；`append_channel_currents_to_states` 重建電流冗餘量；flyvis branch 移除 pumped ions / diffusion states |
> | `jaxley/solver_voltage.py` | 點神經元用 `step_voltage_explicit`（`v += dt * f`）；多區室用 `step_voltage_implicit_with_dhs_solve` |
> | `jaxley/utils/jax_utils.py` | `nested_checkpoint_scan`：gradient checkpointing，降低長序列 BPTT 記憶體用量 |

**voltage_solver 選擇指引**：

| 拓撲 | 推薦 `voltage_solver` | 說明 |
|------|----------------------|------|
| 點神經元 | `"jaxley.dhs"` | flyvis branch 簡化版，無軸向電導 |
| 單分支多區室 | `"jaxley.stone"` | 三對角矩陣求解 |
| 複雜樹狀分支 | `"jaxley.dhs"` | Dendritic Hierarchical Scheduling |
| 稀疏大網路 | `"jax.sparse"` | JAX sparse solver |

```python
import jax
import jax.numpy as jnp
from jaxley.integrate import build_init_and_step_fn
from jaxley.utils.dynamics import build_dynamic_state_utils

init_fn, step_fn = build_init_and_step_fn(
    net,
    voltage_solver="jaxley.dhs",
    solver="bwd_euler",
)
full_pytree_to_states, states_to_full_pytree, states_to_pytree = \
    build_dynamic_state_utils(net)

delta_t  = 0.025   # ms
t_pre_ms = 500     # 暖機時間

all_states, all_params = init_fn(params, None, None, delta_t)

def step_once(carry, ext):
    carry = step_fn(carry, all_params, ext, external_inds, delta_t)
    return carry, carry["v"]

# 暖機（無梯度）
warmup_ext = jnp.zeros((int(t_pre_ms / delta_t), net.ncomp))
all_states, _ = jax.lax.scan(step_once, all_states, warmup_ext)

# 正式模擬（BPTT 梯度流通）
all_states, activities = jax.lax.scan(step_once, all_states, externals)
# activities shape: (T, n_neurons)

# 或一鍵模擬（純推理，無訓練）
import jaxley as jx
v = jx.integrate(net, delta_t=delta_t, t_max=t_max)
```

---

## Step 5：任務輸出（解碼 / 讀出層）

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `step_fn` 回傳的 `all_states["v"]` 即為各神經元活動；從中取出目標神經元 index 的切片即可 |
> | —（使用者自行實作）| 解碼器 / 讀出層不屬於 Jaxley；建議獨立寫成 JAX 函數，與 `loss_fn` 串接以支援 BPTT |

```python
import jax.numpy as jnp

# 範例 A：加權解碼（光流、方向選擇性）
output_inds = [...]               # 輸出神經元 index
output_activity = activities[:, output_inds]  # (T, n_output)
decoder_weights = jnp.array(...)  # 可訓練解碼器權重
pred = jnp.einsum("ti,ij->tj", output_activity, decoder_weights)

# 範例 B：棘波分類（spike count → label）
spike_counts = jnp.sum(activities > threshold, axis=0)  # (n_neurons,)
pred = jnp.dot(spike_counts, readout_weights)            # (n_classes,)

# 範例 C：直接讀電壓（voltage clamp 分析）
final_v = activities[-1, :]   # 最後一步電壓
```

---

## Step 6：損失函數 + BPTT 優化

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | `jaxley/integrate.py` | `step_fn` 被包在 `jax.lax.scan` 內；整個 scan 對 `params` 可微，JAX 自動處理 BPTT |
> | `jaxley/utils/dynamics.py` | 若使用外部 solver，`full_pytree_to_states` / `states_to_full_pytree` 在 `loss_fn` 內呼叫 |
> | `jaxley/utils/jax_utils.py` | `nested_checkpoint_scan` 降低長序列 BPTT 記憶體 |
> | —（外部套件 `optax`）| Adam optimizer、學習率 schedule 均用 `optax` 實作 |

```python
import jax
import optax

def loss_fn(params, externals, target):
    all_states, all_params = init_fn(params, None, None, delta_t)

    def _step(carry, ext):
        carry = step_fn(carry, all_params, ext, external_inds, delta_t)
        return carry, carry["v"]

    _, activities = jax.lax.scan(_step, all_states, externals)
    pred = decode(activities)   # 自訂解碼函數
    return jnp.mean((pred - target) ** 2)

optimizer = optax.adam(learning_rate=5e-5)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, externals, target):
    loss, grads = jax.value_and_grad(loss_fn)(params, externals, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for i in range(n_iterations):
    batch = next(dataloader)
    params, opt_state, loss = train_step(
        params, opt_state, batch["externals"], batch["target"]
    )
    print(f"[{i:04d}] loss = {loss:.5f}")
```

---

## Step 7：Checkpoint 儲存

> **需要修改的 Jaxley 檔案**
>
> | 檔案 | 修改原因 |
> |------|----------|
> | —（不需修改 Jaxley）| Jaxley 的 `params` 是標準 JAX pytree（Python list of dict），可直接序列化 |

```python
import pickle
from pathlib import Path

chkpt_dir = Path("results/my_model/chkpts")
chkpt_dir.mkdir(parents=True, exist_ok=True)

# 儲存
with open(chkpt_dir / f"step_{i:06d}.pkl", "wb") as f:
    pickle.dump({
        "params":    params,
        "opt_state": opt_state,
        "step":      i,
        "loss":      float(loss),
    }, f)

# 讀取
with open(chkpt_dir / "step_000100.pkl", "rb") as f:
    chkpt = pickle.load(f)
params    = chkpt["params"]
opt_state = chkpt["opt_state"]
```

---

## 各步驟涉及的 Jaxley 檔案總覽

| 步驟 | 核心檔案 | 說明 |
|------|---------|------|
| Step 0 Connectivity | `modules/compartment.py`、`modules/cell.py`、`modules/network.py`、`connect.py`、`modules/base.py` | 拓撲結構，不含 ODE |
| Step 1 ODE 定義 | `channels/channel.py`、`channels/hh.py`、`synapses/synapse.py`、`synapses/ionotropic.py`、`synapses/tanh_rate.py`、`synapses/tanh_conductance.py`、`utils/syn_utils.py` | Channel / Synapse 的 ODE RHS |
| Step 2 參數初始化 | `modules/base.py`、`utils/cell_utils.py` | `set`、`make_trainable`、`get_parameters` |
| Step 3 注入輸入 | `modules/base.py`、`stimulus.py` | `stimulate`、`clamp`、`record` |
| Step 4 執行 ODE | `integrate.py`、`utils/dynamics.py`、`modules/base.py`、`solver_voltage.py`、`utils/jax_utils.py` | `build_init_and_step_fn` + `scan` + 暖機 |
| Step 5 任務輸出 | `integrate.py`（取 `all_states["v"]`）、使用者自訂 | 解碼器 / 讀出層不在 Jaxley 內 |
| Step 6 損失優化 | `integrate.py`、`utils/dynamics.py`、`utils/jax_utils.py`、`optax`（外部）| BPTT + Adam |
| Step 7 Checkpoint | —（不需修改 Jaxley）| `pickle` 或 `orbax-checkpoint` |

---

## 參考資源

- [Jaxley flyvis branch 說明](./FLYVIS_BRANCH.md)
- [FlyVis × Jaxley 整合指南](./JAXLEY_FLYVIS_INTEGRATION.md)
- [Jaxley 官方文件](https://jaxley.readthedocs.io/en/latest/)
- [Jaxley 論文（bioRxiv）](https://www.biorxiv.org/content/10.1101/2024.08.21.608979)
- [dynamax](https://github.com/probml/dynamax)
- [optax](https://optax.readthedocs.io/)
- [orbax-checkpoint](https://orbax.readthedocs.io/)
