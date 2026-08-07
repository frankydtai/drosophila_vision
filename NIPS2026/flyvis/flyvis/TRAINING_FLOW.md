# flyvis 訓練流程

> 對照論文 *A connectome-based model of the Drosophila visual system* (Nature 2024) 與原始碼。

---

## 0. 訓練前：神經資料準備

訓練開始之前，必須先準備好兩類靜態神經資料：**Connectivity Matrix** 與 **Neural Dynamics 參數**。這些資料決定了網路的結構，在訓練中保持**固定不變**。

### 0-A. Connectivity Matrix（突觸連接矩陣）

#### 原始資料來源

| 資料集 | 覆蓋範圍 | 神經元數量 |
|--------|----------|------------|
| **FIB-25**（Janelia FlyEM）| 7 個 medulla columns | 702 個 |
| **FIB-19**（Janelia FlyEM）| 整個 optic lobe | 1,099 個 |
| **文獻補充**（Rivera-Alba et al., Tuthill et al.）| Lamina（R1–R8, L1–L5）| 手工資料 |

合計 **1,801 個 reconstructed neurons**，830 個有手動標注的 column 位置。

#### 建構流程

```
[FIB-25 + FIB-19 FIB-SEM 電鏡重建]
        ↓
1. Probabilistic EM Algorithm（位置推算）
   輸入：830 個 hand-annotated reference positions + 3D synapse 座標
   輸出：所有 1,801 個 neurons 的 retinotopic (u,v) column 座標
        ↓
2. 統計每對 cell-type 的突觸 offset 分佈
   → 604 個 average convolutional filters
   → 每個 filter：{src_type, tar_type, offsets: [(du,dv, n_syn), ...]}
        ↓
3. 填補 Convex Hull 空洞（fill_hull, n_syn_fill=1）
   標記補充 edge：n_syn_certainty < 1.0
        ↓
4. 確定突觸正負號（sign）
   Cholinergic → +1（興奮）
   GABAergic / Glutamatergic / Histaminergic → -1（抑制）
   來源：Özel et al. 2021 神經傳導物質 transcriptomics
        ↓
5. fib25-fib19_v2.2.json（65 cell types, 605 edges）
        ↓  ConnectomeFromAvgFilters(extent=15, n_syn_fill=1)
           flyvis/connectome/connectome.py:109
        ↓
45,669 neurons + 1,513,231 edges（hex lattice 展開）
```

#### 展開後的資料結構（`data/connectome/` 目錄）

```
nodes/
  ├── index          # 45,669 個整數 index
  └── type           # 每個 neuron 的細胞類型字串
edges/
  ├── source_index   # pre-synaptic neuron index
  ├── target_index   # post-synaptic neuron index
  ├── n_syn          # 突觸數量（來自 EM 重建）← 固定
  ├── sign           # ±1 興奮/抑制              ← 固定
  └── n_syn_certainty  # ≥1.0 已觀測；<1.0 補充推算
```

> **程式碼位置**：`flyvis/connectome/connectome.py` — `ConnectomeFromAvgFilters`

---

### 0-B. Neural Dynamics 參數初始化

由 `flyvis/network/initialization.py` 處理，參數共用於同一 cell type 的所有神經元：

| 參數 | 類別 | 初始值 | 是否可訓練 | groupby |
|------|------|--------|------------|---------|
| `time_const`（$\tau_i$）| `TimeConstant` | 0.05 s（50 ms）| 是 | cell type |
| `bias`（$V_i^{\text{rest}}$）| `RestingPotential` | $\mathcal{N}(0.5, \sigma^2)$ 取樣 | 是 | cell type |
| `syn_strength`（$\alpha_{t_i t_j}$）| `SynapseCountScaling` | $0.01 / \langle N \rangle$ | 是 | edge type |
| `sign`（$\sigma$）| `SynapseSign` | 來自 connectome | **否** | edge type |
| `syn_count`（$N$）| `SynapseCount` | 來自 connectome | **否** | edge type |

> 50 個訓練模型（ensemble）全部使用**相同的初始參數值**，差異僅來自隨機梯度下降的隨機性。

---

### 0-C. 快速執行（程式碼）

```python
# 建構 connectome（第一次執行會快取到 data/connectome/）
from flyvis.connectome.connectome import ConnectomeFromAvgFilters
connectome = ConnectomeFromAvgFilters(
    file="fib25-fib19_v2.2.json",
    extent=15,
    n_syn_fill=1,
)
# connectome.nodes.index.shape  → (45669,)
# connectome.edges.n_syn.shape  → (1513231,)

# 建立網路（自動從 connectome 初始化所有參數）
from flyvis.network.network import Network
network = Network(connectome=connectome, dynamics_config={"type": "PPNeuronIGRSynapses"})
# network 現在含有 734 個可學習參數
```

---

## 1. 整體架構

```
[FIB-SEM 電鏡重建] → fib25-fib19_v2.2.json → ConnectomeFromAvgFilters
                                                        ↓
[MPI Sintel 光流資料集]                       45,669 neurons
        ↓                                    1,513,231 edges
MultiTaskSolver.__init__()          flyvis/solver.py:95
  ├── Network.__init__()             flyvis/network/network.py
  ├── DecoderGAVP.__init__()         flyvis/task/decoder.py
  ├── Task.__init__()                flyvis/task/tasks.py
  ├── Adam optimizer
  └── HyperParamScheduler
        ↓
MultiTaskSolver.train()             flyvis/solver.py:220
  ├── [每 epoch] network.steady_state()   灰色刺激 500 ms 暖機
  └── [每 batch]
        ├── network.stimulus.add_input()  注入視覺輸入
        ├── network.forward()             Euler 積分 T 時間步
        ├── decoder[task](activity)       解碼光流
        ├── task.loss(y_est, y)           L2 loss
        ├── loss.backward()              BPTT
        ├── optimizer.step()             更新 734 個參數
        └── penalty(activity)            活動懲罰項
        ↓
MultiTaskSolver.checkpoint()        flyvis/solver.py:390
  └── 儲存 network / decoder / optimizer state dict
```

---

## 2. 可訓練參數（論文 Methods）

共 **734 個自由參數**（與神經元數量 45,669 無關）：

| 參數 | 數量 | 初始化 | 約束 | 程式碼位置 |
|------|------|--------|------|------------|
| 突觸縮放因子 $\alpha_{t_i t_j}$ | 604 | $0.01 / \langle N_{t_i t_j} \rangle$ | $\alpha \geq 0$（clamped）| `initialization.py` → `SynapticStrength` |
| 靜止電位 $V_i^{\text{rest}}$ | 65 | $\mathcal{N}(\mu, \sigma^2)$ 取樣 | 無 | `initialization.py` → `RestingPotential` |
| 膜時間常數 $\tau_i$ | 65 | 50 ms | $\tau_i \geq \Delta t$（clamped）| `initialization.py` → `TimeConstant` |

**固定參數**（不訓練）：
- 突觸符號 $\sigma \in \{-1, +1\}$：來自神經傳導物質分析
- 突觸數量 $N$：來自 connectome 電子顯微鏡重建

---

## 3. 每個 Epoch 詳細步驟

### 步驟 A：計算穩態初始電壓

```python
# flyvis/solver.py:299
steady_state = self.network.steady_state(
    t_pre=0.5,          # 500 ms 灰色刺激
    dt=self.task.dataset.dt,
    batch_size=dataloader.batch_size,
    value=0.5,          # 灰色亮度
)
```

> **論文對應**：每次訓練前以均勻灰色刺激跑網路至穩態，作為初始電壓。

---

### 步驟 B：注入視覺刺激

```python
# flyvis/solver.py:312-317
n_samples, n_frames, _, _ = data["lum"].shape
self.network.stimulus.zero(n_samples, n_frames)
self.network.stimulus.add_input(data["lum"])  # shape: (samples, frames, hexals)
```

> **對應**：`flyvis/network/stimulus.py` — `Stimulus.buffer`，即公式 (1) 中的 $e_i$

---

### 步驟 C：前向傳播（Euler 積分）

```python
# flyvis/solver.py:323-327
activity = self.network(
    self.network.stimulus(),
    self.task.dataset.dt,   # Δt = 1/50 s = 20 ms
    state=steady_state,
)
# 輸出 shape: (batch, T, n_nodes=45669)
```

內部每個時間步呼叫：

```python
# flyvis/network/dynamics.py:185-218  (PPNeuronIGRSynapses.write_state_velocity)
vel.nodes.activity = (
    1 / torch.max(params.nodes.time_const, torch.tensor(dt).float())  # 1/τ_i
    * (
        -state.nodes.activity                                          # -V_i
        + params.nodes.bias                                            # V^rest_i
        + target_sum(
            params.edges.weight * self.activation(state.sources.activity)
        )                                                              # Σ_j s_ij
        + x_t                                                          # e_i
    )
)
# 接著 network.py: state += dt * velocity  (Euler step)
```

> **論文公式 (1)**：$\tau_i \dot{V}_i = -V_i + \sum_j s_{ij} + V_i^{\text{rest}} + e_i$

---

### 步驟 D：解碼光流

```python
# flyvis/solver.py:332
y_est = self.decoder[task](activity)
```

解碼器（`flyvis/task/decoder.py` — `DecoderGAVP`）依序執行：
1. ReLU 整流（避免負動態範圍）
2. 映射至笛卡爾坐標
3. 空間卷積層 + BatchNorm + Softplus + Dropout
4. 第二層空間卷積 → 2D 光流輸出 $\hat{Y} \in \mathbb{R}^{N \times C \times 2}$

---

### 步驟 E：計算損失

```python
# flyvis/solver.py:336-342
losses[task] = self.task.loss(y_est, y, task)
loss = sum(losses.values())
```

> **論文損失函數**：$\mathcal{L}(\hat{Y}, Y) = \|\hat{Y} - Y\|_2$（L2 norm，對 MPI Sintel ground truth 光流）

---

### 步驟 F：反向傳播與參數更新

```python
# flyvis/solver.py:344-347
loss.backward(retain_graph=True)   # BPTT（Backpropagation Through Time）
self.optimizer.step()              # Adam 更新 734 個生物物理參數
```

> **論文優化器設定**：
> - Adam：$\beta_1=0.9$，$\beta_2=0.999$
> - 學習率：$5\times10^{-5}$ → $5\times10^{-6}$（10 步線性衰減）
> - Batch size：4

---

### 步驟 G：活動懲罰項

```python
# flyvis/solver.py:350
self.penalty(activity=activity, iteration=self.iteration)
```

`Penalty` 類別（`solver.py:700+`）可施加額外的活動正則化，由各自的懲罰優化器更新。

---

### 步驟 H：Checkpoint

```python
# flyvis/solver.py:422-468  (MultiTaskSolver.checkpoint)
torch.save(chkpt, checkpoint_path / f"chkpt_{idx:05}")
```

儲存內容：
```
data/results/<network_name>/
  ├── chkpts/chkpt_00000     # network + decoder + optimizer state dict
  ├── loss.h5                # 每次 iteration 的總損失
  ├── loss_flow.h5           # 光流任務損失
  ├── activity.h5            # 平均神經活動
  └── best_chkpt_index.h5   # 驗證集損失最低的 checkpoint
```

---

## 4. 優化設定總覽

| 設定 | 值 | 程式碼位置 |
|------|----|------------|
| 優化器 | Adam | `solver.py:_init_solver` |
| $\beta_1, \beta_2$ | 0.9, 0.999 | config YAML |
| 學習率 | $5\times10^{-5}$ → $5\times10^{-6}$ | `HyperParamScheduler` |
| Batch size | 4 | task config |
| 損失函數 | L2 norm | `flyvis/task/tasks.py` |
| 梯度方法 | BPTT | `loss.backward()` |
| 重複訓練 | 50 個獨立初始化的 DMN | `flyvis/network/ensemble.py` |
| 時間步長 $\Delta t$ | 20 ms（50 Hz）| `task.dataset.dt` |
| 暖機時間 | 500 ms | `config.t_pre_train = 0.5` |

---

## 5. 關鍵檔案索引

| 功能 | 檔案 | 主要類別/函數 |
|------|------|---------------|
| 訓練主迴圈 | `flyvis/solver.py` | `MultiTaskSolver.train()` |
| 神經元動態（公式 1）| `flyvis/network/dynamics.py` | `PPNeuronIGRSynapses.write_state_velocity()` |
| 網路前向傳播 | `flyvis/network/network.py` | `Network.forward()` |
| 參數初始化 | `flyvis/network/initialization.py` | `TimeConstant`, `RestingPotential`, `SynapticStrength` |
| 視覺輸入注入 | `flyvis/network/stimulus.py` | `Stimulus.add_input()` |
| 光流解碼 | `flyvis/task/decoder.py` | `DecoderGAVP` |
| 損失計算 | `flyvis/task/tasks.py` | `Task.loss()` |
| Checkpoint 管理 | `flyvis/network/directories.py` | `NetworkDir` |
| 集成訓練 | `flyvis/network/ensemble.py` | `EnsembleDir` |
