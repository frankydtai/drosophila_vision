# 物理性质保留的数学推导

## 1. 问题定义

在神经元简化过程中，当我们删除共线的中间节点时，需要保证简化后的神经元保持与原始神经元相同的电生理特性。

## 2. 关键物理量

### 2.1 膜表面积 (Membrane Surface Area)

对于圆柱形compartment：
$$A_{membrane} = 2\pi r L$$

其中：
- $r$ = 半径 (radius)
- $L$ = 长度 (length)

### 2.2 膜电容 (Membrane Capacitance)

$$C_m = c_m \cdot A_{membrane} = c_m \cdot 2\pi r L$$

其中：
- $c_m$ = 比电容 (specific capacitance)，通常 $\approx 1 \, \mu F/cm^2$

### 2.3 轴向电阻 (Axial Resistance)

$$R_{axial} = \frac{R_a \cdot L}{\pi r^2}$$

其中：
- $R_a$ = 轴向电阻率 (axial resistivity)，通常 $\approx 100 \, \Omega \cdot cm$

### 2.4 膜电导 (Membrane Conductance)

$$G_m = g_m \cdot A_{membrane} = g_m \cdot 2\pi r L$$

其中：
- $g_m$ = 比电导 (specific conductance)

## 3. 共线节点合并问题

### 3.1 原始配置

考虑三个共线节点 A-B-C：

**段1 (A→B):**
- 长度: $L_1$
- 半径: $r_1$
- 表面积: $A_1 = 2\pi r_1 L_1$
- 电容: $C_1 = c_m \cdot 2\pi r_1 L_1$
- 轴向电阻: $R_1 = \frac{R_a L_1}{\pi r_1^2}$

**段2 (B→C):**
- 长度: $L_2$
- 半径: $r_2$
- 表面积: $A_2 = 2\pi r_2 L_2$
- 电容: $C_2 = c_m \cdot 2\pi r_2 L_2$
- 轴向电阻: $R_2 = \frac{R_a L_2}{\pi r_2^2}$

### 3.2 目标：等效单段 (A→C)

我们需要找到等效参数 $(L_{eq}, r_{eq})$ 使得：

**等效段:**
- 长度: $L_{eq}$
- 半径: $r_{eq}$
- 表面积: $A_{eq} = 2\pi r_{eq} L_{eq}$
- 电容: $C_{eq} = c_m \cdot 2\pi r_{eq} L_{eq}$
- 轴向电阻: $R_{eq} = \frac{R_a L_{eq}}{\pi r_{eq}^2}$

## 4. 保留策略推导

### 4.1 策略1：保留总表面积和总电容

**约束条件:**
1. 长度相加: $L_{eq} = L_1 + L_2$
2. 表面积守恒: $A_{eq} = A_1 + A_2$

**推导:**

$$A_{eq} = A_1 + A_2$$
$$2\pi r_{eq} L_{eq} = 2\pi r_1 L_1 + 2\pi r_2 L_2$$
$$r_{eq} L_{eq} = r_1 L_1 + r_2 L_2$$
$$r_{eq} (L_1 + L_2) = r_1 L_1 + r_2 L_2$$

**解:**
$$\boxed{r_{eq} = \frac{r_1 L_1 + r_2 L_2}{L_1 + L_2}}$$

这是**长度加权平均半径**。

**验证电容守恒:**
$$C_{eq} = c_m \cdot 2\pi r_{eq} L_{eq} = c_m \cdot 2\pi \cdot \frac{r_1 L_1 + r_2 L_2}{L_1 + L_2} \cdot (L_1 + L_2)$$
$$= c_m \cdot 2\pi (r_1 L_1 + r_2 L_2)$$
$$= c_m \cdot 2\pi r_1 L_1 + c_m \cdot 2\pi r_2 L_2 = C_1 + C_2 \quad \checkmark$$

### 4.2 策略2：保留轴向电阻

**约束条件:**
1. 长度相加: $L_{eq} = L_1 + L_2$
2. 轴向电阻守恒（串联）: $R_{eq} = R_1 + R_2$

**推导:**

$$R_{eq} = R_1 + R_2$$
$$\frac{R_a L_{eq}}{\pi r_{eq}^2} = \frac{R_a L_1}{\pi r_1^2} + \frac{R_a L_2}{\pi r_2^2}$$
$$\frac{L_{eq}}{r_{eq}^2} = \frac{L_1}{r_1^2} + \frac{L_2}{r_2^2}$$
$$\frac{L_1 + L_2}{r_{eq}^2} = \frac{L_1}{r_1^2} + \frac{L_2}{r_2^2}$$

**解:**
$$r_{eq}^2 = \frac{L_1 + L_2}{\frac{L_1}{r_1^2} + \frac{L_2}{r_2^2}}$$

$$\boxed{r_{eq} = \sqrt{\frac{L_1 + L_2}{\frac{L_1}{r_1^2} + \frac{L_2}{r_2^2}}}}$$

这是**调和平均的变体**。

### 4.3 策略3：混合策略（加权）

可以定义一个加权组合：

$$r_{eq} = \alpha \cdot r_{eq,\text{area}} + (1-\alpha) \cdot r_{eq,\text{resistance}}$$

其中 $\alpha \in [0, 1]$ 是权重参数。

## 5. 策略比较

### 5.1 策略1（表面积守恒）的优点

✓ **简单**: 只需要长度加权平均  
✓ **保留电容**: 对于电缆方程很重要  
✓ **保留膜电导**: $G_m$ 也守恒  
✓ **物理直观**: 总膜面积不变

### 5.2 策略2（轴向电阻守恒）的优点

✓ **保留轴向传导**: 对于电信号传播很重要  
✓ **精确的电阻网络**: 等效电路完全相同

### 5.3 策略2的缺点

✗ **不保留表面积**: 可能改变总电容  
✗ **计算复杂**: 需要平方根运算

## 6. 推荐方案

### 6.1 默认使用策略1（表面积守恒）

**理由:**
1. **Jaxley的假设**: Jaxley使用 $A = 2\pi r L$ 计算表面积，策略1完美匹配
2. **电容主导**: 在许多神经元模型中，膜电容是最重要的参数
3. **简单高效**: 计算简单，数值稳定
4. **保守性**: 不会引入额外的误差

### 6.2 特殊情况使用策略2

当需要精确保留轴向电阻时（例如研究电信号传播速度），可以使用策略2。

## 7. 数值示例

### 示例1：相同半径

假设 $r_1 = r_2 = 1.0 \, \mu m$, $L_1 = 10 \, \mu m$, $L_2 = 20 \, \mu m$

**策略1:**
$$r_{eq} = \frac{1.0 \times 10 + 1.0 \times 20}{10 + 20} = \frac{30}{30} = 1.0 \, \mu m$$

**策略2:**
$$r_{eq} = \sqrt{\frac{30}{\frac{10}{1^2} + \frac{20}{1^2}}} = \sqrt{\frac{30}{30}} = 1.0 \, \mu m$$

两种策略结果相同 ✓

### 示例2：不同半径

假设 $r_1 = 2.0 \, \mu m$, $r_2 = 1.0 \, \mu m$, $L_1 = L_2 = 10 \, \mu m$

**策略1:**
$$r_{eq} = \frac{2.0 \times 10 + 1.0 \times 10}{10 + 10} = \frac{30}{20} = 1.5 \, \mu m$$

**策略2:**
$$r_{eq} = \sqrt{\frac{20}{\frac{10}{4} + \frac{10}{1}}} = \sqrt{\frac{20}{12.5}} = 1.265 \, \mu m$$

策略1给出更大的半径（保留更多表面积）

**验证表面积:**
- 原始: $A_1 + A_2 = 2\pi(2 \times 10 + 1 \times 10) = 60\pi$
- 策略1: $A_{eq} = 2\pi \times 1.5 \times 20 = 60\pi$ ✓
- 策略2: $A_{eq} = 2\pi \times 1.265 \times 20 \approx 50.6\pi$ ✗

**验证轴向电阻:**
- 原始: $R_1 + R_2 = R_a(\frac{10}{4\pi} + \frac{10}{\pi}) = \frac{12.5 R_a}{\pi}$
- 策略1: $R_{eq} = R_a \frac{20}{2.25\pi} = \frac{8.89 R_a}{\pi}$ ✗
- 策略2: $R_{eq} = R_a \frac{20}{1.6\pi} = \frac{12.5 R_a}{\pi}$ ✓

## 8. 实现建议

```python
def compute_equivalent_radius_area_preserving(r1, L1, r2, L2):
    """策略1：保留表面积"""
    return (r1 * L1 + r2 * L2) / (L1 + L2)

def compute_equivalent_radius_resistance_preserving(r1, L1, r2, L2):
    """策略2：保留轴向电阻"""
    return np.sqrt((L1 + L2) / (L1 / r1**2 + L2 / r2**2))

def compute_equivalent_radius_hybrid(r1, L1, r2, L2, alpha=0.7):
    """策略3：混合（alpha=1为纯表面积，alpha=0为纯电阻）"""
    r_area = compute_equivalent_radius_area_preserving(r1, L1, r2, L2)
    r_resistance = compute_equivalent_radius_resistance_preserving(r1, L1, r2, L2)
    return alpha * r_area + (1 - alpha) * r_resistance
```

## 9. 结论

**推荐使用策略1（表面积守恒）**作为默认方法，因为：
1. 与Jaxley的表面积计算方式一致
2. 保留了最重要的电生理参数（电容）
3. 计算简单，数值稳定
4. 物理意义清晰

对于需要精确保留轴向电阻的特殊应用，可以提供策略2作为选项。
