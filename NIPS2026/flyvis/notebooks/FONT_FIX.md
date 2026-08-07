# Notebooks 中文字体修复说明

## 问题
在 Jupyter Notebook 中运行时，使用 `matplotlib.use('Agg')` 后端会导致中文显示问题。

## 解决方案
已将所有 notebooks 更新为：
1. 移除 `matplotlib.use('Agg')` 后端（仅在命令行脚本中使用）
2. 添加字体警告过滤
3. 配置跨平台中文字体支持

## 更新的文件
- `03_network_hierarchy.ipynb` ✓
- `04_network_activation.ipynb` ✓

## 字体配置
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')

import matplotlib.pyplot as plt
import locale
try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

import matplotlib.font_manager as fm
import platform
system = platform.system()

if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'DejaVu Sans']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
```

## 使用方法
1. 在 Jupyter 中打开 notebook
2. 运行所有单元格
3. 中文应该能正常显示

如果仍有问题，请重启 Jupyter kernel。
