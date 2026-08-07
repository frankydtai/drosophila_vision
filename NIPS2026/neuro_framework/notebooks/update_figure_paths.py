"""
Update all Python scripts to save figures to figures/ subdirectory
"""
import re
from pathlib import Path

# Files to update
files_to_update = [
    'test_demo.py',
    '02_full_connectome_visualization.py',
    '03_fafb_full_brain.py',
    '04_network_visualization.py'
]

notebooks_dir = Path('/Users/lengyuner/Desktop/NIPS2026/neuro_framework/notebooks')

for filename in files_to_update:
    filepath = notebooks_dir / filename
    if not filepath.exists():
        print(f"⚠️  Skipping {filename} (not found)")
        continue
    
    print(f"Processing {filename}...")
    
    # Read file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern 1: PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'fig*.png'
    pattern1 = r"(PROJECT_ROOT\s*/\s*'neuro_framework'\s*/\s*'notebooks'\s*/\s*')([^']+\.png')"
    replacement1 = r"\1figures/\2"
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: output_dir / 'network_*.png'
    pattern2 = r"(output_path\s*=\s*output_dir\s*/\s*f?['\"])([^'\"]+\.png)"
    replacement2 = r"\1figures/\2"
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: Direct paths like 'fig*.png' or f'fig*.png'
    pattern3 = r"(['\"])(fig_[^'\"]+\.png|network_[^'\"]+\.png)(['\"])"
    replacement3 = r"\1figures/\2\3"
    content = re.sub(pattern3, replacement3, content)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {filename}")

print("\n✅ All files updated!")
print("Figures will now be saved to: notebooks/figures/")
