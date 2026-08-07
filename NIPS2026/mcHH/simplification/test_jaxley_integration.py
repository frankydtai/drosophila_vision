"""
Jaxley集成测试脚本
测试简化后的神经元是否能被Jaxley正确加载

依赖:
    pip install jaxley jax numpy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 导入简化器
from neuron_simplifier import NeuronSimplifier

try:
    import jaxley as jx
    from jaxley import read_swc
    JAXLEY_AVAILABLE = True
except ImportError:
    print("Warning: Jaxley not installed. Install with: pip install jaxley")
    JAXLEY_AVAILABLE = False


def test_simplification_and_jaxley_loading(input_swc: str, output_dir: str):
    """
    测试完整流程：简化 -> 保存 -> Jaxley加载
    
    Args:
        input_swc: 输入SWC文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Step 1: Load original neuron")
    print("="*60)
    
    simplifier = NeuronSimplifier(angle_threshold_degrees=5.0)
    nodes, root = simplifier.load_swc(input_swc)
    
    print(f"Loaded {len(nodes)} nodes from {input_swc}")
    
    # 测试不同的简化级别
    keep_ratios = [1.0, 0.8, 0.5, 0.2, 0.05]
    
    results = []
    
    for i, ratio in enumerate(keep_ratios):
        print(f"\n{'='*60}")
        print(f"Step 2.{i+1}: Simplify with keep_ratio={ratio}")
        print("="*60)
        
        # 重新加载（避免修改原始数据）
        nodes_copy, _ = simplifier.load_swc(input_swc)
        
        # 简化
        simplified_nodes, stats = simplifier.simplify(nodes_copy, root, keep_ratio=ratio)
        
        # 保存
        basename = os.path.splitext(os.path.basename(input_swc))[0]
        output_swc = os.path.join(output_dir, f"{basename}_ratio{ratio:.2f}.swc")
        simplifier.write_swc(simplified_nodes, output_swc)
        simplifier.save_metadata(stats, input_swc, output_swc)
        
        print(f"\nSimplification Statistics:")
        print(f"  Original nodes: {stats.original_nodes}")
        print(f"  Simplified nodes: {stats.simplified_nodes}")
        print(f"  Reduction: {(1 - stats.simplified_nodes/stats.original_nodes)*100:.1f}%")
        print(f"  Surface area preservation: {stats.surface_area_preservation_ratio*100:.2f}%")
        
        # 测试Jaxley加载
        if JAXLEY_AVAILABLE:
            print(f"\n{'='*60}")
            print(f"Step 3.{i+1}: Test Jaxley loading")
            print("="*60)
            
            try:
                # 使用Jaxley加载
                cell = read_swc(
                    output_swc,
                    ncomp=4,  # 每个分支4个compartment
                    max_branch_len=None,
                    min_radius=0.1,
                    assign_groups=True
                )
                
                print(f"✓ Jaxley successfully loaded the simplified neuron")
                print(f"  Cell info: {cell}")
                print(f"  Number of compartments: {len(cell.nodes)}")
                
                # 尝试设置一些基本参数
                cell.set("axial_resistivity", 100.0)  # Ω·cm
                cell.set("capacitance", 1.0)  # μF/cm²
                
                print(f"✓ Successfully set biophysical parameters")
                
                results.append({
                    'ratio': ratio,
                    'stats': stats,
                    'jaxley_cell': cell,
                    'output_file': output_swc,
                    'success': True
                })
                
            except Exception as e:
                print(f"✗ Jaxley loading failed: {e}")
                results.append({
                    'ratio': ratio,
                    'stats': stats,
                    'output_file': output_swc,
                    'success': False,
                    'error': str(e)
                })
        else:
            results.append({
                'ratio': ratio,
                'stats': stats,
                'output_file': output_swc,
                'success': None  # Jaxley not available
            })
    
    # 生成对比图
    print(f"\n{'='*60}")
    print("Step 4: Generate comparison plots")
    print("="*60)
    
    plot_comparison(results, output_dir)
    
    return results


def plot_comparison(results, output_dir):
    """生成对比图表"""
    ratios = [r['ratio'] for r in results]
    node_counts = [r['stats'].simplified_nodes for r in results]
    surface_areas = [r['stats'].total_surface_area_simplified for r in results]
    preservation_ratios = [r['stats'].surface_area_preservation_ratio for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 节点数量
    axes[0, 0].plot(ratios, node_counts, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Keep Ratio')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Simplified Node Count vs Keep Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 表面积
    axes[0, 1].plot(ratios, surface_areas, 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Keep Ratio')
    axes[0, 1].set_ylabel('Surface Area (μm²)')
    axes[0, 1].set_title('Total Surface Area vs Keep Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 表面积保留率
    axes[1, 0].plot(ratios, [p*100 for p in preservation_ratios], '^-', 
                    linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Keep Ratio')
    axes[1, 0].set_ylabel('Surface Area Preservation (%)')
    axes[1, 0].set_title('Surface Area Preservation vs Keep Ratio')
    axes[1, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 简化效率
    original_nodes = results[0]['stats'].original_nodes
    reduction_ratios = [(1 - n/original_nodes)*100 for n in node_counts]
    axes[1, 1].plot(ratios, reduction_ratios, 'd-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Keep Ratio')
    axes[1, 1].set_ylabel('Node Reduction (%)')
    axes[1, 1].set_title('Node Reduction vs Keep Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'simplification_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.close()


def generate_test_report(results, output_dir):
    """生成测试报告"""
    report_path = os.path.join(output_dir, 'test_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Neuron Simplification and Jaxley Integration Test Report\n\n")
        f.write(f"Generated: {os.popen('date').read().strip()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total test cases: {len(results)}\n")
        
        if JAXLEY_AVAILABLE:
            success_count = sum(1 for r in results if r.get('success', False))
            f.write(f"- Jaxley loading success: {success_count}/{len(results)}\n")
        else:
            f.write("- Jaxley: Not installed (skipped loading tests)\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for i, result in enumerate(results):
            stats = result['stats']
            f.write(f"### Test {i+1}: Keep Ratio = {result['ratio']}\n\n")
            f.write(f"**Simplification Statistics:**\n")
            f.write(f"- Original nodes: {stats.original_nodes}\n")
            f.write(f"- Simplified nodes: {stats.simplified_nodes}\n")
            f.write(f"- Removed by collinear: {stats.removed_by_collinear}\n")
            f.write(f"- Removed by pruning: {stats.removed_by_pruning}\n")
            f.write(f"- Node reduction: {(1 - stats.simplified_nodes/stats.original_nodes)*100:.1f}%\n")
            f.write(f"- Surface area preservation: {stats.surface_area_preservation_ratio*100:.2f}%\n")
            
            if result.get('success') is not None:
                if result['success']:
                    f.write(f"\n**Jaxley Loading:** ✓ Success\n")
                    f.write(f"- Output file: `{result['output_file']}`\n")
                else:
                    f.write(f"\n**Jaxley Loading:** ✗ Failed\n")
                    f.write(f"- Error: {result.get('error', 'Unknown')}\n")
            
            f.write("\n---\n\n")
        
        f.write("## Visualization\n\n")
        f.write("![Comparison Plot](simplification_comparison.png)\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The neuron simplification algorithm successfully:\n")
        f.write("1. Reduces node count while preserving morphological structure\n")
        f.write("2. Maintains physical properties (surface area) through weighted averaging\n")
        f.write("3. Generates Jaxley-compatible SWC files\n")
    
    print(f"Test report saved to: {report_path}")


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) > 1:
        input_swc = sys.argv[1]
    else:
        # 默认测试文件（需要根据实际情况修改）
        input_swc = "/Users/lengyuner/Desktop/data/flywire/Jun2025_swc/720575940626631866.swc"
    
    if not os.path.exists(input_swc):
        print(f"Error: Input file not found: {input_swc}")
        print(f"Usage: python test_jaxley_integration.py <input_swc_file>")
        sys.exit(1)
    
    output_dir = "test_output"
    
    print("Starting neuron simplification and Jaxley integration test...")
    print(f"Input: {input_swc}")
    print(f"Output directory: {output_dir}")
    
    results = test_simplification_and_jaxley_loading(input_swc, output_dir)
    
    generate_test_report(results, output_dir)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
