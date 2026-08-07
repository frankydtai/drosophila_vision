"""
批量处理脚本 - 简化多个神经元文件

用法:
    python batch_simplify.py <input_dir> <output_dir> [options]

示例:
    python batch_simplify.py /path/to/swc_files ./output --angle 5 --ratios 0.2,0.5,0.8
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
import json

from neuron_simplifier import NeuronSimplifier


def parse_args():
    parser = argparse.ArgumentParser(description='Batch neuron simplification')
    
    parser.add_argument('input_dir', type=str,
                       help='Input directory containing SWC files')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for simplified files')
    parser.add_argument('--angle', type=float, default=5.0,
                       help='Angle threshold for collinear merging (degrees, default: 5.0)')
    parser.add_argument('--ratios', type=str, default='0.05,0.2,0.5,0.8,1.0',
                       help='Comma-separated keep ratios (default: 0.05,0.2,0.5,0.8,1.0)')
    parser.add_argument('--pattern', type=str, default='*.swc',
                       help='File pattern to match (default: *.swc)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth file (default: 1, process all)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (default: all)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析keep ratios
    keep_ratios = [float(r.strip()) for r in args.ratios.split(',')]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有SWC文件
    swc_files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    
    if not swc_files:
        print(f"Error: No files matching '{args.pattern}' found in {args.input_dir}")
        sys.exit(1)
    
    # 应用skip和max_files
    swc_files = swc_files[::args.skip]
    if args.max_files:
        swc_files = swc_files[:args.max_files]
    
    print(f"Found {len(swc_files)} files to process")
    print(f"Angle threshold: {args.angle}°")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # 创建简化器
    simplifier = NeuronSimplifier(angle_threshold_degrees=args.angle)
    
    # 统计信息
    all_stats = []
    failed_files = []
    
    # 批量处理
    for swc_path in tqdm(swc_files, desc="Processing neurons"):
        basename = os.path.splitext(os.path.basename(swc_path))[0]
        
        try:
            # 加载
            nodes, root = simplifier.load_swc(swc_path)
            
            # 对每个keep_ratio进行简化
            for ratio in keep_ratios:
                # 重新加载（避免修改原始数据）
                nodes_copy, _ = simplifier.load_swc(swc_path)
                
                # 简化
                simplified_nodes, stats = simplifier.simplify(nodes_copy, root, keep_ratio=ratio)
                
                # 保存
                output_swc = os.path.join(args.output_dir, f"{basename}_ratio{ratio:.2f}.swc")
                simplifier.write_swc(simplified_nodes, output_swc)
                simplifier.save_metadata(stats, swc_path, output_swc)
                
                # 记录统计
                all_stats.append({
                    'file': basename,
                    'ratio': ratio,
                    'stats': stats
                })
        
        except Exception as e:
            print(f"\nError processing {swc_path}: {e}")
            failed_files.append((swc_path, str(e)))
    
    # 生成汇总报告
    print("\n" + "="*60)
    print("Generating summary report...")
    generate_summary_report(all_stats, failed_files, args.output_dir)
    
    print("\n" + "="*60)
    print("Batch processing completed!")
    print(f"Processed: {len(swc_files) - len(failed_files)}/{len(swc_files)} files")
    if failed_files:
        print(f"Failed: {len(failed_files)} files (see summary_report.json)")


def generate_summary_report(all_stats, failed_files, output_dir):
    """生成汇总报告"""
    
    # 按ratio分组统计
    ratio_stats = {}
    for entry in all_stats:
        ratio = entry['ratio']
        stats = entry['stats']
        
        if ratio not in ratio_stats:
            ratio_stats[ratio] = {
                'count': 0,
                'total_original_nodes': 0,
                'total_simplified_nodes': 0,
                'total_original_area': 0,
                'total_simplified_area': 0,
                'avg_preservation': 0
            }
        
        rs = ratio_stats[ratio]
        rs['count'] += 1
        rs['total_original_nodes'] += stats.original_nodes
        rs['total_simplified_nodes'] += stats.simplified_nodes
        rs['total_original_area'] += stats.total_surface_area_original
        rs['total_simplified_area'] += stats.total_surface_area_simplified
        rs['avg_preservation'] += stats.surface_area_preservation_ratio
    
    # 计算平均值
    for ratio, rs in ratio_stats.items():
        if rs['count'] > 0:
            rs['avg_preservation'] /= rs['count']
            rs['avg_node_reduction'] = 1 - (rs['total_simplified_nodes'] / rs['total_original_nodes'])
            rs['avg_area_preservation'] = rs['total_simplified_area'] / rs['total_original_area']
    
    # 保存JSON报告
    report = {
        'summary': {
            'total_files_processed': len(set(e['file'] for e in all_stats)),
            'total_files_failed': len(failed_files),
            'keep_ratios': sorted(ratio_stats.keys())
        },
        'ratio_statistics': ratio_stats,
        'failed_files': [{'file': f, 'error': e} for f, e in failed_files]
    }
    
    json_path = os.path.join(output_dir, 'summary_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成Markdown报告
    md_path = os.path.join(output_dir, 'summary_report.md')
    with open(md_path, 'w') as f:
        f.write("# Batch Simplification Summary Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Total files processed: {report['summary']['total_files_processed']}\n")
        f.write(f"- Total files failed: {report['summary']['total_files_failed']}\n")
        f.write(f"- Keep ratios tested: {report['summary']['keep_ratios']}\n\n")
        
        f.write("## Statistics by Keep Ratio\n\n")
        f.write("| Keep Ratio | Files | Avg Node Reduction | Avg Area Preservation |\n")
        f.write("|-----------|-------|-------------------|----------------------|\n")
        
        for ratio in sorted(ratio_stats.keys()):
            rs = ratio_stats[ratio]
            f.write(f"| {ratio:.2f} | {rs['count']} | "
                   f"{rs['avg_node_reduction']*100:.1f}% | "
                   f"{rs['avg_area_preservation']*100:.1f}% |\n")
        
        if failed_files:
            f.write("\n## Failed Files\n\n")
            for file, error in failed_files:
                f.write(f"- `{file}`: {error}\n")
    
    print(f"Summary reports saved:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
