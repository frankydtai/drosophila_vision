#!/usr/bin/env python3
"""
快速验证脚本：测试 FlyWire 连接组

验证生成的 FlyWire 连接组可以在 Flyvis 框架中正常工作。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_json_format():
    """测试 JSON 文件格式"""
    print("=" * 60)
    print("测试 1: JSON 文件格式")
    print("=" * 60)
    
    import json
    
    json_path = project_root / "flyvis/connectome/flywire_v1.0.json"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ JSON 文件加载成功")
        print(f"  - 文件大小: {json_path.stat().st_size / 1024:.1f} KB")
        
        # 检查必需字段
        required_fields = ['nodes', 'edges', 'input_units', 'output_units']
        for field in required_fields:
            if field in data:
                print(f"✓ 包含字段: {field}")
            else:
                print(f"✗ 缺少字段: {field}")
                return False
                
        # 统计信息
        print(f"\n数据统计:")
        print(f"  - 节点数量: {len(data['nodes'])}")
        print(f"  - 边数量: {len(data['edges'])}")
        print(f"  - 输入类型: {len(data['input_units'])}")
        print(f"  - 输出类型: {len(data['output_units'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON 文件加载失败: {e}")
        return False


def test_connectome_creation():
    """测试连接组创建"""
    print("\n" + "=" * 60)
    print("测试 2: 连接组创建")
    print("=" * 60)
    
    try:
        from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
        
        print("✓ 导入 ConnectomeFromFlyWire 成功")
        
        # 创建连接组
        connectome = ConnectomeFromFlyWire(
            flywire_data_path="flyvis/connectome/flywire_v1.0.json",
            extent=15,
            n_syn_fill=1
        )
        
        print("✓ 连接组创建成功")
        
        # 获取统计信息
        stats = connectome.get_statistics()
        
        print(f"\n连接组统计:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
            
        # 验证关键属性
        assert hasattr(connectome, 'nodes'), "缺少 nodes 属性"
        assert hasattr(connectome, 'edges'), "缺少 edges 属性"
        assert hasattr(connectome, 'unique_cell_types'), "缺少 unique_cell_types 属性"
        
        print("\n✓ 所有必需属性存在")
        
        return True
        
    except Exception as e:
        print(f"✗ 连接组创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connectome_view():
    """测试连接组视图"""
    print("\n" + "=" * 60)
    print("测试 3: 连接组视图")
    print("=" * 60)
    
    try:
        from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
        from flyvis.connectome import ConnectomeView
        
        # 创建连接组
        connectome = ConnectomeFromFlyWire(
            flywire_data_path="flyvis/connectome/flywire_v1.0.json",
            extent=15,
            n_syn_fill=1
        )
        
        # 创建视图
        view = ConnectomeView(connectome)
        
        print("✓ ConnectomeView 创建成功")
        
        # 测试基本方法
        cell_types = view.cell_types_sorted
        print(f"  - 细胞类型数量: {len(cell_types)}")
        print(f"  - 前 10 个类型: {cell_types[:10]}")
        
        # 测试获取源和目标
        if len(cell_types) > 0:
            test_type = cell_types[0]
            sources = view.sources_list(test_type)
            targets = view.targets_list(test_type)
            print(f"\n  - {test_type} 的输入来源: {len(sources)} 种")
            print(f"  - {test_type} 的输出目标: {len(targets)} 种")
        
        print("\n✓ ConnectomeView 功能正常")
        
        return True
        
    except Exception as e:
        print(f"✗ ConnectomeView 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_key_cell_types():
    """测试关键细胞类型"""
    print("\n" + "=" * 60)
    print("测试 4: 关键细胞类型")
    print("=" * 60)
    
    try:
        from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
        
        connectome = ConnectomeFromFlyWire(
            flywire_data_path="flyvis/connectome/flywire_v1.0.json",
            extent=15
        )
        
        # 检查关键细胞类型
        cell_types = set(connectome.unique_cell_types.astype(str))
        
        # 输入类型（光感受器）
        input_types = ['R1-6', 'R7', 'R8']
        print("\n输入类型 (光感受器):")
        for ct in input_types:
            if ct in cell_types:
                print(f"  ✓ {ct}")
            else:
                print(f"  ✗ {ct} (缺失)")
        
        # 输出类型（运动检测）
        output_types = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
        print("\n输出类型 (运动检测):")
        for ct in output_types:
            if ct in cell_types:
                print(f"  ✓ {ct}")
            else:
                print(f"  ✗ {ct} (缺失)")
        
        # 关键中间神经元
        intermediate_types = ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9', 
                             'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'CT1']
        print("\n关键中间神经元:")
        found = 0
        for ct in intermediate_types:
            if ct in cell_types:
                print(f"  ✓ {ct}")
                found += 1
        
        print(f"\n找到 {found}/{len(intermediate_types)} 个关键中间神经元")
        
        return True
        
    except Exception as e:
        print(f"✗ 关键细胞类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_original():
    """与原始 FIB 连接组对比"""
    print("\n" + "=" * 60)
    print("测试 5: 与原始 FIB 连接组对比")
    print("=" * 60)
    
    try:
        from flyvis.connectome import ConnectomeFromAvgFilters
        from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
        
        # 加载原始连接组
        fib_connectome = ConnectomeFromAvgFilters(
            file='fib25-fib19_v2.2.json',
            extent=15
        )
        
        # 加载 FlyWire 连接组
        flywire_connectome = ConnectomeFromFlyWire(
            flywire_data_path="flyvis/connectome/flywire_v1.0.json",
            extent=15
        )
        
        print("\n对比统计:")
        print(f"{'指标':<30} {'FIB':<15} {'FlyWire':<15}")
        print("-" * 60)
        
        fib_stats = {
            'n_neurons': len(fib_connectome.nodes['index']),
            'n_synapses': len(fib_connectome.edges['source_index']),
            'n_cell_types': len(fib_connectome.unique_cell_types),
        }
        
        flywire_stats = flywire_connectome.get_statistics()
        
        for key in ['n_neurons', 'n_synapses', 'n_cell_types']:
            fib_val = fib_stats.get(key, 'N/A')
            flywire_val = flywire_stats.get(key, 'N/A')
            print(f"{key:<30} {str(fib_val):<15} {str(flywire_val):<15}")
        
        print("\n✓ 对比完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("FlyWire 连接组验证测试")
    print("=" * 60)
    
    tests = [
        ("JSON 格式", test_json_format),
        ("连接组创建", test_connectome_creation),
        ("连接组视图", test_connectome_view),
        ("关键细胞类型", test_key_cell_types),
        ("与 FIB 对比", compare_with_original),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n" + "🎉" * 20)
        print("所有测试通过！FlyWire 连接组已准备就绪！")
        print("🎉" * 20)
        print("\n下一步:")
        print("1. 安装完整的 Flyvis 依赖: pip install torch torchvision")
        print("2. 创建网络模型")
        print("3. 开始训练")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
