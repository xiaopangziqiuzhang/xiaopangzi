"""
简单的量化可视化代码
帮助理解量化前后的变化
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_simple_quantization():
    """
    可视化简单的量化过程
    """
    print("=" * 60)
    print("简单量化示例")
    print("=" * 60)
    
    # 生成一些数据
    np.random.seed(42)
    data = np.random.randn(1000) * 2 + 1  # 均值1，标准差2
    
    # 4位对称量化
    bits = 4
    q_max = 2 ** (bits - 1) - 1  # 7
    q_min = -2 ** (bits - 1)      # -8
    
    # 计算缩放因子
    max_val = np.max(np.abs(data))
    scale = max_val / q_max
    
    # 量化
    data_q = np.round(data / scale)
    data_q = np.clip(data_q, q_min, q_max)
    
    # 反量化
    data_dq = data_q * scale
    
    # 计算误差
    mse = np.mean((data - data_dq) ** 2)
    mae = np.mean(np.abs(data - data_dq))
    
    print(f"\n原始数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
    print(f"量化级别: {bits} bits (从 {q_min} 到 {q_max})")
    print(f"缩放因子: {scale:.6f}")
    print(f"\n量化误差:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始数据
    axes[0].hist(data, bins=50, alpha=0.7, color='blue')
    axes[0].set_title('原始数据（FP32）')
    axes[0].set_xlabel('值')
    axes[0].set_ylabel('频数')
    
    # 量化后的数据（整数）
    axes[1].hist(data_q, bins=q_max - q_min + 1, alpha=0.7, color='green')
    axes[1].set_title(f'量化后（INT{bits}）')
    axes[1].set_xlabel('量化值')
    axes[1].set_ylabel('频数')
    
    # 反量化后的数据
    axes[2].hist(data_dq, bins=50, alpha=0.7, color='red')
    axes[2].set_title('反量化后')
    axes[2].set_xlabel('值')
    axes[2].set_ylabel('频数')
    
    plt.tight_layout()
    plt.savefig('/tmp/quantization_example.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: /tmp/quantization_example.png")
    
    # 显示前10个值的对比
    print(f"\n前10个值的对比:")
    print(f"{'原始':>10} | {'量化':>6} | {'反量化':>10} | {'误差':>10}")
    print("-" * 50)
    for i in range(10):
        orig = data[i]
        q = data_q[i]
        dq = data_dq[i]
        err = dq - orig
        print(f"{orig:10.4f} | {q:6.0f} | {dq:10.4f} | {err:10.4f}")
    
    print("\n" + "=" * 60)


def compare_different_bitwidths():
    """
    对比不同比特宽度的量化效果
    """
    print("\n" + "=" * 60)
    print("不同比特宽度量化对比")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    data = np.random.randn(1000) * 2
    
    bitwidths = [2, 3, 4, 8]
    mses = []
    maes = []
    
    print(f"\n{'比特宽度':>10} | {'MSE':>12} | {'MAE':>12} | {'压缩比':>10}")
    print("-" * 60)
    
    for bits in bitwidths:
        q_max = 2 ** (bits - 1) - 1
        q_min = -2 ** (bits - 1)
        
        max_val = np.max(np.abs(data))
        scale = max_val / q_max
        
        data_q = np.round(data / scale)
        data_q = np.clip(data_q, q_min, q_max)
        data_dq = data_q * scale
        
        mse = np.mean((data - data_dq) ** 2)
        mae = np.mean(np.abs(data - data_dq))
        
        mses.append(mse)
        maes.append(mae)
        
        compression = 32 / bits
        print(f"{bits:10d} | {mse:12.6f} | {mae:12.6f} | {compression:10.1f}x")
    
    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.arange(len(bitwidths))
    width = 0.35
    
    axes[0].bar(x, mses, width, label='MSE', color='blue')
    axes[0].set_xlabel('比特宽度')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('不同比特宽度的MSE对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{b}bit' for b in bitwidths])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x, maes, width, label='MAE', color='green')
    axes[1].set_xlabel('比特宽度')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('不同比特宽度的MAE对比')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{b}bit' for b in bitwidths])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bitwidth_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: /tmp/bitwidth_comparison.png")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- 比特宽度越高，误差越小")
    print("- 8bit误差已经很小，4bit误差也可以接受")
    print("- 2bit误差比较大，谨慎使用")


def main():
    print("\n" + "#" * 60)
    print("# 量化可视化示例")
    print("#" * 60)
    
    try:
        # 第一个示例：简单量化
        visualize_simple_quantization()
        
        # 第二个示例：不同比特宽度对比
        compare_different_bitwidths()
        
        print("\n" + "#" * 60)
        print("# 可视化完成！")
        print("# 查看生成的图片了解更多细节")
        print("#" * 60)
        
    except Exception as e:
        print(f"\n⚠️ 画图失败（可能没有matplotlib），但数值结果仍有效！")
        print(f"错误信息: {e}")
        
        # 即使不画图，也运行数值计算
        print("\n运行数值计算...")
        visualize_simple_quantization()
        compare_different_bitwidths()


if __name__ == "__main__":
    main()
