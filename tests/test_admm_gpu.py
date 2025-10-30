import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.feature_engineering.dlfe import DLFE


def _build_laplacian(matrix_size: int) -> np.ndarray:
    """Construct a symmetric Laplacian matrix for testing."""
    rng = np.random.default_rng(42)
    Q = rng.random((matrix_size, matrix_size))
    Q = (Q + Q.T) / 2.0
    np.fill_diagonal(Q, 0.0)
    degrees = np.sum(Q, axis=1)
    L = np.diag(degrees) - Q
    return L


def test_admm_gpu_correctness():
    """Validate that GPU and CPU ADMM outputs match numerically."""
    print("\n" + "=" * 60)
    print("测试：ADMM数值一致性 (CPU vs GPU)")
    print("=" * 60)

    if torch is None or not torch.cuda.is_available():
        print("⚠️ 未检测到可用GPU，跳过该测试")
        return

    np.random.seed(123)
    torch.manual_seed(123)

    n_samples, n_features, target_dim = 512, 60, 12
    X = np.random.randn(n_samples, n_features)
    L = _build_laplacian(n_samples)

    dlfe_cpu = DLFE(target_dim=target_dim, device="cpu")
    A_cpu = dlfe_cpu.admm_optimization(X, L)

    dlfe_gpu = DLFE(target_dim=target_dim, device="cuda")
    dlfe_gpu.optimization_history = {'objective': [], 'constraint_violation': [], 'relative_change': [], 'iterations': 0}
    A_gpu = dlfe_gpu._admm_optimization_gpu(X, L)

    abs_diff = np.abs(A_cpu - A_gpu)
    rel_diff = abs_diff / (np.abs(A_cpu) + 1e-10)

    print(f"\n   最大绝对误差: {np.max(abs_diff):.2e}")
    print(f"   平均绝对误差: {np.mean(abs_diff):.2e}")
    print(f"   最大相对误差: {np.max(rel_diff):.2e}")
    print(f"   平均相对误差: {np.mean(rel_diff):.2e}")

    close = np.allclose(A_cpu, A_gpu, atol=1e-10, rtol=1e-8)
    print(f"\n   ✅ 数值等价性: {'通过' if close else '失败'}")

    assert close, "GPU 与 CPU 的 ADMM 结果不一致"
    print("=" * 60)


def test_admm_gpu_memory():
    """Measure GPU memory footprint during ADMM."""
    print("\n" + "=" * 60)
    print("测试：ADMM GPU 内存占用")
    print("=" * 60)

    if torch is None or not torch.cuda.is_available():
        print("⚠️ 未检测到可用GPU，跳过该测试")
        return

    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"   初始GPU内存: {initial_mem:.2f} GB")

    np.random.seed(7)
    n_samples, n_features, target_dim = 6000, 80, 16
    X = np.random.randn(n_samples, n_features)
    L = _build_laplacian(n_samples)

    dlfe_gpu = DLFE(target_dim=target_dim, device="cuda")
    dlfe_gpu.optimization_history = {'objective': [], 'constraint_violation': [], 'relative_change': [], 'iterations': 0}
    _ = dlfe_gpu._admm_optimization_gpu(X, L)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    final_mem = torch.cuda.memory_allocated() / 1024 ** 3

    print(f"   峰值GPU内存: {peak_mem:.2f} GB")
    print(f"   结束GPU内存: {final_mem:.2f} GB")
    print(f"   内存增量: {peak_mem - initial_mem:.2f} GB")

    assert peak_mem < 4.0, f"GPU内存占用过高：{peak_mem:.2f} GB"
    print("   ✅ 内存占用满足 < 4 GB 的预期")
    print("=" * 60)


if __name__ == "__main__":
    test_admm_gpu_correctness()
    test_admm_gpu_memory()
