import logging
import numpy as np

from src.feature_engineering.dlfe import DLFE


def main() -> None:
    """Quick smoke test for ADMM early stopping on GPU path."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("测试ADMM早停优化")
    print("=" * 60)

    np.random.seed(42)
    X_test = np.random.randn(1000, 30)
    L_test = np.random.randn(1000, 1000)
    L_test = (L_test + L_test.T) / 2.0

    dlfe = DLFE(target_dim=10, max_iter=100, device='cuda')
    A_result = dlfe._admm_optimization_gpu(X_test, L_test)

    print("\n✅测试通过!")
    print(f"   结果形状: {A_result.shape}")
    print(f"   实际迭代: {dlfe.optimization_history['iterations']}")
    if dlfe.optimization_history['objective']:
        last_objective = dlfe.optimization_history['objective'][-1]
        print(f"   最终目标值: {last_objective:.4f}")


if __name__ == "__main__":
    main()
