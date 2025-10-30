import numpy as np

from src.feature_engineering.dlfe import DLFE


def main() -> None:
    """GPU float32 特征分解验证脚本。"""
    np.random.seed(0)
    X_test = np.random.randn(5000, 30)

    dlfe = DLFE(target_dim=10, max_iter=5, device='cuda', use_float32_eigh=True)
    dlfe.fit(X_test)

    print("✅ GPU特征分解测试通过")


if __name__ == "__main__":
    main()
