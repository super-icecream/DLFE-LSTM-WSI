# DLFE-LSTM-WSI: 动态局部特征嵌入的光伏功率预测系统

## 项目简介

DLFE-LSTM-WSI是一个基于动态局部特征嵌入和长短期记忆网络的光伏功率预测系统。该系统通过替代昂贵的全天空成像设备，仅使用低成本的气象传感器（气压、湿度等）实现高精度的光伏功率预测。

## 核心技术

### 1. 多尺度特征提取架构
- **宏观层面**: 基于CI和WSI的双路径天气状态识别
- **中观层面**: 基于NCA优化的动态相空间重构(DPSR)
- **微观层面**: 基于ADMM算法的动态局部特征嵌入(DLFE)

### 2. 关键算法
- **VMD分解**: 变分模态分解，将功率信号分解为5个本征模态函数
- **CI计算**: 清晰度指数，基于天文参数计算太阳辐照度透过率
- **WSI计算**: 天气状态指数，基于气象参数的综合评估
- **NCA优化**: 邻域成分分析，学习动态特征权重
- **ADMM优化**: 交替方向乘子法，实现流形学习降维

### 3. 数据处理流程
```
原始数据(P,I,T,Pre,Hum) → VMD分解(5个IMF) → 双路径天气识别 →
DPSR重构(30维) → DLFE降维(30维) → LSTM预测 → 多天气子模型融合
```

## 项目结构

```
DLFE-LSTM-WSI/
├── config/                 # 配置文件
│   ├── config.yaml         # 主配置
│   ├── model_config.yaml   # 模型配置
│   └── data_config.yaml    # 数据配置
├── environment/            # 环境配置
│   ├── environment.yml     # Conda环境
│   ├── requirements.txt    # Python依赖
│   └── setup_env.sh       # 环境安装脚本
├── src/                    # 源代码
│   ├── data_processing/    # 数据处理模块
│   ├── feature_engineering/# 特征工程模块（算法核心）
│   ├── models/            # 模型定义模块
│   ├── training/          # 训练模块
│   └── evaluation/        # 评估模块
├── tests/                 # 测试代码
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── results/          # 结果输出
├── experiments/           # 实验记录
├── docs/                 # 文档
└── scripts/              # 脚本工具
```

## 技术特点

### 1. 参数隔离原则
严格遵循时序数据的参数隔离原则，所有预处理参数仅从训练集学习，防止数据泄露。

### 2. 时序完整性保护
采用70/20/10的严格时序划分，保持数据的时序连续性和因果关系。

### 3. 多天气自适应
针对晴天、多云、阴天三种天气状态构建专门的子模型，提高预测精度。

### 4. 算法创新性
- 首次将VMD分解与双路径天气识别相结合
- 创新性地使用NCA优化进行动态相空间重构
- 提出基于ADMM的动态局部特征嵌入方法

## 运行环境

- Python >= 3.8
- PyTorch >= 1.12
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Scikit-learn >= 1.0.0

## 快速开始

1. **环境安装**
```bash
cd environment
bash setup_env.sh            # 默认安装GPU版本
# 或者 CPU 环境
bash setup_env.sh --cpu
conda activate dlfe-lstm-wsi
```

2. **数据准备**
将甘肃光伏数据放置在 `data/raw/` 目录下

3. **配置修改**
根据实际需求修改 `config/` 目录下的配置文件

4. **运行训练 / 评估**
```bash
python main.py prepare --run-name demo
python main.py train --run-name demo
python main.py test --run-name demo

# 或使用脚本包装
python scripts/prepare_data.py --run-name demo
python scripts/train_model.py --run-name demo
python scripts/evaluate_model.py --run-name demo
```

## 数据格式

输入数据应包含以下列：
- `power` 或 `P`: 光伏功率 (kW)
- `irradiance` 或 `I`: 太阳辐照度 (W/m²)
- `temperature` 或 `T`: 温度 (°C)
- `pressure` 或 `Pre`: 气压 (hPa)
- `humidity` 或 `Hum`: 相对湿度 (%)
- `datetime`: 时间戳

## 模型性能

该系统在甘肃光伏数据集上的预测性能：
- 晴天条件：MAPE < 5%
- 多云条件：MAPE < 8%
- 阴天条件：MAPE < 12%

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 作者

DLFE-LSTM-WSI Team

## 致谢

感谢甘肃省提供的光伏发电数据支持。