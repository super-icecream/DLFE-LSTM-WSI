# 更新日志

本文档记录DLFE-LSTM-WSI项目的所有重要变更。

## [未发布] - 开发中

### 📋 待实现
- [ ] 端到端真实数据回归测试
- [ ] 性能基准测试
- [ ] 推理服务化与部署方案

### ✨ 新增功能
- **原始数据 Excel 适配** (`src/data_processing/data_loader.py`)
  - 支持直接读取甘肃数据集 `data_original/solar_stations` 下的 `.xlsx/.xls`
  - 自动解析 `Time(year-month-day h:m:s)` 为时间索引，兼容多站点加载

### 🔧 技术改进
- 统一功率/辐照度/气压/湿度等列名与单位（功率 MW→kW，气压 kPa→hPa，湿度裁剪至 0-100）
- 扩展异常值检测覆盖温度/气压/湿度，并调整合理范围配置
- `config/config.yaml` / `config/data_config.yaml` 默认指向原始 Excel 目录，反映最新参数约束

### 📚 文档与示例
- 更新 `README.md` 的数据准备章节，补充 Excel 列映射、单位转换及路径配置说明

## [0.4.1] - 2025-09-27
- Commit: e81aba2c

### ✨ 新增功能
- **原始数据 Excel 适配** (`src/data_processing/data_loader.py`)
  - 支持直接读取甘肃数据集 `data_original/solar_stations` 下的 `.xlsx/.xls`
  - 自动解析 `Time(year-month-day h:m:s)` 为时间索引，兼容多站点加载

### 🔧 技术改进
- 统一功率/辐照度/气压/湿度等列名与单位（功率 MW→kW，气压 kPa→hPa，湿度裁剪至 0-100）
- 扩展异常值检测覆盖温度/气压/湿度，并调整合理范围配置
- `config/config.yaml` / `config/data_config.yaml` 默认指向原始 Excel 目录，反映最新参数约束

### 📚 文档与示例
- 更新 `README.md` 的数据准备章节，补充 Excel 列映射、单位转换及路径配置说明

## [0.4.0] - 2025-09-27 - 管线集成与评估增强

### ✨ 新增功能
- **统一主程序入口** (`main.py`)
  - 集成 prepare/train/test 三种模式，贯通数据、特征、训练、评估流水线
- **脚本对齐与实验自动化** (`scripts/`)
  - `prepare_data.py`、`train_model.py`、`evaluate_model.py` 与 `run_experiment.sh` 统一调用主入口
- **评估与可视化体系** (`src/evaluation/metrics.py`, `src/evaluation/visualizer.py`)
  - 支持多时域与多天气指标、显著性检验，输出 Markdown 报告及全套图表
- **GPU 优化训练组件** (`src/training/trainer.py`, `src/training/adaptive_optimizer.py`)
  - 三天气模型并行训练、混合精度、自适应调度与监控脚本

### 🔧 技术改进
- 统一 `src/__init__.py`、`src/utils/`、`src/models/` 导出接口与配置加载
- 数据处理链路新增缓存、持久化与天气标签贯通
- `config/config.yaml`、环境脚本与依赖清单更新以匹配最新参数

### 📚 文档与示例
- 更新 `README.md` 补充执行命令、环境安装与数据格式
- 新增 GPU 模块实现总结 (`docs/GPU_Models_Implementation_Summary.md`)
- 新增 `examples/gpu_model_demo.py` 演示 GPU 模块使用

## [0.3.0] - 2025-09-26 - 特征工程模块

### ✨ 新增功能
- **双路径天气识别系统** (`src/feature_engineering/weather_classifier.py`)
  - CI (清晰度指数) 计算，基于精确天文参数
  - WSI (天气状态指数) 计算，融合气象参数
  - 双路径融合决策机制，支持置信度加权
  - 在线权重更新，基于误差反馈自适应调整

- **动态相空间重构** (`src/feature_engineering/dpsr.py`)
  - NCA (邻域成分分析) 权重优化
  - 时序邻域构建，保持因果关系
  - 动态权重学习，适应时变特征
  - L-BFGS-B优化算法，带正则化约束

- **动态局部特征嵌入** (`src/feature_engineering/dlfe.py`)
  - ADMM算法实现流形学习
  - 高斯核相似度矩阵构建
  - 拉普拉斯矩阵局部结构保持
  - 双子问题迭代优化 (F子问题 + A子问题)

### 🔧 技术特性
- 完整的数学算法实现，数值稳定性优化
- 批处理支持，内存效率优化
- 综合测试用例，算法验证完备
- 参数保存/加载机制，支持模型持久化

## [0.2.0] - 2025-09-26 - 数据处理模块

### ✨ 新增功能
- **智能数据加载器** (`src/data_processing/data_loader.py`)
  - 多站点CSV数据加载与合并
  - 智能列名映射 (P→power, I→irradiance等)
  - 数据完整性检查，物理范围验证
  - 缺失值和异常值自动检测

- **严格时序数据分割** (`src/data_processing/data_splitter.py`)
  - 70/20/10时序划分，保持因果关系
  - 间隙感知分割，防止数据泄露
  - 滑动窗口样本生成，适配LSTM输入
  - 分割信息持久化与验证

- **全面预处理器** (`src/data_processing/preprocessor.py`)
  - 参数隔离原则，仅从训练集学习
  - 多种标准化方法 (MinMax/Standard/Robust/MaxAbs)
  - IQR和Z-score异常值检测
  - 高斯/移动平均/Savitzky-Golay平滑滤波

- **VMD变分模态分解** (`src/data_processing/vmd_decomposer.py`)
  - ADMM优化算法完整实现
  - 功率信号分解为5个IMF分量
  - 自适应模态数选择
  - 分解质量验证与可视化

### 🔧 技术改进
- 严格的参数隔离，防止数据泄露
- 时序完整性保护机制
- 数值稳定性优化
- 内存效率批处理

## [0.1.0] - 2025-09-26 - 项目初始化

### ✨ 新增功能
- **项目架构搭建**
  - 完整目录结构创建
  - 模块化设计，职责清晰分离
  - 配置管理系统

- **环境配置系统**
  - Conda环境配置 (`environment/environment.yml`)
  - Python依赖管理 (`environment/requirements.txt`)
  - 自动化安装脚本 (`environment/setup_env.sh`)

- **配置管理**
  - 主配置文件 (`config/config.yaml`)
  - 模型配置 (`config/model_config.yaml`)
  - 数据配置 (`config/data_config.yaml`)

- **开发工具**
  - Git版本控制配置
  - Python项目标准结构
  - 环境变量模板

### 🏗️ 基础架构
- 模块化代码组织
- 配置驱动设计
- 标准化开发环境
- 版本控制系统

---

## 版本说明

- **[主版本]**: 重大架构变更或不兼容更新
- **[次版本]**: 新功能添加，向后兼容
- **[修订版本]**: Bug修复和小幅改进

## 符号说明

- ✨ 新增功能
- 🔧 技术改进
- 🐛 Bug修复
- 📋 待实现功能
- 🏗️ 基础架构
- 📚 文档更新
- ⚡ 性能优化
- 🔒 安全更新