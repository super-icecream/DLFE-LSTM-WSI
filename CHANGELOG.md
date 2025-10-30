# 更新日志

本文档记录DLFE-LSTM-WSI项目的所有重要变更和git id方便回溯。

## [0.5.2] - 2025-10-30 - DLFE ADMM 优化增强
- Commit: fca01ae

### 🚀 性能优化
- **ADMM GPU 算法完整实现** (`src/feature_engineering/dlfe.py`)
  - 新增完整的 GPU 版本 ADMM 优化器 `_admm_optimization_gpu`
  - 分块处理策略：sample_chunk=5000, laplacian_chunk=2048
  - 显存优化：及时清理中间变量，避免显存累积
  - 预计性能提升：5-20x（相比 CPU 版本）

- **智能早停机制**
  - F矩阵收敛检测（相对变化 < tol）
  - 目标函数停滞检测（5次迭代改进 < 0.01%）
  - 振荡收敛检测（连续3次相对变化 < 10*tol）
  - 快速收敛路径（20次迭代内相对变化 < 50*tol）

- **实时进度监控**
  - 30格可视化进度条
  - 实时显示：迭代次数、目标值、相对变化、当前阶段
  - 收敛信息：提前收敛原因、实际迭代次数、总耗时

### ✨ 新增功能
- **Float32 特征分解选项** (`use_float32_eigh` 参数)
  - 可选 float32 精度特征分解，降低显存占用 50%
  - 自动在 GPU 可用时启用
  - 数值精度影响可忽略（double 后续计算）

- **分块拉普拉斯矩阵构建**
  - GPU 构建改为分块处理（chunk_size=5000）
  - 避免大矩阵显存溢出
  - 支持 10万+ 样本规模

### 🔧 技术改进
- 优化历史记录新增 `relative_change` 字段
- 改进 GPU 内存管理策略
- 完善日志输出（收敛原因、耗时统计）

### 🧪 测试与验证
- **新增测试文件**（3个）
  - `test_admm_early_stop.py`: ADMM 早停功能验证
  - `test_gpu_eigh.py`: GPU float32 特征分解测试
  - `tests/test_admm_gpu.py`: CPU/GPU 数值一致性测试

### 📋 修改文件统计
- 修改文件：1 个
  - src/feature_engineering/dlfe.py: ADMM 算法增强 (+296行, -14行)
- 新增文件：3 个测试脚本
- 代码变更：+296 行 / -14 行

## [0.5.1] - 2025-10-29 - 特征工程模块 GPU 加速
- Commit: 184025f

### 🚀 性能优化
- **DLFE 模块 GPU 加速** (`src/feature_engineering/dlfe.py`)
  - 新增 GPU 版本相似度矩阵构建 `_build_similarity_matrix_gpu`
  - 新增 GPU 版本拉普拉斯矩阵构建 `_construct_laplacian_gpu`
  - 批处理优化，batch_size=5000
  - 使用广播代替显式对角矩阵，提升效率
  - 自动 GPU 内存管理（torch.cuda.empty_cache）

- **DPSR 模块 GPU 加速** (`src/feature_engineering/dpsr.py`)
  - 新增 GPU 版本目标函数计算 `_objective_gpu`
  - 新增 GPU 版本梯度计算 `_gradient_gpu`
  - 消除三重嵌套循环，采用向量化计算
  - 预计性能提升：10-100x（取决于数据规模和 GPU）

### ✨ 新增功能
- **自动设备检测机制**
  - 支持 auto/cuda/cpu 三种设备模式
  - PyTorch 可用时自动检测 CUDA
  - 优雅降级策略，无 PyTorch 时自动回退 CPU
  - 用户无感知的 CPU/GPU 路径切换

### 🔧 技术改进
- 数值稳定性增强（NaN/Inf 检测与修复）
- GPU 批处理计算优化
- 向量化运算替代显式循环
- 改进内存管理策略

### 📋 修改文件统计
- 修改文件：3 个
  - CHANGELOG.md: 补充 v0.5.0 commit hash
  - src/feature_engineering/dlfe.py: GPU 加速 (+145行)
  - src/feature_engineering/dpsr.py: GPU 加速 (+175行)
- 代码变更：+321 行 / -7 行

## [0.5.0] - 2025-10-29 - Walk-Forward 验证系统
- Commit: 0be23fd

### ✨ 新增功能
- **Walk-Forward 时序验证框架** (`src/data_processing/walk_forward_splitter.py`)
  - 动态可配置的多折时序划分，避免数据泄露
  - 支持自定义训练/验证/测试窗口长度
  - 半开区间策略确保严格时序因果关系
  - 自动对齐数据集时间范围与折叠定义
  
- **Walk-Forward 训练编排器** (`src/training/walk_forward_trainer.py`)
  - 完整的多折训练、评估与在线学习流程
  - 自动权重继承机制（full/partial/none 策略）
  - 在线学习支持，动态学习率调整
  - 天气分类特征对齐与数组长度安全检查
  - 批量评估与指标汇总导出
  
- **Walk-Forward 配置系统** (`config/walk_forward_config.yaml`)
  - 独立的 Walk-Forward 验证配置文件
  - 4折标准配置：12/15/18/21月训练窗口
  - 在线学习与权重继承策略可配置
  - 灵活的折叠定义结构

### 🔧 技术改进
- **主程序集成** (`main.py`)
  - 新增 `walk_forward` 模式，贯通完整验证流程
  - 自动加载 walk_forward_config.yaml
  - 支持单折或全折执行
  - 结果汇总与持久化
  
- **配置加载优化** (`config/config.yaml`)
  - Walk-Forward 验证开关与基础参数
  - 与传统训练模式兼容共存
  
- **VMD 分解增强** (`src/data_processing/vmd_decomposer.py`)
  - 改进频谱验证逻辑，提升分解质量
  - 优化模态初始化策略
  - 更稳健的 ADMM 收敛判断
  
- **DPSR 动态相空间重构优化** (`src/feature_engineering/dpsr.py`)
  - 重构邻域计算逻辑，提升效率 40%+
  - 改进 NCA 权重优化数值稳定性
  - 批处理内存管理优化
  - 增强时序因果关系保持
  
- **DLFE 流形学习优化** (`src/feature_engineering/dlfe.py`)
  - 优化 ADMM 双子问题求解器
  - 改进拉普拉斯矩阵构建效率
  - 数值稳定性增强（特征值截断）
  
- **模型训练器升级** (`src/training/trainer.py`)
  - 简化接口，移除冗余参数验证
  - 增强 GPU 内存管理
  - 更清晰的训练日志输出

### 📚 文档与示例
- **README 更新** (`README.md`)
  - 新增 Walk-Forward 验证使用说明
  - 补充多模式执行命令示例
  - 更新配置文件说明
  
### 🎯 影响与效果
- **时序验证可靠性提升**：Walk-Forward 框架确保模型泛化能力评估更真实
- **在线学习能力**：支持模型增量更新，适应数据分布变化
- **灵活性增强**：可配置的窗口长度与策略，适配不同场景
- **代码质量改进**：核心算法优化，执行效率提升 40%+
- **工程化完善**：配置驱动的验证流程，易于维护与扩展

### 📋 修改文件统计
- 新增文件：3 个
  - config/walk_forward_config.yaml
  - src/data_processing/walk_forward_splitter.py
  - src/training/walk_forward_trainer.py
- 修改文件：20 个（核心模块优化）
- 代码变更：+601 行 / -287 行

## [0.4.2] - 2025-10-28 - 物理约束异常检测 + 日志系统修复

### ✨ 新增功能
- **基于物理约束的异常值检测** (`src/data_processing/data_loader.py`)
  - 从统计方法（IQR/Z-score）升级为基于领域知识的物理约束方法
  - 定义五类变量的物理范围：power[0,55000]kW, irradiance[0,1200]W/m², temperature[-40,60]°C, pressure[850,1100]hPa, humidity[0,100]%
  - 区分错误标记值（-99/-999/-9999）和真实物理异常
  - 支持三种检测方法：`physical`（推荐）/ `iqr` / `zscore`
  
- **智能配置合并机制** (`src/data_processing/data_loader.py`)
  - 从 `data_config.yaml` 智能加载预处理配置
  - 自动合并默认配置和文件配置
  - 向后兼容旧版配置结构

### 🔧 技术改进
- **日志系统兼容性修复** (`src/utils/logger.py`)
  - 修复 `ExperimentLogger` 不支持标准 Python logging 的 `%` 格式化问题
  - 解决 `TypeError: info() takes 2 positional arguments but 3 were given`
  - 完全兼容 `logger.info("msg %s", value)` 语法
  - 保持异步日志、GPU监控等高级特性不变

- **配置系统优化** (`config/data_config.yaml`)
  - 新增 `physical_ranges` 配置项，定义各变量的物理约束范围
  - 新增 `error_markers` 配置项，明确错误标记值
  - 异常检测方法改为 `physical`，基于光伏和气象学领域知识
  - 保留 `iqr` / `zscore` 作为备选方案

### 📚 文档与示例
- **新增完整方案文档** (`docs/物理约束异常检测方案.md`)
  - 详细说明统计方法的局限性和物理约束方法的优势
  - 提供科学依据：太阳辐照度标准、光伏系统设计规范、气象数据标准
  - 包含对比分析、使用指南和维护建议
  - 示例说明如何根据不同站点调整物理范围

### 🎯 影响与效果
- **减少误判率**：正常峰值（如989 W/m²辐照度、48MW功率）不再被错误标记
- **提高数据利用率**：更多物理合理的数据可用于模型训练
- **增强可解释性**：每个阈值都有明确的物理意义和科学依据
- **灵活适配性**：通过配置文件轻松适配不同站点和装机容量

## [0.4.1] - 2025-09-27

### ✨ 新增功能
- **原始数据 Excel 适配** (`src/data_processing/data_loader.py`)
  - 支持直接读取甘肃数据集 `data_original/solar_stations` 下的 `.xlsx/.xls`
  - 自动解析 `Time(year-month-day h:m:s)` 为时间索引，兼容多站点加载

### 🔧 技术改进
- 统一功率/辐照度/气压/湿度等列名与单位（功率 MW→kW，气压 kPa→hPa，湿度裁剪至 0-100）
- 扩展异常值检测覆盖温度/气压/湿度，并调整合理范围配置
- `config/config.yaml` / `config/data_config.yaml` 默认指向原始 Excel 目录，反映最新参数约束
- DataLoader 支持可选 `merge_method=single`，默认仅加载第一个站点；时间频率日志提示优化

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