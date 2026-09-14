# MABe Mouse Behavior Recognition

基于小鼠姿态轨迹的离线行为检测方案，使用运动学与交互特征、XGBoost 分类器及时间序列后处理，识别单体动作和小鼠之间的社交行为。

**技术栈：** Python · Polars · NumPy · XGBoost · scikit-learn · SciPy

[竞赛主页](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview) · [数据来源](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) · [官方评分实现](https://www.kaggle.com/code/metric/mabe-f-beta)

## 项目概览

输入为每帧每只小鼠的身体关键点坐标及视频元数据，输出为行为类别、施动者、目标及行为起止帧。代码定义 11 类单体行为与 26 类交互行为，训练时按实验室的实际标注构建分类任务。

| 环节 | 实现 |
| --- | --- |
| 轨迹清洗 | 身体形态异常过滤、速度异常过滤、线性插值与滚动平滑 |
| 单体特征 | 部位距离、速度、加速度、角速度、身体伸展、局部抖动与环境位置 |
| 交互特征 | 鼠间距离、相对朝向、接近速度、自我中心坐标与交互指标 |
| 时间上下文 | 多尺度滚动统计及前后帧时移特征 |
| 模型训练 | 按实验室和行为训练 XGBoost 二分类器，调整正样本权重 |
| 验证策略 | 按视频分组的三折 StratifiedGroupKFold，使用 OOF 预测选择行为阈值 |
| 推理后处理 | 折间概率平均、高斯平滑、阈值归一化竞争与最小时长过滤 |

## 方法流程

```mermaid
flowchart LR
    A[关键点轨迹] --> B[轨迹清洗]
    B --> C[单体与交互特征]
    C --> D[多尺度时间上下文]
    D --> E[XGBoost 行为分类]
    E --> F[概率平滑与行为竞争]
    F --> G[行为起止区间]
```

单体与交互特征分别描述个体运动和双方关系。距离使用像素/厘米比例换算，时间窗口结合视频帧率生成；lag/lead 与居中窗口提供动作发生前后的上下文，适用于离线轨迹分析。

## 代码结构

```text
code/
├── train_exp_1209_new_feature.py   # 训练实验：较低学习率
├── train_exp_1210_v6.py            # 训练实验：学习率与类别权重调整
└── mabe inference.ipynb           # Kaggle 推理与提交文件生成
requirements.txt                  # Python 依赖
```

两个训练实验共享相同的特征设计。1210 版本将单体/交互学习率从 0.002/0.003 调整为 0.02/0.03，并调整正类权重。推理 Notebook 汇集模型文件后，对各折预测取平均。

## 数据与模型

| 资源 | 获取方式 |
| --- | --- |
| 竞赛数据 | 从 [Kaggle Data 页面](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) 获取元数据、关键点轨迹和训练标注 |
| 评分脚本 | 从 [官方 MABe F Beta Notebook](https://www.kaggle.com/code/metric/mabe-f-beta) 的输出获取 `metric.py`；原训练代码对应版本 8 |
| 模型权重 | 运行训练脚本生成各实验室、行为和折的 `model.json` 与 `threshold.txt` |

数据和训练产物保存在本地或 Kaggle 中，仓库通过来源链接与训练代码提供获取入口。原推理 Notebook 的 `mabe-exp-1209` 和 `mabe-exp-1210-v6` 是实验模型挂载名，使用时应替换为实际模型所在目录。

## 运行

原 Notebook 使用 Python 3.11，指定 XGBoost 3.1.1。先安装依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 训练

准备以下数据结构：

```text
MABe-mouse-behavior-detection/
├── train.csv
├── test.csv
├── train_tracking/<lab_id>/<video_id>.parquet
├── train_annotation/<lab_id>/<video_id>.parquet
└── test_tracking/<lab_id>/<video_id>.parquet
```

在所选训练脚本开头设置 `INPUT_DIR`、`WORKING_DIR` 及 `metric.py` 的导入路径，并根据运行环境设置 XGBoost 设备参数，然后执行：

```bash
python code/train_exp_1210_v6.py
```

特征保存至 `self_features/` 和 `pair_features/`；模型和阈值保存至 `results/<lab_id>/<behavior>/fold_<n>/`，同时输出 OOF 预测。

### 推理

在 Kaggle 打开 `code/mabe inference.ipynb`，配置竞赛数据、XGBoost 安装包及训练权重的挂载路径，按顺序运行单元。默认输出为 `/kaggle/working/submission.csv`：

```text
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
```

算法文件保留原始竞赛实验代码。运行需要配套数据、评分脚本及训练产物；依赖清单按代码导入整理，完整环境复现需结合实际运行验证。
