# MABe · 小鼠行为识别

<p align="center">
  <img src="assets/readme/overview.svg" width="100%" alt="MABe：从小鼠关键点轨迹到行为时间区间，采用多尺度特征和五折 LightGBM">
</p>

**从逐帧姿态轨迹中定位小鼠的个体动作与社交行为。** 我实现了运动学与交互特征设计、按视频分组的五折训练、行为阈值优化、可恢复训练和离线推理，输出“哪只鼠、对谁、做了什么、何时开始与结束”。

当前主线为 **LightGBM / fold555**：覆盖 **36 类行为、9 种关键点配置下的 84 个行为任务、420 个折模型**，已完成 Kaggle 隐藏测试集评分。

[预测效果](#预测效果) · [方法设计](#方法设计) · [快速体验](#快速体验) · [训练与推理](#训练与推理) · [结果证据](evidence/)

## 实测结果

| Kaggle Public | Kaggle Private | 评测版本 |
| :---: | :---: | :--- |
| **0.49911** | **0.48777** | Offline Submission V1 · 2026-09-19 赛后提交 |

采用竞赛官方 **MABe F Beta** 指标。提交编号 **56357157**，状态 **COMPLETE**；这组成绩对应本仓库的 LightGBM 推理方案。

[Kaggle 提交版本](https://www.kaggle.com/code/dadfawf/mabe-fold555-offline-submission?scriptVersionId=351047060) · [评分记录](evidence/kaggle-result.json) · [评分时的 Notebook 快照](evidence/scored-notebook.ipynb) · [84 个任务的 OOF 验证结果](evidence/oof-task-scores.csv)

## 预测效果

### 行为发生在什么时候？

下图取自一个**留出视频的折外预测（OOF）**：上方为 `sniff` 的逐帧概率，下方对照真实标注和按保存阈值生成的预测区间。模型拟合时没有使用该视频；图中概率保持原始输出，不做展示性平滑。

<img src="assets/readme/oof-event-detail.png" width="100%" alt="留出视频 2054411054 中 mouse1 对 mouse2 的 sniff 行为：原始 OOF 概率、阈值 0.21、真实标注与预测区间对照">

示例：`section 9 / sniff`，视频 `2054411054`，`mouse1 → mouse2`，第 4 折的 90 秒片段。[片段来源与帧范围](assets/readme/oof-figure-provenance.json)。这里展示单行为二分类结果，完整提交使用多行为竞争后输出区间。

### 多只小鼠的行为如何展开？

下图来自实际生成的提交文件：可见测试视频中，4 只小鼠的个体行为和有向交互被转换成 **943 段行为事件**。每一行表示一个施动者与目标组合，每个色块对应一个导出的预测区间。

<img src="assets/readme/prediction-timeline.png" width="100%" alt="测试视频 438887472 的预测行为时间轴：rear、approach、avoid、chase、attack 和 submit，共 943 个区间">

[对应的预测事件 CSV](examples/predicted-events.csv) 可直接绘图。该图展示公开可见测试视频的预测输出；上表成绩来自 Kaggle 对隐藏测试集的独立运行与评分。

## 方法设计

```text
关键点轨迹 + 视频元数据
          ↓
厘米尺度归一化 · 短缺失插值 · 缺失状态编码
          ↓
个体运动 / 姿态特征  +  双鼠相对运动 / 接触特征
          ↓
随帧率缩放的多尺度时间上下文 + 元数据编码
          ↓
按关键点配置与行为建模 · 视频分组五折 LightGBM
          ↓
折间概率平均 → 行为竞争 → 保存阈值 → 起止帧区间
```

**特征设计：把姿态变成可区分的行为信号。** 个体特征覆盖速度、曲率、姿态形状、头身运动解耦、身体轴向运动和高频微动；交互特征描述鼠间距离、相对朝向、接近/远离以及接触关系。坐标按像素/厘米比例归一化，时间窗口根据帧率缩放，适配不同采集配置。缺失处理填补不超过 0.25 秒的短空缺，同时显式编码关键点可见性与缺失连续长度。

**建模与验证：按视频留出，按行为决策。** 使用 `StratifiedGroupKFold(n_splits=5)`，同一视频的帧保持在同一折。按关键点配置、单体/双鼠类型及行为训练二分类器，每个任务使用 183–418 个特征；以 OOF 预测优化该行为的 F1 阈值。推理时平均五折概率，再进行最大概率行为选择和对应阈值过滤。时间上下文使用前后帧，面向离线视频分析。

**训练工程：让长任务能够跨会话完成。** 特征分批写入有容量上限的 Parquet 缓存，以 LightGBM Sequence 读取训练数据；每折保存检查点，任务完成后归档模型与阈值。时间、任务数和输出空间预算触发安全暂停，续训时校验数据指纹、特征定义、依赖版本和文件校验值，再复用已完成的折。

**推理工程：让训练产物可靠地变成提交文件。** 恢复训练时的元数据词表与特征列顺序，按配置缓存模型并释放训练期的 OOF 数组；逐视频导出行为区间，处理最后一段预测和帧号断点，并检查区间边界、重叠和行为合法性。Kaggle 提交使用离线依赖安装，全程关闭网络。

## 快速体验

无需下载竞赛数据或模型，先用仓库中的真实预测结果生成时间轴：

```bash
git clone https://github.com/Jim-jimu/mabe-behavior-recognition.git
cd mabe-behavior-recognition
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-viz.txt
python scripts/visualize_predictions.py
```

打开 `outputs/figures/prediction-timeline.png` 即可查看结果。这一步重绘已有预测；生成新的预测使用下面的训练与推理入口。

## 训练与推理

### 1. 准备环境与数据

```bash
python -m pip install -r requirements.txt
```

主线环境为 **Python 3.12、LightGBM 4.6.0、scikit-learn 1.8.0、NumPy 1.26.4**，完整版本见 [requirements.txt](requirements.txt)。训练与推理使用 CPU；macOS 若提示缺少 `libomp`，先执行 `brew install libomp`。

从 [Kaggle 竞赛数据页](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) 获取数据并组织为：

```text
data/mabe/
├── train.csv
├── test.csv
├── train_tracking/<lab_id>/<video_id>.parquet
├── train_annotation/<lab_id>/<video_id>.parquet
└── test_tracking/<lab_id>/<video_id>.parquet
```

### 2. 训练与续训

每轮默认最多训练 3 个新任务，使用 4 GiB 特征缓存和 6 GiB 输出预算；模型拟合自动使用可用 CPU 线程。运行后从日志末尾的 `OUTPUT` 找到本轮 `fold555` 目录。

```bash
MABE_DATA_DIR="$PWD/data/mabe" \
MABE_OUTPUT_BASE="$PWD/outputs/run-01" \
MABE_CACHE_DIR="$PWD/outputs/scratch" \
python src/train_fold555_disk_safe.py
```

若 `run_status.json` 为 `paused`，保留本轮产物，下一轮指定它作为续训来源，并使用新的输出目录：

```bash
MABE_DATA_DIR="$PWD/data/mabe" \
MABE_RESUME_DIR="/absolute/path/to/previous/fold555" \
MABE_OUTPUT_BASE="$PWD/outputs/run-02" \
MABE_CACHE_DIR="$PWD/outputs/scratch" \
python src/train_fold555_disk_safe.py
```

重复至 `run_status.json` 为 `complete`。每轮输出携带之前已完成的模型；确认新一轮保存完整后，可归档旧轮次释放空间。磁盘和运行时间充足时，可设置 `MABE_MAX_NEW_TASKS=0 MABE_MAX_HOURS=0` 取消分批与软时间限制。

<details>
<summary>资源参数与检查点内容</summary>

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MABE_MODEL_JOBS` | `-1` | 模型线程数，受环境可用 CPU 配额约束 |
| `MABE_MAX_NEW_TASKS` | `3` | 每轮新增任务上限，`0` 为不限 |
| `MABE_MAX_HOURS` | `3` | 软时间预算，在检查点边界暂停 |
| `MABE_CACHE_GB` | `4` | 特征磁盘缓存预算 |
| `MABE_OUTPUT_GB` | `6` | 当前输出目录树的空间预算 |

完整 `fold555` 包含 `thresholds.pkl`、类别词表、关键点配置、环境与训练配置，以及各任务的五折模型、特征列、OOF 预测和校验记录。特征生成保持串行以控制峰值占用，模型拟合使用多线程。任务与折检查点的完整性校验由训练入口自动执行。

</details>

### 3. 生成新的行为预测

```bash
python src/submit_fold555.py \
  --checkpoint /absolute/path/to/fold555 \
  --data data/mabe \
  --output outputs/inference \
  --threads 4
```

输出 `submission.csv` 和 `submission-report.json`。提交格式为：

```text
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
```

复核全部模型、校验 OOF 分数并保留逐帧预测概率：

```bash
python src/infer_fold555.py \
  --checkpoint /absolute/path/to/fold555 \
  --data data/mabe \
  --output outputs/verification \
  --threads 4 --audit-oof
```

Kaggle 隐藏测试集评分需要提交**关闭网络的 Notebook**，并在 `/kaggle/working/submission.csv` 生成结果。[评分版本快照](evidence/scored-notebook.ipynb) 保留了离线安装与输入校验流程；使用自己的模型时需替换输入挂载和校验值。

### 4. 重绘带标注的 OOF 对照图

```bash
python scripts/visualize_predictions.py \
  --oof /absolute/path/to/fold555/9/sniff/oof_predictions.parquet \
  --metadata data/mabe/train.csv \
  --output outputs/figures
```

## 结果如何对应到代码

| 证据 | 内容 |
| --- | --- |
| [Kaggle 评分记录](evidence/kaggle-result.json) | Public / Private 分数、提交 ID、Notebook 版本与源码摘要 |
| [OOF 任务结果](evidence/oof-task-scores.csv) | 84 个任务的阈值、特征数、样本数及复算 F1 |
| [检查点清单](evidence/checkpoint-manifest.json) | 模型、特征列、阈值与 OOF 文件的 SHA-256 |
| [推理复核记录](evidence/inference-check.json) | 420 个模型接口检查，可见测试的覆盖与输出一致性 |
| [预测示例](examples/predicted-events.csv) | 已验证的 943 段真实模型输出 |

OOF 表中的分数是**任务级二分类 F1**，阈值由同一套 OOF 预测选择；Kaggle 表中的分数是**隐藏测试集上的官方指标**。Kaggle 原始提交记录需要相应账号访问权限，仓库同时保存评分导出与精确版本快照。模型权重与原始竞赛数据存放在仓库外，可通过上面的训练入口生成自己的检查点。

```text
src/             LightGBM 训练、检查点校验与推理
scripts/         真实预测可视化
tests/          区间导出、训练中断恢复与资源预算检查
evidence/       评分记录、OOF 汇总、校验值与提交快照
examples/       可直接重绘的预测事件
assets/readme/  README 展示图与来源记录
code/           早期 XGBoost 训练实验与推理 Notebook
```

自动检查：`python -m unittest discover -s tests -v`。检查使用小型合成数据验证软件行为，实验成绩来自上述 Kaggle 与 OOF 记录。

`code/` 保留 XGBoost 实验分支，其依赖见 [code/requirements-xgboost.txt](code/requirements-xgboost.txt)，输入路径与模型挂载在相应脚本中配置。

数据与评价方法来自 [MABe Challenge](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)；指标定义见 [官方 MABe F Beta](https://www.kaggle.com/code/metric/mabe-f-beta)。
