

import datetime
import gc
import itertools
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

sys.path.append("/root/.cache/kagglehub/notebooks/metric/mabe-f-beta/output/versions/8")
from metric import score

# const
INPUT_DIR = Path("/root/.cache/kagglehub/competitions/MABe-mouse-behavior-detection")
TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

WORKING_DIR = Path("/content/exp_1209_new_feature")

INDEX_COLS = [
    "video_id",
    "agent_mouse_id",
    "target_mouse_id",
    "video_frame",
]

BODY_PARTS = [
    "ear_left",
    "ear_right",
    "nose",
    "neck",
    "body_center",
    "lateral_left",
    "lateral_right",
    "hip_left",
    "hip_right",
    "tail_base",
    "tail_tip",
]

SELF_BEHAVIORS = [
    "biteobject",
    "climb",
    "dig",
    "exploreobject",
    "freeze",
    "genitalgroom",
    "huddle",
    "rear",
    "rest",
    "run",
    "selfgroom",
]

PAIR_BEHAVIORS = [
    "allogroom",
    "approach",
    "attack",
    "attemptmount",
    "avoid",
    "chase",
    "chaseattack",
    "defend",
    "disengage",
    "dominance",
    "dominancegroom",
    "dominancemount",
    "ejaculate",
    "escape",
    "flinch",
    "follow",
    "intromit",
    "mount",
    "reciprocalsniff",
    "shepherd",
    "sniff",
    "sniffbody",
    "sniffface",
    "sniffgenital",
    "submit",
    "tussle",
]

# read data
train_dataframe = pl.read_csv(INPUT_DIR / "train.csv")

# preprocess behavior labels
train_behavior_dataframe = (
    train_dataframe.filter(pl.col("behaviors_labeled").is_not_null())
    .select(
        pl.col("lab_id"),
        pl.col("video_id"),
        pl.col("behaviors_labeled").map_elements(eval, return_dtype=pl.List(pl.Utf8)).alias("behaviors_labeled_list"),
    )
    .explode("behaviors_labeled_list")
    .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
    .select(
        pl.col("lab_id"),
        pl.col("video_id"),
        pl.col("behaviors_labeled_element").str.split(",").list[0].str.replace_all("'", "").alias("agent"),
        pl.col("behaviors_labeled_element").str.split(",").list[1].str.replace_all("'", "").alias("target"),
        pl.col("behaviors_labeled_element").str.split(",").list[2].str.replace_all("'", "").alias("behavior"),
    )
)

train_self_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(SELF_BEHAVIORS))
train_pair_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(PAIR_BEHAVIORS))

import itertools
import numpy as np
import polars as pl

# 定义身体部位列表
BODY_PARTS = [
    "ear_left", "ear_right", "nose", "neck", "body_center",
    "lateral_left", "lateral_right", "hip_left", "hip_right",
    "tail_base", "tail_tip",
]

# 定义用于计算二阶形态特征的关键部位对
# 这些对的变化能反映具体的单体行为（如伸展、蜷缩、理毛）
MORPH_PAIRS = [
    ("nose", "tail_base"),     # 身体全长 (Rear, Huddle)
    ("nose", "body_center"),   # 前半身伸缩 (Investigate)
    ("nose", "hip_left"),      # 扭头 (Grooming)
    ("nose", "hip_right"),     # 扭头 (Grooming)
    ("lateral_left", "lateral_right") # 身体宽度 (呼吸/姿态挤压)
]

def make_self_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    fps = metadata["frames_per_second"]
    pix_per_cm = metadata["pix_per_cm_approx"]
    video_id = metadata["video_id"]

    # 场地参数
    arena_w = metadata.get("arena_width_cm", 50.0) * pix_per_cm
    arena_h = metadata.get("arena_height_cm", 50.0) * pix_per_cm

    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    # --- 辅助计算函数 ---
    def get_w(period_ms):
        return max(1, int(round(period_ms * fps / 1000.0)))

    def calc_dist(x1, y1, x2, y2):
        if isinstance(x1, str): x1 = pl.col(x1)
        if isinstance(y1, str): y1 = pl.col(y1)
        if isinstance(x2, str): x2 = pl.col(x2)
        if isinstance(y2, str): y2 = pl.col(y2)
        return ((x1 - x2).pow(2) + (y1 - y2).pow(2)).sqrt() / pix_per_cm

    def calc_speed(bp, period_ms):
        w = get_w(period_ms)
        d = ((pl.col(f"agent_x_{bp}").diff()).pow(2) + (pl.col(f"agent_y_{bp}").diff()).pow(2)).sqrt()
        return (d / pix_per_cm * fps).rolling_mean(window_size=w, center=True)

        # 频域/高频能量特征 ---
    def calc_high_freq_energy(bp, period_ms):
        """计算高频震动能量 (代替 FFT): 加速度的绝对值在窗口内的均值"""
        w = get_w(period_ms)
        # 1. 速度
        vx = pl.col(f"agent_x_{bp}").diff()
        vy = pl.col(f"agent_y_{bp}").diff()
        # 2. 加速度 (变化率)
        acc_mag = (vx.diff().pow(2) + vy.diff().pow(2)).sqrt()
        # 3. 能量 (窗口均值)
        return (acc_mag / pix_per_cm * (fps**2)).rolling_mean(window_size=w, center=True)

    def calc_accel(speed_col_name, period_ms):
        w = get_w(period_ms)
        return (pl.col(speed_col_name).diff().abs() * fps).rolling_mean(window_size=w, center=True)

    def calc_heading():
        return pl.arctan2(
            pl.col("agent_y_nose") - pl.col("agent_y_tail_base"),
            pl.col("agent_x_nose") - pl.col("agent_x_tail_base")
        )

    def calc_angular_vel(heading_col_name, period_ms):
        w = get_w(period_ms)
        diff = pl.col(heading_col_name).diff()
        # 处理角度跳变 (-pi 到 pi)
        diff_wrapped = (diff + np.pi).mod(2 * np.pi) - np.pi
        return (diff_wrapped.abs() * fps).rolling_mean(window_size=w, center=True)

    # --- 挖掘特征函数 ---
    def calc_wall_dist():
        dist_x = pl.min_horizontal([pl.col("agent_x_body_center"), pl.lit(arena_w) - pl.col("agent_x_body_center")])
        dist_y = pl.min_horizontal([pl.col("agent_y_body_center"), pl.lit(arena_h) - pl.col("agent_y_body_center")])
        return pl.min_horizontal([dist_x, dist_y]) / pix_per_cm

    def calc_spine_angle():
        v1x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        v1y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        v2x = pl.col("agent_x_tail_base") - pl.col("agent_x_body_center")
        v2y = pl.col("agent_y_tail_base") - pl.col("agent_y_body_center")
        dot = v1x * v2x + v1y * v2y
        mag = (v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt()
        return (dot / (mag + 1e-6))

    def calc_tortuosity(period_ms):
        w = get_w(period_ms)
        step_dist = ((pl.col("agent_x_body_center").diff()).pow(2) + (pl.col("agent_y_body_center").diff()).pow(2)).sqrt()
        path_len = step_dist.rolling_sum(window_size=w, center=True)
        disp = ((pl.col("agent_x_body_center").diff(n=w)).pow(2) + (pl.col("agent_y_body_center").diff(n=w)).pow(2)).sqrt()
        return path_len / (disp + 1e-6)

    def calc_jitter(period_ms):
        w = get_w(period_ms)
        # 计算 鼻尖相对于身体中心 的高频颤动
        rel_x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        rel_y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        rel_speed = ((rel_x.diff()).pow(2) + (rel_y.diff()).pow(2)).sqrt()
        return (rel_speed / pix_per_cm * fps).rolling_mean(window_size=w, center=True)

    def calc_polygon_area():
        # 使用鞋带公式 (Shoelace formula) 计算由 nose, ears, hips, tail 构成的多边形面积
        # 简化版：Nose -> EarL -> HipL -> TailBase -> HipR -> EarR -> Nose
        x_cols = ["agent_x_nose", "agent_x_ear_left", "agent_x_hip_left", "agent_x_tail_base", "agent_x_hip_right", "agent_x_ear_right"]
        y_cols = ["agent_y_nose", "agent_y_ear_left", "agent_y_hip_left", "agent_y_tail_base", "agent_y_hip_right", "agent_y_ear_right"]

        # 构建表达式 (x1y2 - y1x2) + ...
        expr = pl.lit(0)
        for i in range(len(x_cols)):
            j = (i + 1) % len(x_cols)
            expr = expr + (pl.col(x_cols[i]) * pl.col(y_cols[j]) - pl.col(y_cols[i]) * pl.col(x_cols[j]))

        return (expr.abs() * 0.5) / (pix_per_cm * pix_per_cm)

    # --- 数据准备 ---
    group_centroid = tracking.group_by("video_frame").agg(
        pl.col("x").mean().alias("group_center_x"),
        pl.col("y").mean().alias("group_center_y")
    ).sort("video_frame")

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])

    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    mice_ids = range(1, n_mice + 1)
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in mice_ids}

    result = []

    for agent_mouse_id in mice_ids:
        # A. Skeleton
        result_element = pl.DataFrame(
            {
                "video_id": video_id,
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": -1,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        # B. Prepare Data
        curr_pivot = pivot_trackings[agent_mouse_id]
        pivot_cols = curr_pivot.columns
        exprs = [pl.col("video_frame")]
        missing_cols = []

        for bp in BODY_PARTS:
            if f"x_{bp}" in pivot_cols:
                exprs.append(pl.col(f"x_{bp}").alias(f"agent_x_{bp}"))
            else:
                missing_cols.append(pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}"))
            if f"y_{bp}" in pivot_cols:
                exprs.append(pl.col(f"y_{bp}").alias(f"agent_y_{bp}"))
            else:
                missing_cols.append(pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}"))

        agent_df = curr_pivot.select(exprs)
        if missing_cols:
            agent_df = agent_df.with_columns(missing_cols)

        ##
        agent_df = preprocess_tracking(agent_df, metadata)

        agent_df = agent_df.join(group_centroid, on="video_frame", how="left")

        # 预计算向量
        agent_df = agent_df.with_columns(
            calc_heading().alias("temp_heading"),
            calc_dist("agent_x_nose", "agent_y_nose", "agent_x_tail_base", "agent_y_tail_base").alias("temp_len")
        )

        # C. Select Features (基础特征 + 依赖坐标的二阶特征)
        # 我们将 "Deformation Velocity" 和 "Posture Jitter" 移到这里，因为它们需要 agent_x/y 坐标
        features = agent_df.select(
            pl.col("video_frame"),
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(-1).alias("target_mouse_id"),

            # 1. 身体各部位两两之间的距离
            *[
                calc_dist(f"agent_x_{bp1}", f"agent_y_{bp1}", f"agent_x_{bp2}", f"agent_y_{bp2}")
                .alias(f"aa__{bp1}__{bp2}__distance")
                for bp1, bp2 in itertools.combinations(BODY_PARTS, 2)
            ],

            # 2. 各部位速度
            *[
                calc_speed(bp, ms).alias(f"agent__{bp}__speed_{ms}ms")
                for bp in ["ear_left", "ear_right", "tail_base", "body_center", "nose"]
                for ms in [200, 500, 1000]
            ],

            # 3. 高级挖掘特征
            pl.col("temp_heading").alias("agent__heading"),
            pl.col("temp_len").alias("agent__body_length"),
            (pl.col("temp_len") / (calc_dist("agent_x_ear_left", "agent_y_ear_left", "agent_x_ear_right", "agent_y_ear_right") + 1e-6)).alias("agent__elongation"),

            calc_wall_dist().alias("agent__dist_to_wall"), # 趋墙性
            calc_spine_angle().alias("agent__spine_angle"), # 姿态
            calc_dist("agent_x_body_center", "agent_y_body_center", "group_center_x", "group_center_y").alias("agent__dist_to_group_center"), # 社交上下文
            calc_tortuosity(500).alias("agent__tortuosity_500ms"), # 路径曲折度
            calc_jitter(200).alias("agent__nose_jitter_200ms"), # 微动作
            calc_angular_vel("temp_heading", 200).alias("agent__ang_vel_200ms"), # 角速度

            # --- C. 形变速度 (Deformation Velocity) ---
            # 依赖原始坐标，必须在此计算
            *[
                (calc_dist(f"agent_x_{bp1}", f"agent_y_{bp1}", f"agent_x_{bp2}", f"agent_y_{bp2}")
                 .diff() * fps / pix_per_cm)
                .rolling_mean(get_w(200), center=True)
                .alias(f"agent__{bp1}__{bp2}__stretch_vel_200ms")
                for bp1, bp2 in MORPH_PAIRS
            ],

            # --- D. 姿态抖动/剧烈度 (Posture Jitter) ---
            # 依赖原始坐标，必须在此计算
            *[
                calc_dist(f"agent_x_{bp1}", f"agent_y_{bp1}", f"agent_x_{bp2}", f"agent_y_{bp2}")
                .rolling_std(get_w(500), center=True)
                .alias(f"agent__{bp1}__{bp2}__pose_jitter_500ms")
                for bp1, bp2 in MORPH_PAIRS
            ],

            calc_polygon_area().alias("agent__body_area"),

            ## 频域/能量特征 (Grooming, Scratching 关键)
            calc_high_freq_energy("nose", 200).alias("agent__nose__energy_200ms"),
            calc_high_freq_energy("ear_left", 200).alias("agent__ear_l__energy_200ms"),
            calc_high_freq_energy("tail_base", 200).alias("agent__tail__energy_200ms"),
        )

        # 4. 二次加工特征 (依赖步骤3生成的新特征)
        features = features.with_columns(
            # --- A. 加速度 (Acceleration) ---
            calc_accel("agent__body_center__speed_500ms", 500).alias("agent__body_center__accel_500ms"),
            calc_accel("agent__nose__speed_200ms", 200).alias("agent__nose__accel_200ms"),

            # --- B. 速度的波动率 (Speed Stability) ---
            pl.col("agent__body_center__speed_200ms").rolling_std(get_w(500), center=True).alias("agent__speed_std_500ms"),

            # --- E. 复合特征交互 ---
            # 1. Grooming Ratio: (局部抖动 / (整体速度 + 1))
            (pl.col("agent__nose_jitter_200ms").fill_null(0) / (pl.col("agent__body_center__speed_200ms").fill_null(0) + 1.0))
            .alias("agent__interaction__grooming_ratio"),

            # 2. Agility Index: 速度 * 角速度
            (pl.col("agent__body_center__speed_200ms").fill_null(0) * pl.col("agent__ang_vel_200ms").fill_null(0))
            .alias("agent__interaction__agility_index"),

            # 3. Rearing Score Estimate: 身体长 / (速度 + 1)
            (pl.col("agent__body_length").fill_null(0) / (pl.col("agent__body_center__speed_500ms").fill_null(0) + 1.0))
            .alias("agent__interaction__rear_score"),
        )

        # [NEW] 2. 简单的时间平滑 (Long Context)
        # 计算关键特征在 1秒 (30帧) 窗口内的均值和标准差
        # 这样模型知道"它是刚刚开始跑，还是已经跑了一会儿了"
        long_window = get_w(1000)
        cols_to_smooth = [c for c in features.columns if "speed" in c or "energy" in c or "area" in c]

        features = features.with_columns([
            pl.col(c).rolling_mean(long_window, center=True).alias(f"{c}_avg_1s") for c in cols_to_smooth
        ] + [
            pl.col(c).rolling_std(long_window, center=True).alias(f"{c}_std_1s") for c in cols_to_smooth
        ])
        features = features.with_columns([
            pl.col("agent__body_center__speed_500ms").shift(15).alias("agent__speed_lag_15"), # 约0.5秒前
            pl.col("agent__body_center__speed_500ms").shift(-15).alias("agent__speed_lead_15"), # 约0.5秒后

            pl.col("agent__body_center__speed_500ms").shift(10).alias("agent__speed_lag_10"), # 约0.5秒前
            pl.col("agent__body_center__speed_500ms").shift(-10).alias("agent__speed_lead_10"), # 约0.5秒后

            pl.col("agent__body_center__speed_500ms").shift(5).alias("agent__speed_lag_5"), # 约0.5秒前
            pl.col("agent__body_center__speed_500ms").shift(-5).alias("agent__speed_lead_5"), # 约0.5秒后

            # 你也可以加加速度的滞后
            pl.col("agent__body_center__accel_500ms").shift(15).alias("agent__accel_lag_15"),
            pl.col("agent__body_center__accel_500ms").shift(-15).alias("agent__accel_lead_15"),

            pl.col("agent__body_center__accel_500ms").shift(10).alias("agent__accel_lag_10"),
            pl.col("agent__body_center__accel_500ms").shift(-10).alias("agent__accel_lead_10"),

            pl.col("agent__body_center__accel_500ms").shift(5).alias("agent__accel_lag_5"),
            pl.col("agent__body_center__accel_500ms").shift(-5).alias("agent__accel_lead_5"),
        ])
        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")

import itertools
import numpy as np
import polars as pl

BODY_PARTS = [
    "ear_left", "ear_right", "nose", "neck", "body_center",
    "lateral_left", "lateral_right", "hip_left", "hip_right",
    "tail_base", "tail_tip",
]

# 关键交互部位对
KEY_INTERACTION_PAIRS = [
    ("nose", "nose"),
    ("nose", "tail_base"),
    ("nose", "body_center"),
    ("nose", "hip_left"),
    ("nose", "hip_right"),
    ("body_center", "body_center"),
    ("tail_base", "tail_base"),
]

def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:

    fps = metadata["frames_per_second"]
    pix_per_cm = metadata["pix_per_cm_approx"]
    video_id = metadata["video_id"]

    # --- 辅助函数 ---
    def get_w(period_ms):
        return max(1, int(round(period_ms * fps / 1000.0)))

    def calc_dist(x1, y1, x2, y2):
        if isinstance(x1, str): x1 = pl.col(x1)
        if isinstance(y1, str): y1 = pl.col(y1)
        if isinstance(x2, str): x2 = pl.col(x2)
        if isinstance(y2, str): y2 = pl.col(y2)
        return ((x1 - x2).pow(2) + (y1 - y2).pow(2)).sqrt() / pix_per_cm

    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        return calc_dist(
            f"{agent_or_target_1}_x_{body_part_1}", f"{agent_or_target_1}_y_{body_part_1}",
            f"{agent_or_target_2}_x_{body_part_2}", f"{agent_or_target_2}_y_{body_part_2}"
        )

    def body_part_speed(agent_or_target, body_part, period_ms):
        w = get_w(period_ms)
        return (
            ((pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2)
             + (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)).sqrt()
            / pix_per_cm * fps
        ).rolling_mean(window_size=w, center=True)

    def calc_heading(prefix):
        return pl.arctan2(
            pl.col(f"{prefix}_y_nose") - pl.col(f"{prefix}_y_tail_base"),
            pl.col(f"{prefix}_x_nose") - pl.col(f"{prefix}_x_tail_base")
        )

    def calc_relative_heading():
        agent_heading = calc_heading("agent")
        heading_to_target = pl.arctan2(
            pl.col("target_y_body_center") - pl.col("agent_y_body_center"),
            pl.col("target_x_body_center") - pl.col("agent_x_body_center")
        )
        diff = agent_heading - heading_to_target
        return (diff + np.pi).mod(2 * np.pi) - np.pi

    def calc_mutual_facing():
        agent_heading = calc_heading("agent")
        target_heading = calc_heading("target")
        diff = (agent_heading - target_heading).abs()
        return (diff - np.pi).abs()

    def calc_approach_speed(period_ms):
        w = get_w(period_ms)
        dist = body_parts_distance("agent", "body_center", "target", "body_center")
        return (dist.diff() * fps).rolling_mean(w, center=True)

    def calc_relative_speed():
        agent_vx = pl.col("agent_x_body_center").diff()
        agent_vy = pl.col("agent_y_body_center").diff()
        target_vx = pl.col("target_x_body_center").diff()
        target_vy = pl.col("target_y_body_center").diff()
        rel_vx = agent_vx - target_vx
        rel_vy = agent_vy - target_vy
        return (rel_vx.pow(2) + rel_vy.pow(2)).sqrt() / pix_per_cm * fps

    def elongation(prefix):
        d1 = body_parts_distance(prefix, "nose", prefix, "tail_base")
        d2 = body_parts_distance(prefix, "ear_left", prefix, "ear_right")
        return d1 / (d2 + 1e-6)

    def body_angle(prefix):
        v1x = pl.col(f"{prefix}_x_nose") - pl.col(f"{prefix}_x_body_center")
        v1y = pl.col(f"{prefix}_y_nose") - pl.col(f"{prefix}_y_body_center")
        v2x = pl.col(f"{prefix}_x_tail_base") - pl.col(f"{prefix}_x_body_center")
        v2y = pl.col(f"{prefix}_y_tail_base") - pl.col(f"{prefix}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-6)

    def calc_egocentric(agent, target):
        ax = pl.col(f"{agent}_x_nose") - pl.col(f"{agent}_x_tail_base")
        ay = pl.col(f"{agent}_y_nose") - pl.col(f"{agent}_y_tail_base")
        a_len = (ax.pow(2) + ay.pow(2)).sqrt() + 1e-6
        tx = pl.col(f"{target}_x_body_center") - pl.col(f"{agent}_x_body_center")
        ty = pl.col(f"{target}_y_body_center") - pl.col(f"{agent}_y_body_center")
        longitudinal = (ax * tx + ay * ty) / a_len
        lateral = (ax * ty - ay * tx) / a_len
        return [
            (longitudinal / pix_per_cm).alias(f"{agent}_to_{target}__long_dist"),
            (lateral / pix_per_cm).alias(f"{agent}_to_{target}__lat_dist_signed"),
            (lateral / pix_per_cm).abs().alias(f"{agent}_to_{target}__lat_dist"),
        ]

    def calc_facing_angle(agent, target):
        ah_x = pl.col(f"{agent}_x_nose") - pl.col(f"{agent}_x_tail_base")
        ah_y = pl.col(f"{agent}_y_nose") - pl.col(f"{agent}_y_tail_base")
        at_x = pl.col(f"{target}_x_body_center") - pl.col(f"{agent}_x_body_center")
        at_y = pl.col(f"{target}_y_body_center") - pl.col(f"{agent}_y_body_center")
        dot = ah_x * at_x + ah_y * at_y
        mag = (ah_x.pow(2) + ah_y.pow(2)).sqrt() * (at_x.pow(2) + at_y.pow(2)).sqrt()
        return dot / (mag + 1e-6)

    # --- 数据准备 ---
    n_mice = sum([metadata[f"mouse{i}_strain"] is not None for i in range(1, 5)])
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])

    pivot_trackings = {
        mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id)
        for mouse_id in range(1, n_mice + 1)
    }

    result = []

    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        result_element = pl.DataFrame({
            "video_id": video_id,
            "agent_mouse_id": agent_mouse_id,
            "target_mouse_id": target_mouse_id,
            "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
        }, schema={
            "video_id": pl.Int32,
            "agent_mouse_id": pl.Int8,
            "target_mouse_id": pl.Int8,
            "video_frame": pl.Int32,
        })

        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(pl.col("video_frame"), pl.exclude("video_frame").name.prefix("agent_"))
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"), pl.exclude("video_frame").name.prefix("target_")
                ),
                on="video_frame",
                how="inner",
            )
        )

        columns = merged_pivot.columns
        merged_pivot = merged_pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}")
              for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}")
              for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_x_{bp}")
              for bp in BODY_PARTS if f"target_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_y_{bp}")
              for bp in BODY_PARTS if f"target_y_{bp}" not in columns],
        )

        merged_pivot = preprocess_tracking(merged_pivot, metadata)

        # ========== 第一阶段：基础特征 ==========
        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),

            # --- 2. 全部部位距离（用于兼容性）---
            *[
                body_parts_distance("agent", agent_bp, "target", target_bp)
                .alias(f"at__{agent_bp}__{target_bp}__dist")
                for agent_bp, target_bp in itertools.product(BODY_PARTS, repeat=2)
            ],

            # --- 3. 最小距离特征 ---
            pl.min_horizontal([
                body_parts_distance("agent", "nose", "target", bp)
                for bp in ["nose", "ear_left", "ear_right", "neck", "body_center"]
            ]).alias("agent_nose__min_dist_to_target_head"),

            pl.min_horizontal([
                body_parts_distance("agent", "nose", "target", bp)
                for bp in ["hip_left", "hip_right", "tail_base"]
            ]).alias("agent_nose__min_dist_to_target_rear"),

            # --- 4. 速度特征 ---
            *[
                body_part_speed("agent", bp, ms).alias(f"agent__{bp}__speed_{ms}ms")
                for bp in ["body_center", "nose", "tail_base", "ear_left", "ear_right"]
                for ms in [200, 500, 1000]
            ],
            *[
                body_part_speed("target", bp, ms).alias(f"target__{bp}__speed_{ms}ms")
                for bp in ["body_center", "nose", "tail_base", "ear_left", "ear_right"]
                for ms in [200, 500, 1000]
            ],

            # --- 5. 相对速度 ---
            calc_relative_speed().alias("relative_speed"),

            # --- 6. 朝向特征 ---
            calc_heading("agent").alias("agent_heading"),
            calc_heading("target").alias("target_heading"),
            calc_relative_heading().alias("agent_relative_heading"),
            calc_mutual_facing().alias("mutual_facing_angle"),

            # --- 7. 自身中心坐标系 ---
            *calc_egocentric("agent", "target"),
            *calc_egocentric("target", "agent"),

            # --- 8. Facing Score ---
            calc_facing_angle("agent", "target").alias("agent_facing_score"),
            calc_facing_angle("target", "agent").alias("target_facing_score"),

            # --- 9. 接近速度 ---
            calc_approach_speed(200).alias("approach_speed_200ms"),
            calc_approach_speed(500).alias("approach_speed_500ms"),

            # --- 10. 姿态特征 ---
            elongation("agent").alias("agent_elongation"),
            elongation("target").alias("target_elongation"),
            body_angle("agent").alias("agent_body_angle"),
            body_angle("target").alias("target_body_angle"),
        )

        # ========== 第二阶段：派生特征（依赖第一阶段的列）==========
        features = features.with_columns(
            # --- 速度差 ---
            (pl.col("agent__body_center__speed_200ms") - pl.col("target__body_center__speed_200ms"))
            .alias("speed_diff_200ms"),
            (pl.col("agent__body_center__speed_500ms") - pl.col("target__body_center__speed_500ms"))
            .alias("speed_diff_500ms"),

            # --- 速度比 ---
            (pl.col("agent__body_center__speed_200ms") / (pl.col("target__body_center__speed_200ms") + 0.1))
            .alias("speed_ratio_200ms"),

            # --- Contact score ---
            (pl.col("at__body_center__body_center__dist") < 5).cast(pl.Float32)
            .alias("in_contact"),

            # --- Face-to-face score ---
            ((pl.col("mutual_facing_angle") < 0.5) &
             (pl.col("at__nose__nose__dist") < 5)).cast(pl.Float32)
            .alias("face_to_face"),
        )

        # ========== 第三阶段：复合交互特征（依赖第二阶段的列）==========
        features = features.with_columns(
            # Chase score
            ((pl.col("speed_diff_200ms") > 0).cast(pl.Float32) * 0.33 +
             (pl.col("agent_facing_score") > 0.5).cast(pl.Float32) * 0.33 +
             (pl.col("approach_speed_500ms") < 0).cast(pl.Float32) * 0.33)
            .alias("chase_score"),

            # Flee score
            ((pl.col("speed_diff_200ms") < 0).cast(pl.Float32) * 0.33 +
             (pl.col("target_facing_score") < 0).cast(pl.Float32) * 0.33 +
             (pl.col("approach_speed_500ms") > 0).cast(pl.Float32) * 0.33)
            .alias("flee_score"),

            # Sniffing score
            ((pl.col("agent_nose__min_dist_to_target_head") < 3).cast(pl.Float32) +
             (pl.col("agent_nose__min_dist_to_target_rear") < 3).cast(pl.Float32)) *
            (1 / (pl.col("agent__body_center__speed_200ms") + 1))
            .alias("sniff_score"),
        )

        # ========== 第四阶段：时间窗口统计特征 ==========
        long_window = get_w(1000)

        cols_to_smooth = [
            "at__body_center__body_center__dist",
            "approach_speed_500ms",
            "speed_diff_200ms",
            "chase_score",
        ]

        features = features.with_columns([
            pl.col(c).rolling_mean(long_window, center=True).alias(f"{c}_avg_1s")
            for c in cols_to_smooth
        ] + [
            pl.col(c).rolling_std(long_window, center=True).alias(f"{c}_std_1s")
            for c in cols_to_smooth
        ] + [
            pl.col("at__body_center__body_center__dist").rolling_min(long_window, center=True).alias("dist_min_1s"),
            pl.col("at__body_center__body_center__dist").rolling_max(long_window, center=True).alias("dist_max_1s"),
        ])

        # ========== 第五阶段：Lag/Lead特征 ==========
        lag_lead_exprs = []
        for lag in [5, 10, 15, 30]:
            lag_lead_exprs.extend([
                pl.col("at__body_center__body_center__dist").shift(lag).alias(f"dist_lag_{lag}"),
                pl.col("at__body_center__body_center__dist").shift(-lag).alias(f"dist_lead_{lag}"),
                pl.col("approach_speed_500ms").shift(lag).alias(f"approach_lag_{lag}"),
                pl.col("approach_speed_500ms").shift(-lag).alias(f"approach_lead_{lag}"),
                pl.col("agent__body_center__speed_500ms").shift(lag).alias(f"agent_speed_lag_{lag}"),
                pl.col("agent__body_center__speed_500ms").shift(-lag).alias(f"agent_speed_lead_{lag}"),
            ])

        features = features.with_columns(lag_lead_exprs)

        # ========== 第六阶段：趋势特征 ==========
        features = features.with_columns(
            (pl.col("dist_lag_30") - pl.col("at__body_center__body_center__dist")).alias("dist_change_1s"),
            (pl.col("approach_speed_500ms_avg_1s") < -0.5).cast(pl.Float32).alias("approaching_trend"),
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")

def preprocess_tracking(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
    """
    针对宽表格式 (agent_x_nose, target_x_ear_left...) 进行鲁棒清洗
    """
    fps = metadata["frames_per_second"]
    pix_per_cm = metadata["pix_per_cm_approx"]

    # --- 参数设置 ---
    # 1. 形态学阈值: 任何部位距离 Body Center 超过 15cm 视为异常 (老鼠没那么大)
    MAX_DIST_FROM_CENTER_CM = 15.0
    max_dist_px = MAX_DIST_FROM_CENTER_CM * pix_per_cm

    # 2. 速度阈值: 单帧移动超过 100cm/s (瞬移) 视为异常
    MAX_SPEED_CM_S = 100.0
    max_step_px = (MAX_SPEED_CM_S * pix_per_cm) / fps

    # 识别所有坐标列
    x_cols = [c for c in df.columns if "_x_" in c]
    y_cols = [c for c in df.columns if "_y_" in c]

    # 暂存处理表达式
    df_cleaned = df.clone()

    # ===========================
    # Stage 1: 基于身体中心的形态学过滤 (最稳健)
    # ===========================
    # 这种方法比 IQR 更适合 Pose Estimation，因为 IQR 会受到老鼠在场地位置的影响
    prefixes = set([c.split("_x_")[0] for c in x_cols]) # {'agent'} or {'agent', 'target'}

    for prefix in prefixes:
        center_x_col = f"{prefix}_x_body_center"
        center_y_col = f"{prefix}_y_body_center"

        # 如果连身体中心都没有，无法进行此步检查
        if center_x_col not in df.columns:
            continue

        curr_x_cols = [c for c in x_cols if c.startswith(prefix) and "body_center" not in c]

        for xc in curr_x_cols:
            yc = xc.replace("_x_", "_y_")

            # 计算该部位到身体中心的距离
            dist_to_center = (
                (pl.col(xc) - pl.col(center_x_col)).pow(2) +
                (pl.col(yc) - pl.col(center_y_col)).pow(2)
            ).sqrt()

            # 超过阈值置为 None
            df_cleaned = df_cleaned.with_columns([
                pl.when(dist_to_center > max_dist_px).then(None).otherwise(pl.col(xc)).alias(xc),
                pl.when(dist_to_center > max_dist_px).then(None).otherwise(pl.col(yc)).alias(yc)
            ])

    # ===========================
    # Stage 2: 基于速度的瞬移过滤
    # ===========================
    # 对 Body Center 和其他关键点做速度检查
    # 注意：这里我们对所有点都做检查，防止插值前的极端跳变
    for xc in x_cols:
        yc = xc.replace("_x_", "_y_")

        # 计算当前帧与上一帧的距离 (欧氏距离)
        # diff() 计算的是 (t) - (t-1)
        step_dist = ((pl.col(xc).diff()).pow(2) + (pl.col(yc).diff()).pow(2)).sqrt()

        # 如果单步跳变太大，认为当前帧是异常值
        # 注意：这里简单处理，将当前帧置空。更复杂的可以用 shift(-1) 比较
        is_jump = step_dist > max_step_px

        df_cleaned = df_cleaned.with_columns([
            pl.when(is_jump).then(None).otherwise(pl.col(xc)).alias(xc),
            pl.when(is_jump).then(None).otherwise(pl.col(yc)).alias(yc)
        ])

    # ===========================
    # Stage 3: 插值与平滑 (修复 Null)
    # ===========================
    # 1. 线性插值填补刚才产生的 None 以及原始的 NaN
    df_cleaned = df_cleaned.with_columns([
        pl.col(c).interpolate().fill_null(strategy="mean") for c in x_cols + y_cols
    ])

    # 2. Savitzky-Golay 平滑的替代品：Rolling Mean
    # 用于消除微小的抖动噪声
    df_cleaned = df_cleaned.with_columns([
        pl.col(c).rolling_mean(window_size=7, center=True).fill_null(strategy="mean")
        for c in x_cols + y_cols
    ])

    return df_cleaned

def process_video(row):
    """Process a single video to extract self and pair features."""
    lab_id = row["lab_id"]
    video_id = row["video_id"]

    tracking_path = TRAIN_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    tracking = pl.read_parquet(tracking_path)

    self_features = make_self_features(metadata=row, tracking=tracking)
    pair_features = make_pair_features(metadata=row, tracking=tracking)

    self_features.write_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
    pair_features.write_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")

    return video_id


# make data
(WORKING_DIR / "self_features").mkdir(exist_ok=True, parents=True)
(WORKING_DIR / "pair_features").mkdir(exist_ok=True, parents=True)

rows = list(train_dataframe.filter(pl.col("behaviors_labeled").is_not_null()).rows(named=True))
results = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(process_video)(row) for row in rows)

print(f"Processed {len(results)} videos successfully")

del rows, results
gc.collect()

def tune_threshold(oof_action, y_action):
    thresholds = np.arange(0, 1.005, 0.005)
    scores = [f1_score(y_action, (oof_action >= th), zero_division=0) for th in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx]

def train_validate_self(lab_id: str, behavior: str, indices: pl.DataFrame, features: pl.DataFrame, labels: pl.Series):
    result_dir = WORKING_DIR / "results" / lab_id / behavior
    result_dir.mkdir(exist_ok=True, parents=True)

    if labels.sum() == 0:
        with open(result_dir / "f1.txt", "w") as f:
            f.write("0.0\n")
        oof_prediction_dataframe = indices.with_columns(
            pl.Series("fold", [-1] * len(labels), dtype=pl.Int8),
            pl.Series("prediction", [0.0] * len(labels), dtype=pl.Float32),
            pl.Series("predicted_label", [0] * len(labels), dtype=pl.Int8),
        )
        oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")
        return 0.0

    folds = np.ones(len(labels), dtype=np.int8) * -1
    oof_predictions = np.zeros(len(labels), dtype=np.float32)
    oof_prediction_labels = np.zeros(len(labels), dtype=np.int8)

    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            X=features,
            y=labels,
            groups=indices.get_column("video_id"),
        )
    ):
        result_dir_fold = result_dir / f"fold_{fold}"
        result_dir_fold.mkdir(exist_ok=True, parents=True)

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_valid = features[valid_idx]
        y_valid = labels[valid_idx]

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": "gpu",
            "tree_method": "hist",
            "learning_rate": 0.002,
            "max_depth": 6,
            "min_child_weight": 4,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            #"scale_pos_weight": scale_pos_weight,
            "max_bin": 255,
            "seed": 42,
        }
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=features.columns, max_bin=255)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=features.columns)

        evals_result = {}
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=30,
            metric_name="logloss",
            data_name="valid",
            maximize=False,
            save_best=True,
        )
        model = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=3000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[early_stopping_callback],
            evals_result=evals_result,
            verbose_eval=0,
        )

        fold_predictions = model.predict(dvalid)

        threshold = tune_threshold(fold_predictions, y_valid)
        folds[valid_idx] = fold
        oof_predictions[valid_idx] = fold_predictions
        oof_prediction_labels[valid_idx] = (fold_predictions >= threshold).astype(np.int8)

        # save results
        model.save_model(result_dir_fold / "model.json")
        with open(result_dir_fold / "threshold.txt", "w") as f:
            f.write(f"{threshold}\n")

        gc.collect()

    oof_prediction_dataframe = indices.with_columns(
        pl.Series("fold", folds, dtype=pl.Int8),
        pl.Series("prediction", oof_predictions, dtype=pl.Float32),
        pl.Series("predicted_label", oof_prediction_labels, dtype=pl.Int8),
    )
    f1 = f1_score(labels, oof_prediction_labels, zero_division=0)
    with open(result_dir / "f1.txt", "w") as f:
        f.write(f"{f1}\n")

    oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")

    return f1

def train_validate_pair(lab_id: str, behavior: str, indices: pl.DataFrame, features: pl.DataFrame, labels: pl.Series):
    result_dir = WORKING_DIR / "results" / lab_id / behavior
    result_dir.mkdir(exist_ok=True, parents=True)

    if labels.sum() == 0:
        with open(result_dir / "f1.txt", "w") as f:
            f.write("0.0\n")
        oof_prediction_dataframe = indices.with_columns(
            pl.Series("fold", [-1] * len(labels), dtype=pl.Int8),
            pl.Series("prediction", [0.0] * len(labels), dtype=pl.Float32),
            pl.Series("predicted_label", [0] * len(labels), dtype=pl.Int8),
        )
        oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")
        return 0.0

    folds = np.ones(len(labels), dtype=np.int8) * -1
    oof_predictions = np.zeros(len(labels), dtype=np.float32)
    oof_prediction_labels = np.zeros(len(labels), dtype=np.int8)

    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            X=features,
            y=labels,
            groups=indices.get_column("video_id"),
        )
    ):
        result_dir_fold = result_dir / f"fold_{fold}"
        result_dir_fold.mkdir(exist_ok=True, parents=True)

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_valid = features[valid_idx]
        y_valid = labels[valid_idx]

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": "gpu",
            "tree_method": "hist",
            "learning_rate": 0.003,
            "max_depth": 6,
            "min_child_weight": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight**0.8,
            "max_bin": 64,
            "seed": 42,
        }
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=features.columns, max_bin=64)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=features.columns)

        evals_result = {}
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=50,
            metric_name="logloss",
            data_name="valid",
            maximize=False,
            save_best=True,
        )
        model = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=4000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[early_stopping_callback],
            evals_result=evals_result,
            verbose_eval=0,
        )

        fold_predictions = model.predict(dvalid)

        threshold = tune_threshold(fold_predictions, y_valid)
        folds[valid_idx] = fold
        oof_predictions[valid_idx] = fold_predictions
        oof_prediction_labels[valid_idx] = (fold_predictions >= threshold).astype(np.int8)

        # save results
        model.save_model(result_dir_fold / "model.json")
        with open(result_dir_fold / "threshold.txt", "w") as f:
            f.write(f"{threshold}\n")

        gc.collect()

    oof_prediction_dataframe = indices.with_columns(
        pl.Series("fold", folds, dtype=pl.Int8),
        pl.Series("prediction", oof_predictions, dtype=pl.Float32),
        pl.Series("predicted_label", oof_prediction_labels, dtype=pl.Int8),
    )
    f1 = f1_score(labels, oof_prediction_labels, zero_division=0)
    with open(result_dir / "f1.txt", "w") as f:
        f.write(f"{f1}\n")

    oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")

    return f1

groups = train_self_behavior_dataframe.group_by("lab_id", "behavior", maintain_order=True)
total_groups = len(list(groups))
start_time = time.perf_counter()

for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
    if idx == 0:
        tqdm.write(
            f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'ELAPSED TIME':^15}|",
            end="\n",
        )

    tqdm.write(f"|{lab_id:^25}|{behavior:^15}|", end="")
    index_list = []
    feature_list = []
    label_list = []

    for row in group.rows(named=True):
        video_id = row["video_id"]
        agent = row["agent"]

        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))

        data = pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet").filter(
            (pl.col("agent_mouse_id") == agent_mouse_id)
        )
        index = data.select(INDEX_COLS).collect(engine="streaming")
        feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")

        # read annotation
        annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        if annotation_path.exists():
            annotation = (
                pl.scan_parquet(annotation_path)
                .filter((pl.col("action") == behavior) & (pl.col("agent_id") == agent_mouse_id))
                .collect()
            )
        else:
            annotation = pl.DataFrame(
                schema={
                    "agent_id": pl.Int8,
                    "target_id": pl.Int8,
                    "action": str,
                    "start_frame": pl.Int16,
                    "stop_frame": pl.Int16,
                }
            )

        label_frames = set()
        for annotation_row in annotation.rows(named=True):
            label_frames.update(range(annotation_row["start_frame"], annotation_row["stop_frame"]))
        label = index.select(pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label"))

        if label.get_column("label").sum() == 0:
            continue

        index_list.append(index)
        feature_list.append(feature)
        label_list.append(label.get_column("label"))

    if not index_list:
        elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{0:>10,}|{0:>10,}|{0:>10,}|{'-':>10}|{str(elapsed_time):>15}|", end="\n")
        continue

    indices = pl.concat(index_list, how="vertical")
    features = pl.concat(feature_list, how="vertical")
    labels = pl.concat(label_list, how="vertical")

    del index_list, feature_list, label_list
    gc.collect()

    tqdm.write(f"{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")

    f1 = train_validate_self(lab_id, behavior, indices, features, labels)
    tqdm.write(f"{f1:>10.2f}|", end="")

    elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
    tqdm.write(f"{str(elapsed_time):>15}|", end="\n")

    gc.collect()

NEGATIVE_RATIO = 1000000

groups = train_pair_behavior_dataframe.group_by("lab_id", "behavior", maintain_order=True)
total_groups = len(list(groups))
start_time = time.perf_counter()

# 初始化随机生成器用于下采样
rng = np.random.default_rng(42)

for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
    if idx == 0:
        tqdm.write(
            f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'ELAPSED TIME':^15}|",
            end="\n",
        )

    # 调试用：为了快速测试可以取消下面的注释
    # if idx >= 6: break

    tqdm.write(f"|{lab_id:^25}|{behavior:^15}|", end="")
    index_list = []
    feature_list = []
    label_list = []

    for row in group.rows(named=True):
        video_id = row["video_id"]
        agent = row["agent"]
        target = row["target"]

        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
        target_mouse_id = int(re.search(r"mouse(\d+)", target).group(1))

        # 1. Lazy Load & Filter
        # 仅读取需要的行，减少 I/O 和初始内存
        data_scan = pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet").filter(
            (pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id)
        )

        # 2. Read Annotation
        annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        if annotation_path.exists():
            # 仅读取与当前 Agent-Target 对相关的标注
            annotation = (
                pl.scan_parquet(annotation_path)
                .filter(
                    (pl.col("action") == behavior)
                    & (pl.col("agent_id") == agent_mouse_id)
                    & (pl.col("target_id") == target_mouse_id)
                )
                .collect()
            )
        else:
            # 空标注处理
            annotation = pl.DataFrame(schema={"start_frame": pl.Int16, "stop_frame": pl.Int16})

        # 3. 构建 Label 集合
        label_frames = set()
        for annotation_row in annotation.rows(named=True):
            label_frames.update(range(annotation_row["start_frame"], annotation_row["stop_frame"]))

        # 如果没有正样本，根据策略可以选择跳过该视频，或者保留少量负样本
        # 为了训练稳定性，这里跳过完全没有正样本的 Agent-Target 对
        if not label_frames:
            continue

        # 4. 收集数据 (先全部 collect，然后在内存中做 Mask)
        # 注意：如果单个视频非常大，这一步也可以放在 scan 阶段做，但逻辑较复杂
        raw_data = data_scan.collect()
        if raw_data.is_empty():
            continue

        # 提取 Label 列
        current_frames = raw_data["video_frame"]
        is_positive = current_frames.is_in(label_frames)

        # --- 核心修改：负样本下采样 ---
        # 获取正样本数量
        n_pos = is_positive.sum()

        # 创建 Mask
        # 保留所有正样本
        mask_array = is_positive.to_numpy()

        # 随机采样负样本
        # 如果正样本很少，至少保留一些负样本以避免数据量太小
        n_neg_to_keep = max(n_pos * NEGATIVE_RATIO, 100)

        # 找到负样本的索引
        neg_indices = np.where(~mask_array)[0]

        if len(neg_indices) > n_neg_to_keep:
            # 随机选择要保留的负样本索引
            keep_neg_indices = rng.choice(neg_indices, size=n_neg_to_keep, replace=False)
            # 更新 Mask：正样本 OR 被选中的负样本
            final_mask = mask_array.copy()
            # 先把所有负样本置 False (已经是 False 了，但为了逻辑清晰)
            final_mask[neg_indices] = False
            # 把选中的负样本置 True
            final_mask[keep_neg_indices] = True
        else:
            # 负样本不够多，全部保留
            final_mask = np.ones(len(raw_data), dtype=bool)

        # 应用 Filter
        filtered_data = raw_data.filter(pl.lit(final_mask))

        if filtered_data.is_empty():
            continue

        # 5. 分离 Index, Feature, Label
        # 强制转换 Feature 为 Float32 节省内存
        index = filtered_data.select(INDEX_COLS)
        feature = filtered_data.select(pl.exclude(INDEX_COLS)).cast(pl.Float32)
        # 1. 去掉 pl.lit()，直接传 numpy 数组
        # 2. 将 .alias() 改为 .rename()
        label = is_positive.filter(final_mask).cast(pl.Int8).rename("label")

        index_list.append(index)
        feature_list.append(feature)
        label_list.append(label)

    if not index_list:
        elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{0:>10,}|{0:>10,}|{0:>10,}|{'-':>10}|{str(elapsed_time):>15}|", end="\n")
        continue

    # 6. 合并数据
    indices = pl.concat(index_list, how="vertical")
    features = pl.concat(feature_list, how="vertical")
    labels = pl.concat(label_list, how="vertical")

    # 立即释放列表内存
    del index_list, feature_list, label_list, raw_data
    gc.collect()

    tqdm.write(f"{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")

    # 7. 训练
    f1 = train_validate_pair(lab_id, behavior, indices, features, labels)
    tqdm.write(f"{f1:>10.2f}|", end="")

    elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
    tqdm.write(f"{str(elapsed_time):>15}|", end="\n")

    # 每一轮 Lab/Behavior 结束后强制 GC
    del indices, features, labels
    gc.collect()

def robustify(submission: pl.DataFrame, dataset: pl.DataFrame, train_test: str = "train"):
    traintest_directory = INPUT_DIR / f"{train_test}_tracking"

    old_submission = submission.clone()
    submission = submission.filter(pl.col("start_frame") < pl.col("stop_frame"))
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")

    old_submission = submission.clone()
    group_list = []
    for _, group in submission.group_by("video_id", "agent_id", "target_id"):
        group = group.sort("start_frame")
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, row in enumerate(group.rows(named=True)):
            if row["start_frame"] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row["stop_frame"]
        group_list.append(group.filter(pl.Series("mask", mask)))

    submission = pl.concat(group_list)

    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    s_list = []
    for row in dataset.rows(named=True):
        lab_id = row["lab_id"]
        video_id = row["video_id"]
        if row["behaviors_labeled"] is None:
            continue

        if video_id in submission.get_column("video_id").to_list():
            continue

        if isinstance(row["behaviors_labeled"], str):
            continue

        print(f"Video {video_id} has no predictions.")

        path = traintest_directory / f"/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row["behaviors_labeled"])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(",") for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(["agent", "target"]):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, action_row in enumerate(actions.itertuples(index=False)):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row["action"], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat(
            [
                submission,
                pd.DataFrame(s_list, columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]),
            ]
        )
        print("ERROR: Filled empty videos")

    return submission

group_oof_predictions = []
groups = train_behavior_dataframe.group_by("lab_id", "video_id", "agent", "target", maintain_order=True)

for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups))):
    agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
    target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))

    prediction_dataframe_list = []

    for row in group.rows(named=True):
        behavior = row["behavior"]

        oof_path = WORKING_DIR / "results" / lab_id / behavior / "oof_predictions.parquet"
        if not oof_path.exists():
            continue

        prediction = (
            pl.scan_parquet(oof_path)
            .filter(
                (pl.col("video_id") == video_id)
                & (pl.col("agent_mouse_id") == agent_mouse_id)
                & (pl.col("target_mouse_id") == target_mouse_id)
            )
            .select(*INDEX_COLS, (pl.col("prediction") * pl.col("predicted_label")).alias(behavior))
            .collect()
        )

        if len(prediction) == 0:
            continue

        prediction_dataframe_list.append(prediction)

    if not prediction_dataframe_list:
        continue

    prediction_dataframe = pl.concat(prediction_dataframe_list, how="align")

    cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
    prediction_labels_dataframe = prediction_dataframe.with_columns(
        pl.struct(pl.exclude(INDEX_COLS))
        .map_elements(
            lambda row: "none" if sum(row.values()) == 0 else (cols[np.argmax(list(row.values()))]),
            return_dtype=pl.String,
        )
        .alias("prediction")
    ).select(INDEX_COLS + ["prediction"])

    group_oof_prediction = (
        prediction_labels_dataframe.filter((pl.col("prediction") != pl.col("prediction").shift(1)))
        .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))
        .filter(pl.col("prediction") != "none")
        .select(
            pl.col("video_id"),
            ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),
            pl.when(pl.col("target_mouse_id") == -1)
            .then(pl.lit("self"))
            .otherwise("mouse" + pl.col("target_mouse_id").cast(str))
            .alias("target_id"),
            pl.col("prediction").alias("action"),
            pl.col("video_frame").alias("start_frame"),
            pl.col("stop_frame"),
        )
    )

    group_oof_predictions.append(group_oof_prediction)


oof_predictions = pl.concat(group_oof_predictions, how="vertical")
oof_predictions = robustify(oof_predictions, train_dataframe, train_test="train")
oof_predictions.with_row_index("row_id").write_csv(WORKING_DIR / "oof_predictions.csv")

def compute_validation_metrics(submission, verbose=True):
    """Compute and display validation metrics for single vs pair behaviors."""
    # solution_df
    dataset = pl.read_csv(INPUT_DIR / "train.csv").to_pandas()

    solution = []
    for _, row in dataset.iterrows():
        lab_id = row["lab_id"]
        if lab_id.startswith("MABe22"):
            continue

        video_id = row["video_id"]
        path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue

        annot["lab_id"] = lab_id
        annot["video_id"] = video_id
        annot["behaviors_labeled"] = row["behaviors_labeled"]
        annot["target_id"] = np.where(
            annot.target_id != annot.agent_id, annot["target_id"].apply(lambda s: f"mouse{s}"), "self"
        )
        annot["agent_id"] = annot["agent_id"].apply(lambda s: f"mouse{s}")
        solution.append(annot)

    solution = pd.concat(solution)

    try:
        # Separate single and pair behaviors
        submission_single = submission[submission["target_id"] == "self"].copy()
        submission_pair = submission[submission["target_id"] != "self"].copy()

        # Filter solution to match submission videos
        solution_videos = set(submission["video_id"].unique())
        solution = solution[solution["video_id"].isin(solution_videos)]

        if len(solution) == 0:
            return

        # Compute overall F1 score
        overall_f1 = score(solution, submission, "row_id", beta=1.0)
        print(f"\n{'=' * 60}")
        print("PERFORMANCE METRICS")
        print(f"{'=' * 60}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Total predictions: {len(submission)}")
        print(f"  - Single behaviors: {len(submission_single)}")
        print(f"  - Pair behaviors: {len(submission_pair)}")

        # Compute per-action F1 scores using existing scoring function
        solution_pl = pl.DataFrame(solution)
        submission_pl = pl.DataFrame(submission)

        # Add label_key and prediction_key
        solution_pl = solution_pl.with_columns(
            pl.concat_str(
                [
                    pl.col("video_id").cast(pl.Utf8),
                    pl.col("agent_id").cast(pl.Utf8),
                    pl.col("target_id").cast(pl.Utf8),
                    pl.col("action"),
                ],
                separator="_",
            ).alias("label_key"),
        )
        submission_pl = submission_pl.with_columns(
            pl.concat_str(
                [
                    pl.col("video_id").cast(pl.Utf8),
                    pl.col("agent_id").cast(pl.Utf8),
                    pl.col("target_id").cast(pl.Utf8),
                    pl.col("action"),
                ],
                separator="_",
            ).alias("prediction_key"),
        )

        # Group by action and compute metrics
        action_stats = defaultdict(lambda: {"single": {"count": 0, "f1": 0.0}, "pair": {"count": 0, "f1": 0.0}})

        for lab in solution_pl["lab_id"].unique():
            lab_solution = solution_pl.filter(pl.col("lab_id") == lab).clone()
            lab_videos = set(lab_solution["video_id"].unique())
            lab_submission = submission_pl.filter(pl.col("video_id").is_in(lab_videos)).clone()

            # Compute per-action F1 using same logic as single_lab_f1
            label_frames = defaultdict(set)
            prediction_frames = defaultdict(set)

            for row in lab_solution.to_dicts():
                label_frames[row["label_key"]].update(range(row["start_frame"], row["stop_frame"]))

            for row in lab_submission.to_dicts():
                key = row["prediction_key"]
                prediction_frames[key].update(range(row["start_frame"], row["stop_frame"]))

            for key in set(list(label_frames.keys()) + list(prediction_frames.keys())):
                action = key.split("_")[-1]
                mode = "single" if "self" in key else "pair"

                pred_frames = prediction_frames.get(key, set())
                label_frames_set = label_frames.get(key, set())

                tp = len(pred_frames & label_frames_set)
                fn = len(label_frames_set - pred_frames)
                fp = len(pred_frames - label_frames_set)

                if tp + fn + fp > 0:
                    f1 = (1 + 1**2) * tp / ((1 + 1**2) * tp + 1**2 * fn + fp)
                    action_stats[action][mode]["count"] += 1
                    action_stats[action][mode]["f1"] += f1

        # Print per-action summary
        print("\nPer-Action Performance Summary:")
        print(f"{'-' * 60}")
        print(f"{'Action':<20} {'Mode':<10} {'Count':<10} {'Avg F1':<10}")
        print(f"{'-' * 60}")

        for action in sorted(action_stats.keys()):
            for mode in ["single", "pair"]:
                stats = action_stats[action][mode]
                if stats["count"] > 0:
                    avg_f1 = stats["f1"] / stats["count"]
                    print(f"{action:<20} {mode:<10} {stats['count']:<10} {avg_f1:<10.4f}")

        # Summary by mode
        single_actions = [a for a in action_stats.keys() if action_stats[a]["single"]["count"] > 0]
        pair_actions = [a for a in action_stats.keys() if action_stats[a]["pair"]["count"] > 0]

        if single_actions:
            single_avg_f1 = np.mean(
                [
                    action_stats[a]["single"]["f1"] / action_stats[a]["single"]["count"]
                    for a in single_actions
                    if action_stats[a]["single"]["count"] > 0
                ]
            )
            print(f"\nSingle behaviors: {len(single_actions)} actions, Avg F1: {single_avg_f1:.4f}")

        if pair_actions:
            pair_avg_f1 = np.mean(
                [
                    action_stats[a]["pair"]["f1"] / action_stats[a]["pair"]["count"]
                    for a in pair_actions
                    if action_stats[a]["pair"]["count"] > 0
                ]
            )
            print(f"Pair behaviors: {len(pair_actions)} actions, Avg F1: {pair_avg_f1:.4f}")

        print(f"{'=' * 60}\n")

    except Exception as e:
        if verbose:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"\nWarning: Could not compute validation metrics: {error_msg}")
            if verbose:
                print(f"Traceback: {traceback.format_exc()[:300]}")

compute_validation_metrics(submission=pd.read_csv(WORKING_DIR / "oof_predictions.csv"))


