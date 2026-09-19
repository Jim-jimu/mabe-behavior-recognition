# Imports and configs
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from lightgbm import LGBMClassifier
import random
from tqdm import tqdm
from koolbox import Trainer
import numpy as np
import pandas as pd
import itertools
import warnings
import optuna
import joblib
from joblib import Parallel, delayed, parallel_config, cpu_count
import glob
import gc
import json
from collections import defaultdict
import polars as pl
import os
from datetime import datetime
from time import perf_counter
import shutil
from pathlib import Path
import sys
import traceback
import platform
from importlib.metadata import version
# ============================================================================
# 初始化与配置
# ============================================================================
class CFG:
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    # train_path = "./MABe-mouse-behavior-detection/train.csv"
    # test_path = "./MABe-mouse-behavior-detection/test.csv"
    # train_annotation_path = "./MABe-mouse-behavior-detection/train_annotation"
    # train_tracking_path = "./MABe-mouse-behavior-detection/train_tracking"
    # test_tracking_path = "./MABe-mouse-behavior-detection/test_tracking"

    model_path = "/kaggle/input"
    model_name = "fold555"

    SEED = 3407

    mode = "validate"  # 重新训练，不需要原来的模型输入
    output_dir = None  # main() 为本次训练创建独立目录
    feature_jobs = 1   # 分批写盘；可通过 MABE_FEATURE_JOBS 改为 2
    model_jobs = -1    # 模型拟合使用全部可用 CPU；可通过 MABE_MODEL_JOBS 调整
    threshold_trials = 100
    section_start = 1  # 保持原代码的配置编号，供旧推理入口加载
    section_stop = None  # None 训练全部配置；数字为不包含的上界

    n_splits = 5
    cv = StratifiedGroupKFold(n_splits)

    # model = XGBClassifier(
    #     verbosity=0,
    #     random_state=42,
    #     n_estimators=250,
    #     learning_rate=0.08,
    #     max_depth=6,
    #     min_child_weight=5,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     tree_method='gpu_hist',  # 使用GPU加速
    #     device='cuda:0',
    # )


    model = LGBMClassifier(
        verbosity=-1,             # 静默模式
        random_state=42,
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,              # 限制深度
        num_leaves=31,            # 关键参数: LightGBM 是 leaf-wise，配合 max_depth=6，建议 31 (2^5) 到 63 (2^6-1)
        min_child_weight=10,       # 对应 min_sum_hessian_in_leaf，控制叶子节点最小权重和
        subsample=0.8,
        subsample_freq=1,         # 关键参数: LightGBM 需要设置 freq > 0 才能启用 subsample (bagging)
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        # device='gpu',             # 启用 GPU
        # gpu_device_id=1,          # 指定 GPU ID
        n_jobs=-1                 # prepare_training() 按实际 CPU 配额应用线程数
    )



def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
set_global_seeds(CFG.SEED)

# Global constants
# 需要丢弃的身体部位（冗余或噪声较大的关键点）
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']


META_NUM_COLS = [
    "frames_per_second",
    "pix_per_cm_approx",
    "video_width",
    "video_height",
    "arena_width_cm",
    "arena_height_cm",
    "n_mice",
]

META_CAT_COLS = [
    "lab_id",
    "arena_shape",
    "arena_type",
    "tracking_method",
]


META_CAT_ENCODING = "onehot"  # {"onehot", "label"}


META_CAT_CATEGORIES = {}   # col -> List[str]
META_CAT_KNOWN_SET = {}    # col -> Set[str] (fast membership)
META_CAT_SPECIAL_UNK = "__UNK__"
META_CAT_SPECIAL_NA = "__NA__"



META_CAT_ENCODERS = {}
META_CAT_UNK = {}


def meta_to_features(meta: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=meta.index)

    # 数值型
    for col in META_NUM_COLS:
        if col in meta.columns:
            feats[col] = pd.to_numeric(meta[col], errors="coerce").astype(np.float32)

    # === New: derived numeric meta features (robust + domain-general) ===
    eps = np.float32(1e-6)

    # fps transforms
    if "frames_per_second" in feats.columns:
        fps = feats["frames_per_second"].astype(np.float32)
        fps_pos = fps.clip(lower=0)
        feats["fps_log1p"] = np.log1p(fps_pos).astype(np.float32)
        feats["sec_per_frame"] = (1.0 / (fps_pos + eps)).astype(np.float32)

    # pixel <-> cm scale transforms
    if "pix_per_cm_approx" in feats.columns:
        ppc = feats["pix_per_cm_approx"].astype(np.float32)
        ppc_pos = ppc.clip(lower=0)
        feats["cm_per_pix_approx"] = (1.0 / (ppc_pos + eps)).astype(np.float32)
        feats["log_pix_per_cm"] = np.log1p(ppc_pos).astype(np.float32)

    # video geometry (pixels)
    if "video_width" in feats.columns and "video_height" in feats.columns:
        vw = feats["video_width"].astype(np.float32)
        vh = feats["video_height"].astype(np.float32)
        feats["video_aspect"] = (vw / (vh + eps)).astype(np.float32)
        area_v = (vw * vh).astype(np.float32)
        feats["video_area_px"] = area_v
        feats["video_area_log1p"] = np.log1p(area_v.clip(lower=0)).astype(np.float32)
        feats["video_diag_px"] = np.sqrt((vw * vw + vh * vh).clip(lower=0)).astype(np.float32)

    # arena geometry (cm)
    if "arena_width_cm" in feats.columns and "arena_height_cm" in feats.columns:
        aw = feats["arena_width_cm"].astype(np.float32)
        ah = feats["arena_height_cm"].astype(np.float32)
        aw_pos = aw.clip(lower=0)
        ah_pos = ah.clip(lower=0)

        feats["arena_aspect"] = (aw_pos / (ah_pos + eps)).astype(np.float32)
        area_a = (aw_pos * ah_pos).astype(np.float32)
        feats["arena_area_cm2"] = area_a
        feats["arena_area_log1p"] = np.log1p(area_a).astype(np.float32)
        feats["arena_perimeter_cm"] = (2.0 * (aw_pos + ah_pos)).astype(np.float32)
        feats["arena_diag_cm"] = np.sqrt((aw_pos * aw_pos + ah_pos * ah_pos).clip(lower=0)).astype(np.float32)

    # density / per-mouse normalization
    if "n_mice" in feats.columns:
        nm = feats["n_mice"].astype(np.float32).clip(lower=0)
        feats["n_mice_log1p"] = np.log1p(nm).astype(np.float32)
        feats["n_mice_sq"] = (nm * nm).astype(np.float32)

        if "arena_area_cm2" in feats.columns:
            feats["mice_density"] = (nm / (feats["arena_area_cm2"] + eps)).astype(np.float32)
            feats["arena_area_per_mouse"] = (feats["arena_area_cm2"] / (nm + eps)).astype(np.float32)
    # === End new code ===

    # 类别型
    if META_CAT_ENCODING == "onehot":
        for col in META_CAT_COLS:
            if col not in meta.columns:
                continue
            if col not in META_CAT_CATEGORIES:
                # vocab 未构建（理论上 main() 会用 train 构建），这里保守跳过
                continue

            known = META_CAT_CATEGORIES[col]
            known_set = META_CAT_KNOWN_SET[col]

            # 统一转字符串；缺失 -> __NA__；未知 -> __UNK__
            s = meta[col].astype("string")
            s = s.fillna(META_CAT_SPECIAL_NA)
            s = s.where(s.isin(known_set), META_CAT_SPECIAL_UNK)

            # 固定 categories，确保 get_dummies 输出列集合稳定（即便某些类别本批次没出现）
            cat = pd.Categorical(
                s,
                categories=known + [META_CAT_SPECIAL_UNK, META_CAT_SPECIAL_NA],
                ordered=False,
            )

            dummies = pd.get_dummies(
                cat,
                prefix=col,
                prefix_sep="=",
                dtype=np.uint8,  # 省内存；下游 concat 后再统一 float32 也行
            )

            feats = pd.concat([feats, dummies], axis=1)

    else:

        for col in META_CAT_COLS:
            if col in meta.columns and col in META_CAT_ENCODERS:
                enc = META_CAT_ENCODERS[col]
                unk = META_CAT_UNK[col]
                feats[col + "_idx"] = meta[col].map(lambda v: enc.get(v, unk)).astype(np.int32)

    return feats





# ============================================================================
# Missing-value handling (short-gap fill) + missingness-as-signal features
# ============================================================================

MISSING_MAX_GAP_SEC = 0.25  # 只填补 <= 0.25s 的连续缺失
MISSING_EPS = 1e-6


def _fill_small_gaps(df: pd.DataFrame, fps: float, max_gap_sec: float = MISSING_MAX_GAP_SEC) -> pd.DataFrame:
    """只填补短缺失（<= max_gap_sec 秒），长缺失保留 NaN。"""
    try:
        fps = float(fps) if fps is not None and not pd.isna(fps) else 30.0
    except Exception:
        fps = 30.0
    limit = max(1, int(round(max_gap_sec * fps)))

    out = df.copy()
    # 内部空洞插值（不外推）
    out = out.interpolate(method="linear", axis=0, limit=limit, limit_area="inside")
    # 极短边界缺失（仍受 limit 限制）
    out = out.ffill(limit=limit).bfill(limit=limit)
    return out


def _part_ok(mouse_df: pd.DataFrame) -> pd.DataFrame:
    """
    mouse_df columns: MultiIndex (bodypart, x/y)
    return: frames x bodypart (bool), True 表示该部位 x/y 都非空
    """
    # pandas: groupby(axis=1) deprecated -> use transpose
    return mouse_df.notna().T.groupby(level=0).all().T


def _nose_proxy_ok(part_ok: pd.DataFrame) -> pd.Series:
    idx = part_ok.index
    if "nose" in part_ok.columns:
        return part_ok["nose"]
    if "head" in part_ok.columns:
        return part_ok["head"]
    if "ear_left" in part_ok.columns and "ear_right" in part_ok.columns:
        return part_ok["ear_left"] & part_ok["ear_right"]
    return pd.Series(False, index=idx)


def _center_proxy_ok(part_ok: pd.DataFrame) -> pd.Series:
    idx = part_ok.index
    if "body_center" in part_ok.columns:
        return part_ok["body_center"]
    if "neck" in part_ok.columns:
        return part_ok["neck"]
    if "nose" in part_ok.columns and "tail_base" in part_ok.columns:
        return part_ok["nose"] & part_ok["tail_base"]
    if "head" in part_ok.columns and "tail_base" in part_ok.columns:
        return part_ok["head"] & part_ok["tail_base"]
    if "ear_left" in part_ok.columns and "ear_right" in part_ok.columns:
        return part_ok["ear_left"] & part_ok["ear_right"]
    return pd.Series(False, index=idx)


def _streak_len(mask: pd.Series) -> pd.Series:
    """
    mask=True 表示处于“缺失状态”，返回当前连续缺失长度（否则为0）
    """
    m = mask.fillna(False).astype(bool)
    grp = (~m).cumsum()
    out = m.groupby(grp).cumcount() + 1
    out = out.where(m, 0)
    return out


def add_missingness_features_single(X: pd.DataFrame, single_mouse: pd.DataFrame, fps: float, section: int) -> pd.DataFrame:
    part_ok = _part_ok(single_mouse)  # frames x bodypart
    ok_cnt = part_ok.sum(axis=1).astype(np.float32)
    total = float(part_ok.shape[1] if part_ok.shape[1] > 0 else 1)
    miss_ratio = (1.0 - ok_cnt / (total + MISSING_EPS)).astype(np.float32)

    nose_ok = _nose_proxy_ok(part_ok).astype(np.float32)
    center_ok = _center_proxy_ok(part_ok).astype(np.float32)

    X["kp_ok_cnt"] = ok_cnt
    X["kp_missing_ratio"] = miss_ratio
    X["nose_proxy_ok"] = nose_ok
    X["center_proxy_ok"] = center_ok

    # 连续缺失段长度（遮挡/打斗/追逐常见）
    X["nose_missing_streak"] = _streak_len(nose_ok < 0.5).astype(np.float32)
    X["center_missing_streak"] = _streak_len(center_ok < 0.5).astype(np.float32)

    # 多尺度 rolling（轻量、对 domain shift 很稳）
    window = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        roll = dict(min_periods=max(1, ws // 5), center=True)
        X[f"kp_missing_m{w}"] = miss_ratio.rolling(ws, **roll).mean().astype(np.float32)
        X[f"nose_ok_m{w}"] = nose_ok.rolling(ws, **roll).mean().astype(np.float32)
        X[f"center_ok_m{w}"] = center_ok.rolling(ws, **roll).mean().astype(np.float32)

    return X


def add_missingness_features_pair(X: pd.DataFrame, mouse_pair: pd.DataFrame, fps: float) -> pd.DataFrame:
    part_ok_A = _part_ok(mouse_pair["A"])
    part_ok_B = _part_ok(mouse_pair["B"])

    totalA = float(part_ok_A.shape[1] if part_ok_A.shape[1] > 0 else 1)
    totalB = float(part_ok_B.shape[1] if part_ok_B.shape[1] > 0 else 1)

    A_ok_cnt = part_ok_A.sum(axis=1).astype(np.float32)
    B_ok_cnt = part_ok_B.sum(axis=1).astype(np.float32)
    A_miss = (1.0 - A_ok_cnt / (totalA + MISSING_EPS)).astype(np.float32)
    B_miss = (1.0 - B_ok_cnt / (totalB + MISSING_EPS)).astype(np.float32)

    X["A_kp_missing_ratio"] = A_miss
    X["B_kp_missing_ratio"] = B_miss
    X["AB_kp_missing_ratio_mean"] = (0.5 * (A_miss + B_miss)).astype(np.float32)
    X["AB_kp_missing_ratio_diff"] = (A_miss - B_miss).astype(np.float32)

    A_nose_ok = _nose_proxy_ok(part_ok_A).astype(np.float32)
    B_nose_ok = _nose_proxy_ok(part_ok_B).astype(np.float32)
    A_center_ok = _center_proxy_ok(part_ok_A).astype(np.float32)
    B_center_ok = _center_proxy_ok(part_ok_B).astype(np.float32)

    X["A_nose_proxy_ok"] = A_nose_ok
    X["B_nose_proxy_ok"] = B_nose_ok
    X["A_center_proxy_ok"] = A_center_ok
    X["B_center_proxy_ok"] = B_center_ok
    X["AB_both_center_ok"] = (A_center_ok * B_center_ok).astype(np.float32)

    X["A_nose_missing_streak"] = _streak_len(A_nose_ok < 0.5).astype(np.float32)
    X["B_nose_missing_streak"] = _streak_len(B_nose_ok < 0.5).astype(np.float32)

    for w in [15, 30, 60, 120]:
        ws = _scale(w, fps)
        roll = dict(min_periods=max(1, ws // 5), center=True)
        X[f"AB_miss_m{w}"] = X["AB_kp_missing_ratio_mean"].rolling(ws, **roll).mean().astype(np.float32)

    return X



# ============================================================================
# Creating solution data
# ============================================================================
def create_solution_df(dataset):
    """
    创建验证用的标准答案DataFrame

    输入:
        dataset: pd.DataFrame - 训练数据集的元信息

    输出:
        pd.DataFrame - 包含所有标注数据的DataFrame，列包括：
            - lab_id: 实验室ID
            - video_id: 视频ID
            - agent_id: 执行动作的老鼠ID（格式：'mouse1', 'mouse2'等）
            - target_id: 目标老鼠ID或'self'
            - action: 行为类型
            - start_frame, stop_frame: 行为的起止帧
            - behaviors_labeled: 该视频标注的行为列表

    作用:
        整合分散的标注文件，生成一个标准化的 “标准答案” 数据集，
        用于验证模式下计算模型性能指标
    """
    solution = []
    #从dataset中逐行提取lab_id和video_id
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):  #跳过lab_id以MABe22开头的视频
            continue

        video_id = row['video_id']
        path = f"{CFG.train_annotation_path}/{lab_id}/{video_id}.parquet"  #找到该轮lab_id和video_id对应数据
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue

        # 为标注数据添加元信息，包括lab_id、video_id、behaviors_labeled（从元信息中获取）
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        #target_id和agent_id转换为mouse1、mouse2等格式
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot) #将每个标注数据添加到solution列表中

    solution = pd.concat(solution) #将solution列表中的所有标注数据合并成一个DataFrame

    return solution #返回solution

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    生成器函数：逐个生成单只老鼠或老鼠对的数据

    输入:
        dataset: pd.DataFrame - 数据集元信息
        traintest: str - 'train'或'test'，指定数据类型
        traintest_directory: str, optional - tracking数据目录路径
        generate_single: bool - 是否生成单只老鼠数据
        generate_pair: bool - 是否生成老鼠对数据

    输出 (yield):
        对于训练数据:
            ('single', data, meta, label) 或 ('pair', data, meta, label)
            - data: pd.DataFrame - 老鼠的坐标数据
            - meta: pd.DataFrame - 元数据（video_id, agent_id, target_id, video_frame）
            - label: pd.DataFrame - 标签数据

        对于测试数据:
            ('single', data, meta, actions) 或 ('pair', data, meta, actions)
            - data: pd.DataFrame - 老鼠的坐标数据
            - meta: pd.DataFrame - 元数据
            - actions: np.array - 需要预测的行为列表

    作用:
        从tracking文件中读取老鼠的关键点坐标数据，
        并根据需要生成单只老鼠或老鼠对的数据及对应标签
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        # traintest_directory = f"dataset/MABe-mouse-behavior-detection/{traintest}_tracking"
    # 逐行读取视频信息，提取lab_id和video_id
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22') or type(row.behaviors_labeled) != str:  #跳过lab_id以MABe22开头的视频或behaviors_labeled不是字符串的视频
            continue

        scalar_meta = {}
        for col in META_NUM_COLS + META_CAT_COLS:
            if col in dataset.columns:
                scalar_meta[col] = row[col]


        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"  #找到该轮lab_id和video_id对应数据
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:  #如果该视频有5个以上的身体部位，则丢弃bodypart中包含drop_body_parts的行
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])



        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        # === Best: fill only short gaps (keeps long occlusions as NaN) ===
        fps_row = row.frames_per_second if "frames_per_second" in dataset.columns else 30.0
        pvid = _fill_small_gaps(pvid, fps=fps_row, max_gap_sec=MISSING_MAX_GAP_SEC)
        # === End best code ===


        # 将视频元信息中的行为描述字符串解析并转换为结构化的 DataFrame
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        # 训练数据时，读取标注数据
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))  #找到该轮lab_id和video_id对应标注数据
            except FileNotFoundError:
                continue  #如果标注数据不存在，跳过该视频

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")  # 从视频的所有行为中筛选出单鼠行为
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):  # 遍历所有单鼠行为
                try:
                    mouse_id = int(mouse_id_str[-1])  # 提取老鼠ID
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)  #提取该老鼠的所有自身行为类型
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        **scalar_meta,   # <-- 新增: 注入视频级 meta
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label  #类型+特征+元数据+标签
                    else:
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions  #类型+特征+元数据+要预测的行为类型
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2): # int8
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        **scalar_meta,   # <-- 新增: 注入视频级 meta
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ============================================================================
# Transforming coordinates
# ============================================================================
def safe_rolling(series, window, func, min_periods=None):
    """
    安全的滚动窗口计算函数，避免窗口过小导致的计算失败

    输入:
        series: pd.Series - 需要进行滚动计算的时间序列数据
        window: int - 滚动窗口大小（帧数）
        func: callable - 应用于窗口的函数
        min_periods: int, optional - 最小有效数据点数，默认为窗口大小的1/4

    输出:
        pd.Series - 滚动计算后的结果序列

    作用:
        对时间序列数据进行滚动窗口计算，自动处理边界情况
    """
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """
    根据实际帧率缩放窗口大小

    输入:
        n_frames_at_30fps: int - 在30fps下的帧数
        fps: float - 实际视频帧率
        ref: float - 参考帧率，默认30.0

    输出:
        int - 缩放后的帧数（至少为1）

    作用:
        将基于30fps设计的窗口大小转换为适应实际帧率的窗口大小
        例如：如果实际fps=60，则窗口大小会翻倍
    """
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """
    根据实际帧率缩放窗口大小（保留正负符号）

    输入:
        n_frames_at_30fps: int - 在30fps下的帧数（可以为负数）
        fps: float - 实际视频帧率
        ref: float - 参考帧率，默认30.0

    输出:
        int - 缩放后的帧数（保留符号，至少为±1）

    作用:
        类似_scale，但保留符号，用于时间偏移量的缩放
        例如：-10帧在60fps下会变成-20帧
    """
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """
    从元数据中提取视频帧率

    输入:
        meta_df: pd.DataFrame - 包含视频元数据的DataFrame
        fallback_lookup: dict - video_id到fps的映射字典（备用）
        default_fps: float - 默认帧率，默认30.0

    输出:
        float - 视频的帧率

    作用:
        按优先级获取视频帧率：
        1. 从meta_df的frames_per_second列
        2. 从fallback_lookup字典
        3. 使用默认值30.0
    """
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

# ============================================================================
# extract features
# ============================================================================
# 单鼠特征
# 曲率和转向相关的特征
def add_curvature_features(X, center_x, center_y, fps, section):
    new_features = {}

    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    window = [25, 50, 75] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 5)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    window = [30] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 5)).sum()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 多尺度速度特征
def add_multiscale_features(X, center_x, center_y, fps, section):
    new_features = {}
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            new_features[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            new_features[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()
            new_features[f'sp_q25_{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).quantile(0.25)
            # 为了计算 IQR，我们需要先计算 q75
            q75 = speed.rolling(ws, min_periods=max(1, ws // 4)).quantile(0.75)
            new_features[f'sp_q75_{scale}'] = q75
            if f'sp_q25_{scale}' in new_features:
                new_features[f'sp_iqr{scale}'] = q75 - new_features[f'sp_q25_{scale}']

    # 计算 sp_ratio 需要先有 sp_m
    if len(scales) >= 2:
        k1 = f'sp_m{scales[0]}'
        k2 = f'sp_m{scales[-1]}'
        # 如果在本轮计算中生成了这两个特征，或者它们已经在 X 中（虽然这个函数通常是第一次生成它们）
        # 我们优先从 new_features 获取
        val1 = new_features.get(k1, X.get(k1))
        val2 = new_features.get(k2, X.get(k2))

        if val1 is not None and val2 is not None:
            new_features['sp_ratio'] = val1 / (val2 + 1e-6)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 运动状态特征
def add_state_features(X, center_x, center_y, fps, section):
    new_features = {}
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # FIX: speed_ma 已经是 cm/s（上面乘过 fps），分箱阈值也必须是 cm/s 的常数，不能再乘 fps
        bins_cms = [-np.inf, 0.5, 2.0, 5.0, np.inf]  # cm/s
        speed_states = pd.cut(speed_ma, bins=bins_cms, labels=[0, 1, 2, 3]).astype(float)

        window = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
        for w in window:
            ws = _scale(w, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    new_features[f's{state}_{w}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                new_features[f'trans_{w}'] = state_changes.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 长时程特征
def add_longrange_features(X, center_x, center_y, fps):
    new_features = {}
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            new_features[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            new_features[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [30, 60, 120]:
        s = _scale(span, fps)
        new_features[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        new_features[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            new_features[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_posture_stability_features(X, single_mouse, available_body_parts, fps):
    new_features = {}
    if 'ear_left' in available_body_parts and 'ear_right' in available_body_parts:
        ear_left = single_mouse['ear_left']
        ear_right = single_mouse['ear_right']

        # 耳朵对称性
        ear_mid_x = (ear_left['x'] + ear_right['x']) / 2
        ear_mid_y = (ear_left['y'] + ear_right['y']) / 2

        # 耳朵中点稳定性
        ear_mid_std_x = ear_mid_x.rolling(_scale(30, fps),
                                          min_periods=_scale(5, fps)).std()
        ear_mid_std_y = ear_mid_y.rolling(_scale(30, fps),
                                          min_periods=_scale(5, fps)).std()
        new_features['ear_mid_std'] = np.sqrt(ear_mid_std_x ** 2 + ear_mid_std_y ** 2)

    # 身体部位相对位置稳定性
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if nose is not None and 'tail_base' in available_body_parts:
        nose_tail_vec_x = nose['x'] - single_mouse['tail_base']['x']
        nose_tail_vec_y = nose['y'] - single_mouse['tail_base']['y']
        nose_tail_length = np.sqrt(nose_tail_vec_x ** 2 + nose_tail_vec_y ** 2)

        new_features['nose_tail_length_std'] = nose_tail_length.rolling(
            _scale(30, fps), min_periods=_scale(5, fps)).std()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_head_elevation_features(X, single_mouse, available_body_parts, fps, section):
    if section <= 3 or section == 8:
        return X

    new_features = {}
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if center is not None and nose is not None and 'tail_base' in available_body_parts:
        head_elevation = nose['y'] - center['y']  # 头部相对身体中心的高度
        body_angle_vertical = (nose['y'] - single_mouse['tail_base']['y'])  # 身体垂直度
        for w in [10, 20, 30]:
            ws = _scale(w, fps)
            new_features[f'head_elev_mean_{w}'] = head_elevation.rolling(ws).mean()
            new_features[f'head_elev_std_{w}'] = head_elevation.rolling(ws).std()
            new_features[f'body_vert_{w}'] = body_angle_vertical.rolling(ws).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_body_width_features(X, single_mouse, available_body_parts, fps, section):
    if section == 3:
        return X

    new_features = {}
    if 'lateral_left' in available_body_parts and 'lateral_right' in available_body_parts:
        body_width_x = np.abs(single_mouse['lateral_left']['x'] - single_mouse['lateral_right']['x'])
        body_width_y = np.abs(single_mouse['lateral_left']['y'] - single_mouse['lateral_right']['y'])
        body_width = np.sqrt(body_width_x**2 + body_width_y**2)
        for w in [15, 30, 60, 120]:
            ws = _scale(w, fps)
            new_features[f'body_width_mean_{w}'] = body_width.rolling(ws).mean()
            new_features[f'body_width_std_{w}'] = body_width.rolling(ws).std()
            new_features[f'body_width_change_{w}'] = body_width.diff().rolling(ws).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_pose_shape_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {}
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    # 躯干-头部-尾部三角形特征
    if center is not None and nose is not None and 'tail_base' in available_body_parts:
        nose_x = nose['x']
        nose_y = nose['y']
        center_x = center['x']
        center_y = center['y']
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        # 三角形面积（使用叉积公式）
        triangle_area = pd.Series(0.5 * np.abs(
            nose_x * (center_y - tail_y) +
            center_x * (tail_y - nose_y) +
            tail_x * (nose_y - center_y)
        ), index=single_mouse.index)
        new_features['tri_area'] = triangle_area

        # 三角形边长
        nose_center_dist = pd.Series(np.sqrt((nose_x - center_x)**2 + (nose_y - center_y)**2), index=single_mouse.index)
        center_tail_dist = pd.Series(np.sqrt((center_x - tail_x)**2 + (center_y - tail_y)**2), index=single_mouse.index)
        nose_tail_dist = pd.Series(np.sqrt((nose_x - tail_x)**2 + (nose_y - tail_y)**2), index=single_mouse.index)

        # 边长比例（前半身 vs 后半身）
        new_features['tri_front_back_ratio'] = nose_center_dist / (center_tail_dist + 1e-6)

        # 三角形的"紧凑度"：面积 / 周长^2（类似圆形度）
        perimeter = nose_center_dist + center_tail_dist + nose_tail_dist
        new_features['tri_compactness'] = triangle_area / (perimeter**2 + 1e-6)

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'tri_area_m{w}'] = triangle_area.rolling(ws, **roll_params).mean()
            new_features[f'tri_area_s{w}'] = triangle_area.rolling(ws, **roll_params).std()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_head_body_decoupled_features(X, single_mouse, available_body_parts, fps, section):
    """
    添加头部与身体解耦的运动特征
    (优化：批量添加列以避免 DataFrame 碎片化)
    """
    new_features = {} # 使用字典收集新特征

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if center is not None and nose is not None:
        # 1. 头部相对身体的速度
        nose_x = nose['x']
        nose_y = nose['y']
        center_x = center['x']
        center_y = center['y']

        # 头部相对于体心的位置
        rel_nose_x = nose_x - center_x
        rel_nose_y = nose_y - center_y

        # 头部相对速度（相对位置的变化率）
        rel_nose_vx = rel_nose_x.diff() * fps
        rel_nose_vy = rel_nose_y.diff() * fps
        head_rel_speed = pd.Series(np.sqrt(rel_nose_vx**2 + rel_nose_vy**2), index=single_mouse.index)

        new_features['head_rel_speed'] = head_rel_speed

        # 体心速度
        center_vx = center_x.diff() * fps
        center_vy = center_y.diff() * fps
        body_speed = pd.Series(np.sqrt(center_vx**2 + center_vy**2), index=single_mouse.index)

        # 头部速度 / 体心速度比值
        new_features['head_body_speed_ratio'] = head_rel_speed / (body_speed + 1e-6)

        window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'head_rel_sp_m{w}'] = head_rel_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_rel_sp_s{w}'] = head_rel_speed.rolling(ws, **roll_params).std()

        # 2. 头部转动速度（方向变化率）
        # 头部方向向量（体心指向鼻子）
        head_dir_x = nose_x - center_x
        head_dir_y = nose_y - center_y

        # 头部朝向角度
        head_angle = pd.Series(np.arctan2(head_dir_y, head_dir_x), index=single_mouse.index)

        # 角度变化（注意处理-π到π的跳变）
        angle_diff = head_angle.diff()
        angle_diff = pd.Series(np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff), index=single_mouse.index)
        angle_diff = pd.Series(np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff), index=single_mouse.index)

        # 头部转动速度（弧度/秒）
        head_turn_speed = pd.Series(np.abs(angle_diff) * fps, index=single_mouse.index)
        new_features['head_turn_speed'] = head_turn_speed

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'head_turn_m{w}'] = head_turn_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_turn_s{w}'] = head_turn_speed.rolling(ws, **roll_params).std()
            # 头部方向变化的总和（累积转动）
            new_features[f'head_turn_sum{w}'] = head_turn_speed.rolling(ws, **roll_params).sum()

    # 3. 耳朵与鼻子/体心的几何关系
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']) and center is not None and nose is not None:
        ear_left_x = single_mouse['ear_left']['x']
        ear_left_y = single_mouse['ear_left']['y']
        ear_right_x = single_mouse['ear_right']['x']
        ear_right_y = single_mouse['ear_right']['y']
        center_x = center['x']
        center_y = center['y']
        nose_x = nose['x']
        nose_y = nose['y']

        # 耳朵中点
        ear_mid_x = (ear_left_x + ear_right_x) / 2
        ear_mid_y = (ear_left_y + ear_right_y) / 2

        # 耳朵中点到体心的距离
        ear_center_dist = pd.Series(np.sqrt((ear_mid_x - center_x)**2 + (ear_mid_y - center_y)**2), index=single_mouse.index)
        new_features['ear_center_dist'] = ear_center_dist

        # 耳朵中点到鼻子的距离
        ear_nose_dist = pd.Series(np.sqrt((ear_mid_x - nose_x)**2 + (ear_mid_y - nose_y)**2), index=single_mouse.index)
        new_features['ear_nose_dist'] = ear_nose_dist

        # 耳朵-鼻子-体心形成的角度
        vec_nose_ear_x = ear_mid_x - nose_x
        vec_nose_ear_y = ear_mid_y - nose_y
        vec_nose_center_x = center_x - nose_x
        vec_nose_center_y = center_y - nose_y

        dot_product = vec_nose_ear_x * vec_nose_center_x + vec_nose_ear_y * vec_nose_center_y
        norm_product = pd.Series(np.sqrt(vec_nose_ear_x**2 + vec_nose_ear_y**2) *
                       np.sqrt(vec_nose_center_x**2 + vec_nose_center_y**2) + 1e-6, index=single_mouse.index)
        new_features['ear_nose_center_angle'] = dot_product / norm_product

        window = [30, 60] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'ear_center_m{w}'] = ear_center_dist.rolling(ws, **roll_params).mean()
            new_features[f'ear_center_s{w}'] = ear_center_dist.rolling(ws, **roll_params).std()

    # 最后一次性合并所有新特征
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X


def add_body_axis_motion_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {}  # 收集新特征

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    # 如果缺少必要的锚点，跳过
    if nose is None or center is None or 'tail_base' not in available_body_parts:
        return X

    # 1. 计算身体朝向向量（尾部 -> 头部）
    ori_x = nose['x'] - single_mouse['tail_base']['x']
    ori_y = nose['y'] - single_mouse['tail_base']['y']
    ori_norm = np.sqrt(ori_x ** 2 + ori_y ** 2) + 1e-6

    # 归一化朝向向量
    ori_x_unit = ori_x / ori_norm
    ori_y_unit = ori_y / ori_norm

    # 2. 计算运动中心的速度向量
    vx = center['x'].diff() * fps  # cm/s
    vy = center['y'].diff() * fps  # cm/s

    # 3. 将速度分解到体轴坐标系
    # v_forward: 沿身体朝向的速度分量（正=前进，负=后退）
    v_forward = vx * ori_x_unit + vy * ori_y_unit
    # v_lateral: 垂直于身体朝向的速度分量（正=向右，负=向左）
    # 法向量：(-ori_y_unit, ori_x_unit) 指向右侧
    v_lateral = vx * (-ori_y_unit) + vy * ori_x_unit

    # 将结果转换为 Series 并设置索引
    v_forward = pd.Series(v_forward.values, index=single_mouse.index)
    v_lateral = pd.Series(v_lateral.values, index=single_mouse.index)

    # 4. 基础特征
    new_features['v_forward'] = v_forward
    new_features['v_lateral'] = v_lateral
    new_features['v_forward_abs'] = np.abs(v_forward)
    new_features['v_lateral_abs'] = np.abs(v_lateral)

    # 侧向 vs 前向速度比
    lat_over_fwd = np.abs(v_lateral) / (np.abs(v_forward) + 1e-6)
    new_features['lat_over_fwd'] = lat_over_fwd

    window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
    # 5. 多尺度统计
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 4), center=True)

        # 前向速度统计
        new_features[f'v_fwd_m{w}'] = v_forward.rolling(ws, **roll_params).mean()
        new_features[f'v_fwd_s{w}'] = v_forward.rolling(ws, **roll_params).std()
        new_features[f'v_fwd_abs_m{w}'] = np.abs(v_forward).rolling(ws, **roll_params).mean()

        # 侧向速度统计
        new_features[f'v_lat_m{w}'] = v_lateral.rolling(ws, **roll_params).mean()
        new_features[f'v_lat_s{w}'] = v_lateral.rolling(ws, **roll_params).std()
        new_features[f'v_lat_abs_m{w}'] = np.abs(v_lateral).rolling(ws, **roll_params).mean()

        # 侧向/前向比值统计
        new_features[f'lat_fwd_ratio_m{w}'] = lat_over_fwd.rolling(ws, **roll_params).mean()

    # 6. 行为模式指标
    # 前进时间占比（v_forward > 阈值）
    fwd_threshold = 1.0  # cm/s
    is_forward = (v_forward > fwd_threshold).astype(float)
    is_backward = (v_forward < -fwd_threshold).astype(float)
    is_lateral = (np.abs(v_lateral) > np.abs(v_forward)).astype(float)

    window = [30, 60] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # 前进/后退/侧移时间占比
        new_features[f'forward_ratio_{w}'] = is_forward.rolling(ws, **roll_params).mean()
        new_features[f'backward_ratio_{w}'] = is_backward.rolling(ws, **roll_params).mean()
        new_features[f'lateral_dom_ratio_{w}'] = is_lateral.rolling(ws, **roll_params).mean()

    # 7. 滑移角（身体朝向与运动方向的偏差）
    # 运动方向角
    theta_move = np.arctan2(vy, vx)
    # 身体朝向角
    theta_body = np.arctan2(ori_y, ori_x)

    # 滑移角（-π 到 π）
    slip_angle = theta_move - theta_body
    # 归一化到 -π 到 π
    slip_angle = np.arctan2(np.sin(slip_angle), np.cos(slip_angle))
    slip_angle = pd.Series(slip_angle.values, index=single_mouse.index)

    new_features['slip_angle'] = slip_angle
    new_features['slip_angle_abs'] = np.abs(slip_angle)

    window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
    # 滑移角统计
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 4), center=True)

        new_features[f'slip_m{w}'] = slip_angle.rolling(ws, **roll_params).mean()
        new_features[f'slip_s{w}'] = slip_angle.rolling(ws, **roll_params).std()
        new_features[f'slip_abs_m{w}'] = np.abs(slip_angle).rolling(ws, **roll_params).mean()

    # 滑移角对齐程度（身体朝向与运动方向一致的时间占比）
    align_threshold = np.pi / 6  # 30度
    is_aligned = (np.abs(slip_angle) < align_threshold).astype(float)

    window = [30, 60] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'slip_align_ratio_{w}'] = is_aligned.rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_high_freq_micromotion_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {} # 收集新特征

    # 获取前端锚点（nose代理）和运动中心（body_center代理）
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    center = _get_substitute_body_center(single_mouse, available_body_parts)

    # 1. 高频头部微运动特征
    if nose is not None:
        nose_x = nose['x']
        nose_y = nose['y']

        # 头部瞬时速度（帧间位移）
        head_vx = nose_x.diff() * fps
        head_vy = nose_y.diff() * fps
        head_speed = pd.Series(np.sqrt(head_vx ** 2 + head_vy ** 2), index=single_mouse.index)

        # --- 1.1 短窗口高频头部运动统计 ---
        # 使用非常短的窗口捕捉高频微运动
        for w in [3, 5, 7]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)

            # 头部速度的短窗口统计
            new_features[f'head_hf_m{w}'] = head_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_hf_s{w}'] = head_speed.rolling(ws, **roll_params).std()
            new_features[f'head_hf_max{w}'] = head_speed.rolling(ws, **roll_params).max()

        # --- 1.2 头部加速度（二阶微分，捕捉颤动/抖动） ---
        head_ax = head_vx.diff() * fps
        head_ay = head_vy.diff() * fps
        head_accel = pd.Series(np.sqrt(head_ax ** 2 + head_ay ** 2), index=single_mouse.index)
        new_features['head_accel'] = head_accel

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'head_accel_m{w}'] = head_accel.rolling(ws, **roll_params).mean()
            new_features[f'head_accel_s{w}'] = head_accel.rolling(ws, **roll_params).std()

        # --- 1.3 头部抖动指数（jitter index）---
        # 连续帧之间速度方向的变化（检测快速来回抖动）
        head_dir = pd.Series(np.arctan2(head_vy, head_vx + 1e-8), index=single_mouse.index)
        head_dir_change = head_dir.diff()
        # 处理角度跳变
        head_dir_change = pd.Series(
            np.where(head_dir_change > np.pi, head_dir_change - 2 * np.pi, head_dir_change),
            index=single_mouse.index
        )
        head_dir_change = pd.Series(
            np.where(head_dir_change < -np.pi, head_dir_change + 2 * np.pi, head_dir_change),
            index=single_mouse.index
        )
        head_jitter = pd.Series(np.abs(head_dir_change), index=single_mouse.index)
        new_features['head_jitter'] = head_jitter

        for w in [5, 10, 15]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'head_jitter_m{w}'] = head_jitter.rolling(ws, **roll_params).mean()
            # 方向变化总和（累积抖动量）
            new_features[f'head_jitter_sum{w}'] = head_jitter.rolling(ws, **roll_params).sum()

    # 2. 尾巴高频微运动特征
    if 'tail_base' in available_body_parts:
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        # 尾巴瞬时速度
        tail_vx = tail_x.diff() * fps
        tail_vy = tail_y.diff() * fps
        tail_speed = pd.Series(np.sqrt(tail_vx ** 2 + tail_vy ** 2), index=single_mouse.index)

        # --- 2.1 短窗口尾巴高频运动统计 ---
        for w in [3, 5, 7]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_hf_m{w}'] = tail_speed.rolling(ws, **roll_params).mean()
            new_features[f'tail_hf_s{w}'] = tail_speed.rolling(ws, **roll_params).std()

        # --- 2.2 尾巴加速度 ---
        tail_ax = tail_vx.diff() * fps
        tail_ay = tail_vy.diff() * fps
        tail_accel = pd.Series(np.sqrt(tail_ax ** 2 + tail_ay ** 2), index=single_mouse.index)
        new_features['tail_accel'] = tail_accel

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_accel_m{w}'] = tail_accel.rolling(ws, **roll_params).mean()

        # --- 2.3 尾巴抖动指数 ---
        tail_dir = pd.Series(np.arctan2(tail_vy, tail_vx + 1e-8), index=single_mouse.index)
        tail_dir_change = tail_dir.diff()
        tail_dir_change = pd.Series(
            np.where(tail_dir_change > np.pi, tail_dir_change - 2 * np.pi, tail_dir_change),
            index=single_mouse.index
        )
        tail_dir_change = pd.Series(
            np.where(tail_dir_change < -np.pi, tail_dir_change + 2 * np.pi, tail_dir_change),
            index=single_mouse.index
        )
        tail_jitter = pd.Series(np.abs(tail_dir_change), index=single_mouse.index)
        new_features['tail_jitter'] = tail_jitter

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_jitter_m{w}'] = tail_jitter.rolling(ws, **roll_params).mean()

        # --- 2.4 尾巴相对于体心的局部运动 ---
        if center is not None:
            center_x = center['x']
            center_y = center['y']
            body_vx = center_x.diff() * fps
            body_vy = center_y.diff() * fps

            rel_tail_vx = tail_vx - body_vx
            rel_tail_vy = tail_vy - body_vy
            rel_tail_speed = pd.Series(np.sqrt(rel_tail_vx ** 2 + rel_tail_vy ** 2), index=single_mouse.index)
            new_features['tail_rel_speed'] = rel_tail_speed

            for w in [10, 20]:
                ws = _scale(w, fps)
                roll_params = dict(min_periods=max(1, ws // 3), center=True)
                new_features[f'tail_rel_m{w}'] = rel_tail_speed.rolling(ws, **roll_params).mean()

            # 静止时的尾巴活动（如果已计算is_body_still）
            if 'is_body_still' in X.columns:
                is_body_still = X['is_body_still']
                still_tail_activity = is_body_still * rel_tail_speed
                new_features['still_tail_activity'] = still_tail_activity

                for w in [10, 20]:
                    ws = _scale(w, fps)
                    roll_params = dict(min_periods=max(1, ws // 3), center=True)
                    new_features[f'still_tail_act_m{w}'] = still_tail_activity.rolling(ws, **roll_params).mean()

    # 4. 头尾协调/独立运动特征
    if nose is not None and 'tail_base' in available_body_parts:
        nose_x = nose['x']
        nose_y = nose['y']
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        head_vx = nose_x.diff() * fps
        head_vy = nose_y.diff() * fps
        tail_vx = tail_x.diff() * fps
        tail_vy = tail_y.diff() * fps

        head_speed = pd.Series(np.sqrt(head_vx ** 2 + head_vy ** 2), index=single_mouse.index)
        tail_speed = pd.Series(np.sqrt(tail_vx ** 2 + tail_vy ** 2), index=single_mouse.index)

        # --- 4.1 头尾速度比 ---
        head_tail_speed_ratio = head_speed / (tail_speed + 1e-6)
        new_features['head_tail_speed_ratio'] = head_tail_speed_ratio

        for w in [10, 20]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_ratio_m{w}'] = head_tail_speed_ratio.rolling(ws, **roll_params).mean()

        # --- 4.2 头尾运动方向一致性 ---
        # 点积归一化：1表示同向运动，-1表示反向运动，0表示垂直
        head_speed_safe = head_speed + 1e-6
        tail_speed_safe = tail_speed + 1e-6
        head_tail_dir_dot = (head_vx * tail_vx + head_vy * tail_vy) / (head_speed_safe * tail_speed_safe)
        new_features['head_tail_dir_consistency'] = head_tail_dir_dot

        for w in [10, 20, 30]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_dir_m{w}'] = head_tail_dir_dot.rolling(ws, **roll_params).mean()

        # --- 4.3 头尾独立运动指数 ---
        # 低一致性 + 高速度 = 独立运动（可能是不同行为阶段）
        head_tail_independence = (1 - head_tail_dir_dot.abs()) * (head_speed + tail_speed)
        new_features['head_tail_independence'] = head_tail_independence

        for w in [15, 30]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_indep_m{w}'] = head_tail_independence.rolling(ws, **roll_params).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_arena_spatial_features(X, single_mouse, available_body_parts, fps, section, video_id):
    global arena_data
    # 获取场地信息
    try:
        arena_info = arena_data.loc[video_id]
        arena_width = arena_info['arena_width_cm']
        arena_height = arena_info['arena_height_cm']
        arena_shape = arena_info['arena_shape']
    except (KeyError, TypeError):
        # 如果找不到场地信息，跳过
        return X

    # 检查场地尺寸是否有效
    if pd.isna(arena_width) or pd.isna(arena_height) or arena_width <= 0 or arena_height <= 0:
        return X

    # 判断场地形状：圆形 vs 矩形（非圆形都按矩形处理）
    is_circular = False
    if pd.notna(arena_shape):
        shape_lower = str(arena_shape).lower()
        if 'circle' in shape_lower or 'circular' in shape_lower or 'round' in shape_lower:
            is_circular = True

    # 获取运动中心代理点
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    if center is None:
        return X
    center_x = center['x']
    center_y = center['y']

    # ============================================
    # 使用数据范围估算 Arena 边界位置
    # ============================================
    # 使用整个视频序列的坐标范围估算 arena 边界
    x_min = center_x.min()
    x_max = center_x.max()
    y_min = center_y.min()
    y_max = center_y.max()

    # Arena 中心（基于实际数据范围）
    arena_center_x = (x_min + x_max) / 2
    arena_center_y = (y_min + y_max) / 2

    # 定义边界区域宽度（距离墙 15% 的区域视为边界区域）
    border_ratio = 0.15
    border_width = min(arena_width, arena_height) * border_ratio

    if is_circular:
        # 圆形场地的特征计算
        radius = min(arena_width, arena_height) / 2

        # 到场地中心的距离
        dist_to_center = pd.Series(
            np.sqrt((center_x - arena_center_x) ** 2 + (center_y - arena_center_y) ** 2),
            index=single_mouse.index
        )
        X['dist_to_arena_center'] = dist_to_center

        # 归一化到中心的距离（0=中心，1=边界）
        X['dist_to_center_norm'] = dist_to_center / (radius + 1e-6)

        # 到边界（圆周）的距离
        dist_to_nearest_wall = (radius - dist_to_center).clip(lower=0)
        X['dist_to_nearest_wall'] = dist_to_nearest_wall

        # 归一化到墙距离（0=贴墙，1=中心）
        X['dist_to_wall_norm'] = dist_to_nearest_wall / (radius + 1e-6)

        # 空间区域二值指示特征
        wall_threshold = radius * border_ratio  # 靠墙阈值
        central_radius = radius * 0.3  # 中央 30% 区域

        is_near_wall = (dist_to_nearest_wall < wall_threshold).astype(float)
        is_in_center = (dist_to_center < central_radius).astype(float)

        X['is_near_wall'] = is_near_wall
        X['is_in_center'] = is_in_center
        # 圆形场地没有角落
        X['is_in_corner'] = pd.Series(0.0, index=single_mouse.index)

    else:
        # 矩形场地的特征计算
        # 到四面墙的距离（基于数据范围估算边界）
        dist_to_left = center_x - x_min
        dist_to_right = x_max - center_x
        dist_to_bottom = center_y - y_min
        dist_to_top = y_max - center_y

        X['dist_to_left_wall'] = dist_to_left
        X['dist_to_right_wall'] = dist_to_right
        X['dist_to_bottom_wall'] = dist_to_bottom
        X['dist_to_top_wall'] = dist_to_top

        # 到最近墙的距离
        dist_to_nearest_wall = pd.concat(
            [dist_to_left, dist_to_right, dist_to_bottom, dist_to_top], axis=1
        ).min(axis=1)
        X['dist_to_nearest_wall'] = dist_to_nearest_wall

        # 到最近墙的归一化距离（0=贴墙，1=中心）
        max_dist_to_wall = min(arena_width, arena_height) / 2
        X['dist_to_wall_norm'] = dist_to_nearest_wall / (max_dist_to_wall + 1e-6)

        # 到场地中心的距离
        dist_to_center = pd.Series(
            np.sqrt((center_x - arena_center_x) ** 2 + (center_y - arena_center_y) ** 2),
            index=single_mouse.index
        )
        X['dist_to_arena_center'] = dist_to_center

        # 归一化到中心的距离（0=中心，1=角落）
        max_dist_to_center = np.sqrt((arena_width / 2) ** 2 + (arena_height / 2) ** 2)
        X['dist_to_center_norm'] = dist_to_center / (max_dist_to_center + 1e-6)

        # 到四个角落的距离
        dist_to_corner_bl = pd.Series(
            np.sqrt((center_x - x_min) ** 2 + (center_y - y_min) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_br = pd.Series(
            np.sqrt((center_x - x_max) ** 2 + (center_y - y_min) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_tl = pd.Series(
            np.sqrt((center_x - x_min) ** 2 + (center_y - y_max) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_tr = pd.Series(
            np.sqrt((center_x - x_max) ** 2 + (center_y - y_max) ** 2),
            index=single_mouse.index
        )

        # 到最近角落的距离
        dist_to_nearest_corner = pd.concat(
            [dist_to_corner_bl, dist_to_corner_br, dist_to_corner_tl, dist_to_corner_tr],
            axis=1
        ).min(axis=1)
        X['dist_to_nearest_corner'] = dist_to_nearest_corner

        # 空间区域二值指示特征
        is_near_wall = (dist_to_nearest_wall < border_width).astype(float)
        X['is_near_wall'] = is_near_wall

        # 是否在中央区域
        central_radius = min(arena_width, arena_height) * 0.3  # 中央 30% 区域
        is_in_center = (dist_to_center < central_radius).astype(float)
        X['is_in_center'] = is_in_center

        # 是否在角落区域
        corner_radius = min(arena_width, arena_height) * 0.2  # 角落 20% 区域
        is_in_corner = (dist_to_nearest_corner < corner_radius).astype(float)
        X['is_in_corner'] = is_in_corner

    # 6. 头部朝向与墙面的关系
    nose = _get_substitute_nose(single_mouse, available_body_parts)

    if nose is not None:
        nose_x = nose['x']
        nose_y = nose['y']

        # 头部朝向向量（体心指向头部）
        head_dir_x = nose_x - center_x
        head_dir_y = nose_y - center_y
        head_dir_norm = np.sqrt(head_dir_x ** 2 + head_dir_y ** 2) + 1e-6

        # 归一化朝向
        head_dir_x_unit = head_dir_x / head_dir_norm
        head_dir_y_unit = head_dir_y / head_dir_norm

        if is_circular:
            # 圆形场地：法线方向指向圆心
            # 从老鼠位置指向 arena 中心的单位向量
            to_center_x = arena_center_x - center_x
            to_center_y = arena_center_y - center_y
            to_center_norm = np.sqrt(to_center_x ** 2 + to_center_y ** 2) + 1e-6
            to_center_x_unit = to_center_x / to_center_norm
            to_center_y_unit = to_center_y / to_center_norm

            # 面向墙 = 朝向与指向圆心方向相反（负的点积）
            facing_nearest_wall = -(head_dir_x_unit * to_center_x_unit + head_dir_y_unit * to_center_y_unit)
            X['facing_nearest_wall'] = facing_nearest_wall
        else:
            # 矩形场地：到各墙的方向向量
            # 左墙：(-1, 0)，右墙：(1, 0)，下墙：(0, -1)，上墙：(0, 1)
            # 计算头部朝向与最近墙方向的点积
            wall_dirs = pd.DataFrame({
                'left': -head_dir_x_unit,  # 朝左
                'right': head_dir_x_unit,  # 朝右
                'bottom': -head_dir_y_unit,  # 朝下
                'top': head_dir_y_unit  # 朝上
            }, index=single_mouse.index)

            wall_dists = pd.DataFrame({
                'left': X['dist_to_left_wall'],
                'right': X['dist_to_right_wall'],
                'bottom': X['dist_to_bottom_wall'],
                'top': X['dist_to_top_wall']
            }, index=single_mouse.index)

            # 找到最近墙的索引
            # nearest_wall_idx = wall_dists.idxmin(axis=1) # 原代码会触发警告

            # 修复: 仅对非全NA的行计算idxmin，避免FutureWarning
            nearest_wall_idx = pd.Series(index=wall_dists.index, dtype='object')
            valid_rows = wall_dists.notna().any(axis=1)
            if valid_rows.any():
                nearest_wall_idx.loc[valid_rows] = wall_dists.loc[valid_rows].idxmin(axis=1)

            # 计算是否面向最近的墙
            facing_nearest_wall = pd.Series(index=single_mouse.index, dtype=float)
            for wall_name in ['left', 'right', 'bottom', 'top']:
                mask = (nearest_wall_idx == wall_name)
                facing_nearest_wall.loc[mask] = wall_dirs.loc[mask, wall_name]

            X['facing_nearest_wall'] = facing_nearest_wall

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            X[f'facing_wall_m{w}'] = X['facing_nearest_wall'].rolling(ws, **roll_params).mean()

    return X

# 获取替代的body_center和nose
def _get_substitute_body_center(mouse_data, avail_parts):
    # 1. 有 body_center 就用 body_center
    if 'body_center' in avail_parts:
        return mouse_data['body_center']

    # 对于没有body_center的(section7、8、9)
    # 2. 使用neck
    if 'neck' in avail_parts:
        return mouse_data['neck']
    # 3. 使用 nose + tail_base 中点
    if 'nose' in avail_parts and 'tail_base' in avail_parts:
        return (mouse_data['nose'] + mouse_data['tail_base']) / 2

    # 4. 使用 head + tail_base 中点
    if 'head' in avail_parts and 'tail_base' in avail_parts:
        return (mouse_data['head'] + mouse_data['tail_base']) / 2

    # 5. 使用耳朵中点(实际用不上)
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        return (mouse_data['ear_left'] + mouse_data['ear_right']) / 2

    return None

def _get_substitute_nose(mouse_data, avail_parts):
    # 1. 有 nose 就用 nose
    if 'nose' in avail_parts:
        return mouse_data['nose']

    # 对于没有nose的(section7)
    # 2. 使用 head 替代
    if 'head' in avail_parts:
        return mouse_data['head']

    # 3. 使用耳朵中点(实际用不上)
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        return (mouse_data['ear_left'] + mouse_data['ear_right']) / 2

    return None

# 双鼠交互特征
def add_ear_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}
    lag = _scale(10, fps)
    ear_types = ['left', 'right']
    for ear_type in ear_types:
        ear_col = f'ear_{ear_type}'
        if ear_col in avail_A and ear_col in avail_B:
            shA = mouse_pair['A'][ear_col].shift(lag)
            shB = mouse_pair['B'][ear_col].shift(lag)

            new_features[f'sp_A_{ear_type}'] = np.square(mouse_pair['A'][ear_col] - shA).sum(axis=1, skipna=False)
            new_features[f'sp_AB_{ear_type}'] = np.square(mouse_pair['A'][ear_col] - shB).sum(axis=1, skipna=False)
            new_features[f'sp_B_{ear_type}'] = np.square(mouse_pair['B'][ear_col] - shB).sum(axis=1, skipna=False)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_nose_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}
    if 'nose' in avail_A and 'nose' in avail_B:
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            shA_n = mouse_pair['A']['nose'].shift(l)
            shB_n = mouse_pair['B']['nose'].shift(l)
            past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
            new_features[f'appr_{lag}'] = cur - past

        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x']) ** 2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y']) ** 2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            new_features[f'nn_lg{lag}'] = nn.shift(l)
            new_features[f'nn_ch{lag}'] = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            new_features[f'cl_ps{lag}'] = is_cl.rolling(l, min_periods=1).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_body_with_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps):
    # 获取运动中心代理点
    center_A = _get_substitute_body_center(mouse_pair['A'], avail_A)
    center_B = _get_substitute_body_center(mouse_pair['B'], avail_B)

    # 如果任一老鼠没有可用的运动中心，跳过
    if center_A is None or center_B is None:
        return X

    new_features = {}

    # 相对位置
    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = center_A['x'].diff()
    A_vy = center_A['y'].diff()
    B_vx = center_B['x'].diff()
    B_vy = center_B['y'].diff()
    # A、B的速度
    A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
    B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)

    # 1. 相对速度分解：沿连线方向 vs 垂直方向
    # A沿A->B方向的速度分量
    A_vel_along = (A_vx * rel_x + A_vy * rel_y) / (rel_dist + 1e-6)
    # A垂直于A->B方向的速度分量
    A_vel_perp = (A_vx * (-rel_y) + A_vy * rel_x) / (rel_dist + 1e-6)

    # B沿B->A方向的速度分量（注意方向相反）
    B_vel_along = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (rel_dist + 1e-6)
    B_vel_perp = (B_vx * rel_y + B_vy * (-rel_x)) / (rel_dist + 1e-6)

    new_features['A_vel_along'] = A_vel_along
    new_features['A_vel_perp'] = A_vel_perp
    new_features['B_vel_along'] = B_vel_along
    new_features['B_vel_perp'] = B_vel_perp

    # 2. 追逐指标
    # A追B：A沿向B方向移动 + 距离在缩小
    dist_change = rel_dist.diff()
    A_chasing = ((A_vel_along > 0) & (dist_change < 0)).astype(float)
    B_chasing = ((B_vel_along > 0) & (dist_change > 0)).astype(float)

    new_features['A_chasing'] = A_chasing
    new_features['B_chasing'] = B_chasing

    # 追逐强度（速度 * 接近率）
    new_features['A_chase_intensity'] = A_vel_along * (-dist_change) / (A_speed + 1e-6)
    new_features['B_chase_intensity'] = B_vel_along * dist_change / (B_speed + 1e-6)

    # 3. 逃跑指标
    # B逃离A：B沿远离A方向移动 + A在接近
    A_escaping = ((A_vel_along < 0) & (B_vel_along < 0)).astype(float)
    B_escaping = ((B_vel_along < 0) & (A_vel_along > 0)).astype(float)

    new_features['A_escaping'] = A_escaping
    new_features['B_escaping'] = B_escaping

    # 4. 侧向躲避（垂直分量占主导）
    A_sidestepping = (np.abs(A_vel_perp) > np.abs(A_vel_along)).astype(float)
    B_sidestepping = (np.abs(B_vel_perp) > np.abs(B_vel_along)).astype(float)

    new_features['A_sidestepping'] = A_sidestepping
    new_features['B_sidestepping'] = B_sidestepping

    # 5. 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # 追逐时间占比
        new_features[f'A_chasing_p{window}'] = A_chasing.rolling(ws, **roll_params).mean()
        new_features[f'B_chasing_p{window}'] = B_chasing.rolling(ws, **roll_params).mean()

        # 逃跑时间占比
        new_features[f'A_escaping_p{window}'] = A_escaping.rolling(ws, **roll_params).mean()
        new_features[f'B_escaping_p{window}'] = B_escaping.rolling(ws, **roll_params).mean()

        # 侧向移动占比
        new_features[f'A_sidestep_p{window}'] = A_sidestepping.rolling(ws, **roll_params).mean()
        new_features[f'B_sidestep_p{window}'] = B_sidestepping.rolling(ws, **roll_params).mean()

        # 追逐强度统计
        # 注意：这里需要先从new_features获取intensity
        new_features[f'A_chase_int_m{window}'] = new_features['A_chase_intensity'].rolling(ws, **roll_params).mean()
        new_features[f'B_chase_int_m{window}'] = new_features['B_chase_intensity'].rolling(ws, **roll_params).mean()

    # 6. 追逐方向一致性（持续追逐 vs 来回拉锯）
    for window in [30, 60]:
        ws = _scale(window, fps)
        # 方向一致性：同号的比例
        new_features[f'A_chase_consist_{window}'] = (A_vel_along > 0).astype(float).rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

        new_features[f'B_chase_consist_{window}'] = (B_vel_along > 0).astype(float).rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_body_without_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    new_features = {} # 收集新特征

    # 通用变量
    center_A = mouse_pair['A']['body_center']
    center_B = mouse_pair['B']['body_center']

    # 相对位置
    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # 相对速度
    rel_vx = rel_x.diff() * fps
    rel_vy = rel_y.diff() * fps
    rel_speed = np.sqrt(rel_vx ** 2 + rel_vy ** 2)

    # 相对加速度
    rel_ax = rel_vx.diff() * fps
    rel_ay = rel_vy.diff() * fps
    rel_accel = np.sqrt(rel_ax ** 2 + rel_ay ** 2)
    new_features['rel_accel'] = rel_accel

    # 相对速度方向角度
    rel_angle = pd.Series(np.arctan2(rel_vy, rel_vx), index=rel_vx.index)
    # 相对速度方向变化率（转向速度）
    angle_diff = rel_angle.diff()
    # 处理角度跳变（-π到π）
    angle_diff = pd.Series(np.where(angle_diff > np.pi, angle_diff - 2 * np.pi, angle_diff), index=rel_angle.index)
    angle_diff = pd.Series(np.where(angle_diff < -np.pi, angle_diff + 2 * np.pi, angle_diff), index=rel_angle.index)
    rel_turn_rate = np.abs(angle_diff) * fps
    new_features['rel_turn_rate'] = rel_turn_rate

    A_vx = center_A['x'].diff()
    A_vy = center_A['y'].diff()
    B_vx = center_B['x'].diff()
    B_vy = center_B['y'].diff()
    # A、B的速度
    A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
    B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)

    # 运动方向角度
    A_angle = pd.Series(np.arctan2(A_vy, A_vx), index=A_vx.index)
    B_angle = pd.Series(np.arctan2(B_vy, B_vx), index=B_vx.index)

    # 方向差（处理角度跳变）
    angle_diff_2 = A_angle - B_angle
    angle_diff_2 = pd.Series(np.where(angle_diff_2 > np.pi, angle_diff_2 - 2 * np.pi, angle_diff_2), index=angle_diff_2.index)
    angle_diff_2 = pd.Series(np.where(angle_diff_2 < -np.pi, angle_diff_2 + 2 * np.pi, angle_diff_2), index=angle_diff_2.index)

    # 开始提取不同特征
    new_features['v_cls'] = (rel_dist < 5.0).astype(float)
    new_features['cls'] = ((rel_dist >= 5.0) & (rel_dist < 15.0)).astype(float)
    new_features['med'] = ((rel_dist >= 15.0) & (rel_dist < 30.0)).astype(float)
    new_features['far'] = (rel_dist >= 30.0).astype(float)

    cd_full = np.square(center_A - center_B).sum(axis=1, skipna=False)
    coord = A_vx * B_vx + A_vy * B_vy
    for w in [5, 15, 30, 60]:
        ws = _scale(w, fps)
        roll = dict(min_periods=1, center=True)
        new_features[f'd_m{w}'] = cd_full.rolling(ws, **roll).mean()
        new_features[f'd_s{w}'] = cd_full.rolling(ws, **roll).std()
        new_features[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
        new_features[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

        d_var = cd_full.rolling(ws, **roll).var()
        new_features[f'int{w}'] = 1 / (1 + d_var)

        new_features[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
        new_features[f'co_s{w}'] = coord.rolling(ws, **roll).std()
    w = _scale(30, fps)
    new_features['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                   (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    val = (A_vx * B_vx + A_vy * B_vy) / (np.sqrt(A_vx ** 2 + A_vy ** 2) * np.sqrt(B_vx ** 2 + B_vy ** 2) + 1e-6)
    for off in [-30, -20, -10, 0, 10, 20, 30]:
        o = _scale_signed(off, fps)
        new_features[f'va_{off}'] = val.shift(-o)

    '''
    原add_interaction_features函数内的特征
    - A_ld*, B_ld*: 领先/跟随指标（谁在追谁）
    - chase_*: 追逐行为强度
    - sp_cor*: 速度相关性（同步运动程度）
    '''
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx ** 2 + A_vy ** 2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx ** 2 + B_vy ** 2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        new_features[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        new_features[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    new_features[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx ** 2 + A_vy ** 2)
        B_sp = np.sqrt(B_vx ** 2 + B_vy ** 2)
        new_features[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    '''
    原add_advanced_interaction_dynamics函数内的特征
    '''
    def simple_dtw_distance(seq1, seq2):
        """简化的DTW距离计算"""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 0
        # 使用欧氏距离作为基础
        return np.mean(np.abs(seq1[:min(n, m)] - seq2[:min(n, m)]))

    # 计算速度序列的DTW-like距离
    # 注意：这里需要逐行计算，难以向量化，可能仍会比较慢
    # 但我们可以尽量减少 DataFrame 操作
    window_size = _scale(30, fps)
    if len(A_vx) > window_size:
        A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
        B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)

        # 使用 numpy 数组加速
        A_speed_vals = A_speed.values
        B_speed_vals = B_speed.values
        dtw_distances = np.full(len(A_speed), np.nan)

        for i in range(window_size, len(A_speed)):
            window_A = A_speed_vals[i - window_size:i]
            window_B = B_speed_vals[i - window_size:i]
            dtw_distances[i] = simple_dtw_distance(window_A, window_B)

        new_features['speed_dtw_distance'] = pd.Series(dtw_distances, index=X.index)

    # 2. 领导-跟随关系的动态变化
    # 领导力指标（基于速度方向和相对位置的相关性）
    A_leadership = (A_vx * rel_x + A_vy * rel_y) / (rel_dist + 1e-6)
    B_leadership = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (rel_dist + 1e-6)
    leadership_asymmetry = A_leadership - B_leadership
    new_features['leadership_asymmetry'] = leadership_asymmetry
    # 3. 交互势能（基于距离和速度）
    # 类似物理中的势能概念：距离越近，交互势能越高
    interaction_potential = 1 / (rel_dist + 1e-6)
    new_features['interaction_potential'] = interaction_potential
    # 4. 逃避/接近行为的量化
    approach_rate = -rel_dist.diff() * fps  # 正表示接近，负表示远离
    new_features['approach_rate'] = approach_rate
    # 逃避行为的检测（突然的远离）
    sudden_escape = (approach_rate < -20).astype(float)  # 阈值可调整
    new_features['sudden_escape'] = sudden_escape

    '''
    原add_relative_trajectory_features函数内的特征
    '''
    # 相对路径曲率（相对速度方向的变化率 / 相对速度）
    rel_curvature = rel_turn_rate / (rel_speed + 1e-6)
    new_features['rel_curvature'] = rel_curvature

    # 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'rel_turn_rate_m{window}'] = rel_turn_rate.rolling(ws, **roll_params).mean()
        new_features[f'rel_curvature_m{window}'] = rel_curvature.rolling(ws, **roll_params).mean()
        new_features[f'rel_accel_m{window}'] = rel_accel.rolling(ws, **roll_params).mean()
        new_features[f'rel_speed_m{window}'] = rel_speed.rolling(ws, **roll_params).mean()
        new_features[f'rel_speed_s{window}'] = rel_speed.rolling(ws, **roll_params).std()

    '''
    原add_motion_pattern_similarity_features函数内的特征
    '''
    # 方向相似性（余弦相似度）
    direction_similarity = np.cos(angle_diff_2)
    new_features['direction_similarity'] = direction_similarity

    # 速度相似性（归一化的速度差）
    speed_similarity = 1 - np.abs(A_speed - B_speed) / (A_speed + B_speed + 1e-6)
    new_features['speed_similarity'] = speed_similarity

    # 多尺度统计：滚动窗口内的相关性
    for window in [60, 120]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 6), center=True)

        # 方向相似性均值
        new_features[f'direction_sim_m{window}'] = direction_similarity.rolling(ws, **roll_params).mean()

        # 速度相似性均值
        new_features[f'speed_sim_m{window}'] = speed_similarity.rolling(ws, **roll_params).mean()

    '''
    原add_symmetric_asymmetric_features函数内的特征
    '''
    # 1. 速度差和速度比
    speed_diff = A_speed - B_speed
    speed_ratio = A_speed / (B_speed + 1e-6)

    new_features['speed_diff'] = speed_diff
    new_features['speed_ratio'] = speed_ratio

    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'speed_diff_m{window}'] = speed_diff.rolling(ws, **roll_params).mean()
        new_features[f'speed_diff_s{window}'] = speed_diff.rolling(ws, **roll_params).std()
        new_features[f'speed_ratio_m{window}'] = speed_ratio.rolling(ws, **roll_params).mean()

    # 2. 朝向差（如果已有朝向特征）
    if 'A_face_B' in X.columns and 'B_face_A' in X.columns:
        # 朝向不对称性：A面向B但B不面向A，或反之
        facing_asymmetry = X['A_face_B'] - X['B_face_A']
        new_features['facing_asymmetry'] = facing_asymmetry

        for window in [15, 30]:
            ws = _scale(window, fps)
            new_features[f'facing_asym_m{window}'] = facing_asymmetry.rolling(
                ws, min_periods=max(1, ws // 5), center=True
            ).mean()

    # 3. 活动半径差（路径长度）
    for window in [30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # A的路径长度
        A_path_length = A_speed.rolling(ws, **roll_params).sum()
        # B的路径长度
        B_path_length = B_speed.rolling(ws, **roll_params).sum()

        # 路径长度差
        path_diff = A_path_length - B_path_length
        new_features[f'path_diff_{window}'] = path_diff

        # 路径长度比
        new_features[f'path_ratio_{window}'] = A_path_length / (B_path_length + 1e-6)

    # 4. 加速度差
    A_ax = A_vx.diff() * fps
    A_ay = A_vy.diff() * fps
    B_ax = B_vx.diff() * fps
    B_ay = B_vy.diff() * fps

    A_accel = np.sqrt(A_ax**2 + A_ay**2)
    B_accel = np.sqrt(B_ax**2 + B_ay**2)

    accel_diff = A_accel - B_accel
    new_features['accel_diff'] = accel_diff

    for window in [15, 30]:
        ws = _scale(window, fps)
        new_features[f'accel_diff_m{window}'] = accel_diff.rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_tail_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'tail_base' not in avail_A or 'tail_base' not in avail_B:
        return X

    new_features = {}
    tail_A = mouse_pair['A']['tail_base']
    tail_B = mouse_pair['B']['tail_base']

    # 尾部相对位置
    tail_rel_x = tail_B['x'] - tail_A['x']
    tail_rel_y = tail_B['y'] - tail_A['y']
    tail_rel_dist = np.sqrt(tail_rel_x ** 2 + tail_rel_y ** 2)
    new_features['tail_rel_dist'] = tail_rel_dist

    # 尾部相对速度
    tail_rel_vx = tail_rel_x.diff() * fps
    tail_rel_vy = tail_rel_y.diff() * fps
    tail_rel_speed = np.sqrt(tail_rel_vx ** 2 + tail_rel_vy ** 2)
    new_features['tail_rel_speed'] = tail_rel_speed

    # 尾部接近/远离速度
    tail_approach_rate = -tail_rel_dist.diff() * fps
    new_features['tail_approach_rate'] = tail_approach_rate

    # 如果同时有nose，计算尾部-头部相对位置
    if 'nose' in avail_A and 'nose' in avail_B:
        nose_A = mouse_pair['A']['nose']
        nose_B = mouse_pair['B']['nose']

        # A的尾部到B的头部距离
        new_features['tailA_noseB_dist'] = np.sqrt((tail_A['x'] - nose_B['x']) ** 2 + (tail_A['y'] - nose_B['y']) ** 2)

        # B的尾部到A的头部距离
        new_features['tailB_noseA_dist'] = np.sqrt((tail_B['x'] - nose_A['x']) ** 2 + (tail_B['y'] - nose_A['y']) ** 2)

    # 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'tail_rel_dist_m{window}'] = tail_rel_dist.rolling(ws, **roll_params).mean()
        new_features[f'tail_rel_speed_m{window}'] = tail_rel_speed.rolling(ws, **roll_params).mean()
        new_features[f'tail_approach_rate_m{window}'] = tail_approach_rate.rolling(ws, **roll_params).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_nose_tail_body_features(X, mouse_pair, avail_A, avail_B, fps):
    # 检查是否有body_center
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    new_features = {}

    # 体心距离
    center_A = mouse_pair['A']['body_center']
    center_B = mouse_pair['B']['body_center']

    vec_AB = center_B - center_A  # A -> B
    vec_BA = center_A - center_B  # B -> A

    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x ** 2 + rel_y ** 2)

    # 如果有nose和tail_base，计算朝向相关的重叠特征
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        # 身体朝向
        ori_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        ori_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        dot_A = ori_A['x'] * vec_AB['x'] + ori_A['y'] * vec_AB['y']
        dot_B = ori_B['x'] * vec_BA['x'] + ori_B['y'] * vec_BA['y']
        norm_A = (np.sqrt(ori_A['x'] ** 2 + ori_A['y'] ** 2) *
                  np.sqrt(vec_AB['x'] ** 2 + vec_AB['y'] ** 2) + 1e-6)
        norm_B = (np.sqrt(ori_B['x'] ** 2 + ori_B['y'] ** 2) *
                  np.sqrt(vec_BA['x'] ** 2 + vec_BA['y'] ** 2) + 1e-6)

        new_features['A_face_B'] = dot_A / norm_A
        new_features['B_face_A'] = dot_B / norm_B

        # 在 B 的身体坐标系中表达 A 的位置：前后(front) + 左右(side)
        ori_B_norm = np.sqrt(ori_B['x'] ** 2 + ori_B['y'] ** 2) + 1e-6
        front_BA = (vec_BA['x'] * ori_B['x'] + vec_BA['y'] * ori_B['y']) / ori_B_norm
        side_BA = (vec_BA['x'] * (-ori_B['y']) + vec_BA['y'] * ori_B['x']) / ori_B_norm

        new_features['A_front_of_B'] = front_BA
        new_features['A_side_of_B'] = side_BA

        # 在 A 的身体坐标系中表达 B 的位置：前后(front) + 左右(side)
        ori_A_norm = np.sqrt(ori_A['x'] ** 2 + ori_A['y'] ** 2) + 1e-6
        front_AB = (vec_AB['x'] * ori_A['x'] + vec_AB['y'] * ori_A['y']) / ori_A_norm
        side_AB = (vec_AB['x'] * (-ori_A['y']) + vec_AB['y'] * ori_A['x']) / ori_A_norm

        new_features['B_front_of_A'] = front_AB
        new_features['B_side_of_A'] = side_AB

        # 朝向对齐度（余弦相似度）
        ori_alignment = (ori_A['x'] * ori_B['x'] + ori_A['y'] * ori_B['y']) / (
            np.sqrt(ori_A['x']**2 + ori_A['y']**2) * np.sqrt(ori_B['x']**2 + ori_B['y']**2) + 1e-6
        )
        new_features['body_ori_alignment'] = ori_alignment

        # 重叠概率：距离很近 + 朝向对齐
        overlap_score = (1 / (rel_dist + 1.0)) * np.abs(ori_alignment)
        new_features['body_overlap_score'] = overlap_score

        # 重叠状态（二值）
        overlap_binary = ((rel_dist < 8.0) & (np.abs(ori_alignment) > 0.7)).astype(float)
        new_features['body_overlap_binary'] = overlap_binary

        # 滚动窗口统计
        for window in [15, 30, 60]:
            ws = _scale(window, fps)
            roll_params = dict(min_periods=max(1, ws // 5), center=True)
            new_features[f'overlap_score_m{window}'] = overlap_score.rolling(ws, **roll_params).mean()
            new_features[f'overlap_binary_p{window}'] = overlap_binary.rolling(ws, **roll_params).mean()

        # 垂直重叠特征（A在B上方或下方）
        # A在B正上方：front接近0，side接近0，距离很近
        vertical_overlap = ((np.abs(front_BA) < 3.0) &
                           (np.abs(side_BA) < 3.0) &
                           (rel_dist < 8.0)).astype(float)
        new_features['vertical_overlap'] = vertical_overlap

        for window in [15, 30]:
            ws = _scale(window, fps)
            new_features[f'vertical_overlap_p{window}'] = vertical_overlap.rolling(
                ws, min_periods=max(1, ws // 5), center=True
            ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 双鼠交互高级特征：接触类型+角色分化
def _get_front_anchor(mouse_df, avail_parts, mouse_name=''):
    """
    获取老鼠的前端锚点（用于接触检测）

    优先级：nose > head > 耳朵中点

    参数
    ------
    mouse_df : pd.DataFrame
        单只老鼠的关键点数据
    avail_parts : Index
        可用的身体部位列表

    返回
    ------
    anchor : pd.DataFrame or None
        前端锚点的坐标 (x, y)
    anchor_name : str or None
        锚点名称
    """
    # 优先级1: nose
    if 'nose' in avail_parts:
        return mouse_df['nose'], 'nose'

    # 优先级2: head
    if 'head' in avail_parts:
        return mouse_df['head'], 'head'

    # 优先级3: 耳朵中点
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        ear_mid = pd.DataFrame({
            'x': (mouse_df['ear_left']['x'] + mouse_df['ear_right']['x']) / 2,
            'y': (mouse_df['ear_left']['y'] + mouse_df['ear_right']['y']) / 2,
        }, index=mouse_df.index)
        return ear_mid, 'ear_mid'

    # 无可用锚点
    return None, None

def add_contact_semantic_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}

    # 获取A和B的前端锚点
    anchor_A, anchorA_name = _get_front_anchor(mouse_pair['A'], avail_A, 'Mouse A')
    anchor_B, anchorB_name = _get_front_anchor(mouse_pair['B'], avail_B, 'Mouse B')

    # 1. A前端锚点到B各部位的最小距离
    if anchor_A is not None:
        distances_to_B = {}
        for part in avail_B:
            if part in mouse_pair['B'].columns.get_level_values(0):
                part_B = mouse_pair['B'][part]
                dist = np.sqrt((anchor_A['x'] - part_B['x'])**2 + (anchor_A['y'] - part_B['y'])**2)
                distances_to_B[part] = dist

        if distances_to_B:
            # A前端锚点到B任意部位的最小距离
            min_dist_A_front_to_B = pd.concat(distances_to_B.values(), axis=1).min(axis=1)
            new_features['A_front_to_B_min'] = min_dist_A_front_to_B

            # 特定部位的距离（如果存在）
            if 'nose' in distances_to_B:
                new_features['A_front_to_B_nose'] = distances_to_B['nose']
            if 'head' in distances_to_B:
                new_features['A_front_to_B_head'] = distances_to_B['head']
            if 'body_center' in distances_to_B:
                new_features['A_front_to_B_center'] = distances_to_B['body_center']
            if 'tail_base' in distances_to_B:
                new_features['A_front_to_B_tail'] = distances_to_B['tail_base']

            # 接触阈值特征（多个阈值）
            for threshold in [3.0, 5.0, 8.0]:  # cm
                contact_mask = (min_dist_A_front_to_B < threshold).astype(float)
                new_features[f'A_front_contact_{int(threshold)}'] = contact_mask

                # 滚动窗口内的接触占比
                for window in [15, 30, 60]:
                    ws = _scale(window, fps)
                    new_features[f'A_front_contact_{int(threshold)}_p{window}'] = contact_mask.rolling(
                        ws, min_periods=max(1, ws // 5), center=True
                    ).mean()

    # 2. B前端锚点到A各部位的最小距离（对称特征）
    if anchor_B is not None:
        distances_to_A = {}
        for part in avail_A:
            if part in mouse_pair['A'].columns.get_level_values(0):
                part_A = mouse_pair['A'][part]
                dist = np.sqrt((anchor_B['x'] - part_A['x'])**2 + (anchor_B['y'] - part_A['y'])**2)
                distances_to_A[part] = dist

        if distances_to_A:
            min_dist_B_front_to_A = pd.concat(distances_to_A.values(), axis=1).min(axis=1)
            new_features['B_front_to_A_min'] = min_dist_B_front_to_A

            if 'nose' in distances_to_A:
                new_features['B_front_to_A_nose'] = distances_to_A['nose']
            if 'head' in distances_to_A:
                new_features['B_front_to_A_head'] = distances_to_A['head']
            if 'body_center' in distances_to_A:
                new_features['B_front_to_A_center'] = distances_to_A['body_center']
            if 'tail_base' in distances_to_A:
                new_features['B_front_to_A_tail'] = distances_to_A['tail_base']

            for threshold in [3.0, 5.0, 8.0]:
                contact_mask = (min_dist_B_front_to_A < threshold).astype(float)
                new_features[f'B_front_contact_{int(threshold)}'] = contact_mask

                for window in [15, 30, 60]:
                    ws = _scale(window, fps)
                    new_features[f'B_front_contact_{int(threshold)}_p{window}'] = contact_mask.rolling(
                        ws, min_periods=max(1, ws // 5), center=True
                    ).mean()

    # 3. 接触持续时间特征
    # 注意：需要检查 min_dist 特征是否已在本轮计算中生成
    A_min_dist = new_features.get('A_front_to_B_min', X.get('A_front_to_B_min'))
    if A_min_dist is not None:
        contact_binary = (A_min_dist < 5.0).astype(int)
        for window in [30, 60]:
            ws = _scale(window, fps)
            new_features[f'A_contact_duration_{window}'] = contact_binary.rolling(
                ws, min_periods=1, center=True
            ).sum()

    B_min_dist = new_features.get('B_front_to_A_min', X.get('B_front_to_A_min'))
    if B_min_dist is not None:
        contact_binary = (B_min_dist < 5.0).astype(int)
        for window in [30, 60]:
            ws = _scale(window, fps)
            new_features[f'B_contact_duration_{window}'] = contact_binary.rolling(
                ws, min_periods=1, center=True
            ).sum()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def transform_single(single_mouse, body_parts_tracked, fps, section, video_id=None):
    available_body_parts = single_mouse.columns.get_level_values(0)

    # 初始化特征字典
    features = {}

    # 1. 身体部位间距离
    for p1, p2 in itertools.combinations(body_parts_tracked, 2):
        if p1 in available_body_parts and p2 in available_body_parts:
            features[f"{p1}+{p2}"] = np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)

    # 先创建初始 DataFrame
    X = pd.DataFrame(features, index=single_mouse.index)
    # Reindex using list comprehension to ensure order (optional, but keeps consistency)
    cols = [f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)]
    X = X.reindex(columns=cols, copy=False)


    # === Best: missingness-as-signal features ===
    X = add_missingness_features_single(X, single_mouse, fps, section)
    # === End best code ===


    # 重新开始收集后续特征
    new_features = {}

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        new_features['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        new_features['sp_lf'] = np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False)
        new_features['sp_rt'] = np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False)
        new_features['sp_lf2'] = np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False)
        new_features['sp_rt2'] = np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            new_features[f'ear_o{off}'] = ear_d.shift(-o)
        w = _scale(30, fps)
        new_features['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    if 'tail_base' in available_body_parts and center is not None and nose is not None:
        v1 = nose - center
        v2 = single_mouse['tail_base'] - center
        new_features['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    if 'tail_base' in available_body_parts and nose is not None:
        nt_dist = np.sqrt((nose['x'] - single_mouse['tail_base']['x'])**2 +
                          (nose['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            new_features[f'nt_lg{lag}'] = nt_dist.shift(l)
            new_features[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    if center is not None:
        cx = center['x']
        cy = center['y']

        # === Best: winsorize once, then cheap rolling max-min (avoid rolling quantile cost) ===
        cx_clip = cx.clip(lower=cx.quantile(0.01), upper=cx.quantile(0.99))
        cy_clip = cy.clip(lower=cy.quantile(0.01), upper=cy.quantile(0.99))
        # === End best code ===

        window = [5, 15, 30, 60] if section == 9 else [15, 30, 60, 120]
        for w in window:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)

            new_features[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            new_features[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            new_features[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            new_features[f'cy_s{w}'] = cy.rolling(ws, **roll).std()

            # FIX: robust range (after winsorize)
            new_features[f'x_rng{w}'] = cx_clip.rolling(ws, **roll).max() - cx_clip.rolling(ws, **roll).min()
            new_features[f'y_rng{w}'] = cy_clip.rolling(ws, **roll).max() - cy_clip.rolling(ws, **roll).min()



            new_features[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            new_features[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

    # 一次性合并目前的特征
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # 调用子函数（每个子函数现在都会返回一个新的合并后的 DataFrame）
    if center is not None:
        X = add_curvature_features(X, cx, cy, fps, section)
        X = add_multiscale_features(X, cx, cy, fps, section)
        X = add_state_features(X, cx, cy, fps, section)
        X = add_longrange_features(X, cx, cy, fps)

    X = add_posture_stability_features(X, single_mouse, available_body_parts, fps)
    X = add_head_elevation_features(X, single_mouse, available_body_parts, fps, section)
    X = add_body_width_features(X, single_mouse, available_body_parts, fps, section)
    X = add_pose_shape_features(X, single_mouse, available_body_parts, fps, section)
    X = add_head_body_decoupled_features(X, single_mouse, available_body_parts, fps, section)
    X = add_body_axis_motion_features(X, single_mouse, available_body_parts, fps, section)
    X = add_high_freq_micromotion_features(X, single_mouse, available_body_parts, fps, section)
    if video_id is not None:
        X = add_arena_spatial_features(X, single_mouse, available_body_parts, fps, section, video_id)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    features = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            features[f"{p1}+{p2}"] = np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)

    X = pd.DataFrame(features, index=mouse_pair.index)
    cols = [f"{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)]
    X = X.reindex(columns=cols, copy=False)


    # === Best: missingness-as-signal features for A/B ===
    X = add_missingness_features_pair(X, mouse_pair, fps)
    # === End best code ===


    new_features = {}
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        new_features['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # 添加双鼠交互特征
    X = add_nose_tail_body_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_nose_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_tail_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_ear_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_body_with_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_body_without_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_contact_semantic_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)

# 评估函数
class HostVisibleError(Exception):
    pass

# 训练和提交
def robustify(submission, dataset, traintest, traintest_directory=None):
    """
    对提交结果进行鲁棒性处理和验证

    输入:
        submission: pd.DataFrame - 预测结果，包含列：
            video_id, agent_id, target_id, action, start_frame, stop_frame
        dataset: pd.DataFrame - 数据集元信息
        traintest: str - 'train'或'test'
        traintest_directory: str, optional - tracking数据目录

    输出:
        pd.DataFrame - 处理后的提交结果

    作用:
        1. 删除无效的预测（start_frame >= stop_frame）
        2. 删除重叠的预测区间
        3. 为没有预测的视频填充默认预测
        确保提交格式符合比赛要求
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        # traintest_directory = f"dataset/MABe-mouse-behavior-detection/{traintest}_tracking"

    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")

    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])

    submission = pd.concat(group_list)

    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue

        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions.")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)

    return submission

def predict_multiclass(pred, meta, thresholds):
    """
    将多个二分类预测结果转换为多分类预测区间

    输入:
        pred: pd.DataFrame - 预测概率矩阵，每列对应一个行为类别
        meta: pd.DataFrame - 元数据，包含video_id, agent_id, target_id, video_frame
        thresholds: dict - 每个行为的阈值字典

    输出:
        pd.DataFrame - 预测结果，包含列：
            video_id, agent_id, target_id, action, start_frame, stop_frame

    作用:
        1. 对每一帧选择概率最高的行为
        2. 应用阈值过滤低置信度预测
        3. 将连续的相同预测合并为时间区间
        4. 处理视频边界情况
    """
    ama = np.argmax(pred.values, axis=1)
    max_proba = pred.max(axis=1).values

    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]

    ama = np.where(max_proba >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]

    mask = ama_changes.values >= 0
    mask[-1] = False

    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })

    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame

    return submission_part

def tune_threshold(oof_action, y_action):
    """
    使用Optuna优化二分类阈值以最大化F1分数

    输入:
        oof_action: np.array - 交叉验证的预测概率
        y_action: np.array - 真实标签（0或1）

    输出:
        float - 最优阈值（0到1之间）

    作用:
        通过网格搜索找到使F1分数最大的概率阈值，
        用于将概率预测转换为二分类结果
    """
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0, 1, step=0.01)
        return f1_score(y_action, (oof_action >= threshold), zero_division=0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    study.optimize(objective, n_trials=CFG.threshold_trials, n_jobs=1)
    return study.best_params["threshold"]


# Disk-backed training and resumable checkpoints.
import ast
import hashlib
import tempfile
import contextlib
from collections import OrderedDict
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT_VERSION = 2
BATCH_ROWS = 8192
log_file = None
checkpoint_thresholds = {'single': {}, 'pair': {}}
checkpoint_scores = []
resume_root = None
resume_scores = {}
run_config = {}
feature_store = None
fast_resume_keys = set()
resume_format = 0
run_started = 0
max_hours = 3.0
max_new_tasks = 3
new_tasks_this_run = 0
cache_budget = 4 * 1024**3
output_budget = 6 * 1024**3


def log_print(message):
    print(message, flush=True)
    if log_file is not None and not log_file.closed:
        log_file.write(str(message) + '\n')
        log_file.flush()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def feature_signature(source):
    # Ignore comments and runtime settings; compare the actual feature/label code.
    start = source.index('drop_body_parts = [')
    stop = source.index('def tune_threshold(')
    return hashlib.sha256(ast.dump(ast.parse(source[start:stop]), include_attributes=False).encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def atomic_dump(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(value, tmp, compress=3)
    tmp.replace(path)


def atomic_npy(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)


def write_status(status, error=None):
    if CFG.output_dir:
        atomic_json({'status': status, 'updated_at': datetime.now().isoformat(),
                     'trained_tasks': len(checkpoint_scores), 'new_tasks_this_run': new_tasks_this_run,
                     'cache_peak_bytes': feature_store.peak_bytes if feature_store is not None else 0,
                     'resume_source': str(resume_root) if resume_root is not None else None,
                     'error': error},
                    Path(CFG.output_dir) / 'run_status.json')


def checkpoint_task(row):
    key = (str(row['section']), row['kind'], row['action'])
    checkpoint_scores[:] = [r for r in checkpoint_scores
                            if (str(r['section']), r['kind'], r['action']) != key]
    checkpoint_scores.append(row)
    checkpoint_thresholds[row['kind']].setdefault(str(row['section']), {})[row['action']] = float(row['threshold'])
    out = Path(CFG.output_dir)
    atomic_dump(checkpoint_thresholds, out / 'thresholds.pkl')
    atomic_dump(pd.DataFrame(checkpoint_scores), out / 'scores.pkl')
    tmp = out / 'scores.csv.tmp'
    pd.DataFrame(checkpoint_scores).to_csv(tmp, index=False)
    tmp.replace(out / 'scores.csv')
    write_status('running')


def model_semantics(params):
    return {key: value for key, value in params.items() if key not in ('n_jobs', 'verbosity')}


def check_resume_contract(root, config, configurations, categories):
    root = Path(root)
    old = json.loads((root / 'config.json').read_text())
    if old.get('checkpoint_format', 1) not in (1, FORMAT_VERSION):
        raise ValueError('Resume checkpoint format differs from this script.')
    if model_semantics(old['model']) != model_semantics(config['model']):
        raise ValueError('Resume model parameters differ from this run.')
    for key in ('n_splits', 'cv', 'seed', 'threshold_trials'):
        if old[key] != config[key]:
            raise ValueError(f'Resume setting differs: {key}')
    if feature_signature((root / 'train_fold555.py').read_text()) != config['feature_signature']:
        raise ValueError('Resume feature/label code differs from this run.')
    if json.loads((root / 'body_part_configurations.json').read_text()) != configurations:
        raise ValueError('Resume body-part configuration numbering differs.')
    if json.loads((root / 'meta_categories.json').read_text()) != categories:
        raise ValueError('Resume metadata categories differ.')
    previous_packages = json.loads((root / 'environment.json').read_text())['packages']
    if previous_packages != {name: version(name) for name in previous_packages}:
        raise ValueError('Install the original package versions before resuming.')
    if 'dataset_signature' in old and old['dataset_signature'] != config['dataset_signature']:
        raise ValueError('Resume competition data changed (metadata or parquet contents).')
    return old


def dataset_signature(root):
    """One streaming hash pass; paths, labels and tracking values define the data version."""
    h = hashlib.sha256()
    paths = [root / 'train.csv', root / 'test.csv']
    for folder in ('train_tracking', 'train_annotation'):
        paths.extend(sorted((root / folder).rglob('*.parquet')))
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(digest_file(path).encode())
    return h.hexdigest()


def prepare_training():
    global train, test, arena_data, body_parts_tracked_list, log_file
    global checkpoint_thresholds, checkpoint_scores, resume_root, resume_scores, run_config
    global fast_resume_keys, resume_format, run_started, max_hours, max_new_tasks, new_tasks_this_run, cache_budget, output_budget
    run_started = perf_counter()
    max_hours = float(os.environ.get('MABE_MAX_HOURS', '3'))
    max_new_tasks = int(os.environ.get('MABE_MAX_NEW_TASKS', '3'))
    cache_budget = int(float(os.environ.get('MABE_CACHE_GB', '4')) * 1024**3)
    output_budget = int(float(os.environ.get('MABE_OUTPUT_GB', '6')) * 1024**3)
    if max_hours < 0 or max_new_tasks < 0 or cache_budget < 0 or output_budget <= 0:
        raise ValueError('Resource limits must be non-negative; output budget must be positive.')
    fast_resume_keys, new_tasks_this_run, resume_format = set(), 0, 0
    if CFG.mode != 'validate':
        raise ValueError('This script is a training entry point; mode must be validate.')
    available = cpu_count()
    for field, env_name in [('feature_jobs', 'MABE_FEATURE_JOBS'), ('model_jobs', 'MABE_MODEL_JOBS')]:
        number = int(os.environ.get(env_name, getattr(CFG, field)))
        if number != -1 and number < 1:
            raise ValueError(f'{env_name} must be -1 or a positive integer.')
        setattr(CFG, field, available if number == -1 else min(number, available))
    CFG.feature_jobs = 1  # Keep bounded feature generation serial; model fitting remains multi-CPU.
    root = Path(os.environ.get('MABE_DATA_DIR', Path(CFG.train_path).parent))
    required = ('train.csv', 'test.csv', 'train_tracking', 'train_annotation')
    if not all((root / item).exists() for item in required):
        choices = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
                   if all((p.parent / item).exists() for item in required)]
        if len(choices) != 1 or 'MABE_DATA_DIR' in os.environ:
            raise FileNotFoundError('Attach the MABe competition data or set MABE_DATA_DIR.')
        root = choices[0]
    CFG.train_path, CFG.test_path = str(root / 'train.csv'), str(root / 'test.csv')
    CFG.train_tracking_path, CFG.train_annotation_path = str(root / 'train_tracking'), str(root / 'train_annotation')
    CFG.test_tracking_path = str(root / 'test_tracking')
    base = Path(os.environ.get('MABE_OUTPUT_BASE', '/kaggle/working' if Path('/kaggle').exists() else str(Path.cwd())))
    out = base / ('mabe_train_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')) / 'fold555'
    out.mkdir(parents=True, exist_ok=False)
    CFG.output_dir = str(out)
    log_file = (out / 'training.log').open('w', buffering=1)
    checkpoint_thresholds, checkpoint_scores, resume_scores = {'single': {}, 'pair': {}}, [], {}
    CFG.model.set_params(n_jobs=CFG.model_jobs)
    source = Path(__file__).read_text()
    (out / 'train_fold555.py').write_text(source)
    packages = {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'koolbox', 'optuna', 'polars', 'pyarrow', 'joblib', 'scipy', 'tqdm')}
    atomic_json({'python': sys.version, 'platform': platform.platform(), 'packages': packages}, out / 'environment.json')
    log_print(f'Data: {root}\nOutput: {out}\nCPU available={available}; feature workers={CFG.feature_jobs}; model threads={CFG.model_jobs}')
    train, test = pd.read_csv(CFG.train_path), pd.read_csv(CFG.test_path)
    for dataset in (train, test):
        dataset['n_mice'] = 4 - dataset[[f'mouse{i}_strain' for i in range(1, 5)]].isna().sum(axis=1)
    for mapping in (META_CAT_CATEGORIES, META_CAT_KNOWN_SET, META_CAT_ENCODERS, META_CAT_UNK):
        mapping.clear()
    for col in META_CAT_COLS:
        if col in train:
            values = sorted(train[col].dropna().astype(str).unique().tolist())
            META_CAT_CATEGORIES[col], META_CAT_KNOWN_SET[col] = values, set(values)
            if META_CAT_ENCODING == 'label':
                META_CAT_ENCODERS[col] = {v: i for i, v in enumerate(values)}
                META_CAT_UNK[col] = len(values)
    body_parts_tracked_list = list(np.unique(train.body_parts_tracked))
    arena_data = pd.concat([dataset[['video_id', 'arena_width_cm', 'arena_height_cm', 'arena_shape']]
                           for dataset in (train, test)], ignore_index=True).drop_duplicates('video_id').set_index('video_id')
    atomic_json(body_parts_tracked_list, out / 'body_part_configurations.json')
    atomic_json(META_CAT_CATEGORIES, out / 'meta_categories.json')
    log_print('Checking competition data fingerprint for safe resume...')
    run_config = {'model': CFG.model.get_params(), 'n_splits': CFG.n_splits, 'cv': repr(CFG.cv),
                  'seed': CFG.SEED, 'threshold_trials': CFG.threshold_trials,
                  'section_start': CFG.section_start, 'section_stop': CFG.section_stop,
                  'data_root': str(root), 'available_cpus': available,
                  'feature_jobs': CFG.feature_jobs, 'model_jobs': CFG.model_jobs,
                  'feature_signature': feature_signature(source), 'dataset_signature': dataset_signature(root),
                  'checkpoint_format': FORMAT_VERSION, 'cache_budget_bytes': cache_budget, 'output_budget_bytes': output_budget,
                  'max_hours': max_hours, 'max_new_tasks': max_new_tasks}
    atomic_json(run_config, out / 'config.json')
    value = os.environ.get('MABE_RESUME_DIR', '').strip()
    resume_root = Path(value).expanduser() if value else None
    if resume_root is not None:
        if not (resume_root / 'config.json').is_file():
            matches = [p.parent for p in resume_root.rglob('config.json')
                       if (p.parent / 'train_fold555.py').is_file()]
            if len(matches) != 1:
                raise ValueError('MABE_RESUME_DIR must identify one previous fold555 output directory.')
            resume_root = matches[0]
        old_config = check_resume_contract(resume_root, run_config, body_parts_tracked_list, META_CAT_CATEGORIES)
        resume_format = old_config.get('checkpoint_format', 0)
        if 'dataset_signature' not in old_config:
            log_print('LEGACY RESUME: old data fingerprint unavailable; feature code, configuration, categories and each task\'s frame labels will be checked. Use the same competition data.')
        if (resume_root / 'scores.pkl').is_file():
            for row in joblib.load(resume_root / 'scores.pkl').to_dict('records'):
                resume_scores[(str(row['section']), row['kind'], row['action'])] = row
        for marker in resume_root.glob('*/*/task_complete.json'):
            row = json.loads(marker.read_text())['score']
            resume_scores[(str(row['section']), row['kind'], row['action'])] = row
        log_print(f'Resume source: {resume_root}; recorded completed tasks={len(resume_scores)}')
    else:
        log_print('Starting fresh training.')
    if resume_root is not None:
        carry_progress(out, old_config)
    log_print(f'LIMITS: feature cache={cache_budget/1024**3:.1f} GiB; output budget={output_budget/1024**3:.1f} GiB; max new tasks={max_new_tasks or "unlimited"}; soft time limit={max_hours or "unlimited"} hours; no automatic ZIP')
    write_status('running')
    return out


def ensure_disk(folder, needed):
    free = shutil.disk_usage(folder).free
    if free < needed + 256 * 1024 ** 2:
        raise RuntimeError(f'Not enough temporary disk: need {(needed + 256*1024**2)/1024**3:.2f} GiB, free={free/1024**3:.2f} GiB. Set MABE_CACHE_DIR to a larger scratch disk.')


def action_index(records, action):
    chosen = [dict(r) for r in records if r['actions'].get(action, {}).get('rows', 0)]
    count = sum(r['actions'][action]['rows'] for r in chosen)
    arrays = {'label': np.empty(count, dtype=np.int8), 'video_id': np.empty(count, dtype=np.int64),
              'video_frame': np.empty(count, dtype=np.int64), 'agent': np.empty(count, dtype=np.uint8),
              'target': np.empty(count, dtype=np.uint8)}
    offset = 0
    for record in chosen:
        directory = Path(record['directory'])
        labels = pd.read_parquet(directory / 'labels.parquet', columns=[action])[action]
        mask = labels.notna().to_numpy()
        meta = pd.read_parquet(directory / 'meta.parquet').loc[mask]
        end = offset + int(mask.sum())
        record['offset'], record['count'] = offset, end - offset
        arrays['label'][offset:end] = labels[mask].to_numpy(dtype=np.int8)
        arrays['video_id'][offset:end] = meta.video_id.to_numpy()
        arrays['video_frame'][offset:end] = meta.video_frame.to_numpy()
        arrays['agent'][offset:end] = [int(value.removeprefix('mouse')) for value in meta.agent_id]
        arrays['target'][offset:end] = [0 if value == 'self' else int(value.removeprefix('mouse')) for value in meta.target_id]
        offset = end
    return chosen, arrays


def task_columns(records):
    # The original pipeline unions tracking columns across videos before appending
    # metadata. Missing keypoints can introduce extra columns in a later video.
    tracking = dict.fromkeys(name for record in records for name in record['tracking_columns'])
    metadata = dict.fromkeys(name for record in records for name in record['metadata_columns'])
    if tracking.keys() & metadata.keys():
        raise ValueError('Tracking and metadata feature names overlap.')
    return list(tracking) + list(metadata)


def identity_frame(arrays, start, end):
    target = arrays['target'][start:end]
    return pd.DataFrame({'video_id': arrays['video_id'][start:end],
                         'agent_id': [f'mouse{x}' for x in arrays['agent'][start:end]],
                         'target_id': ['self' if x == 0 else f'mouse{x}' for x in target],
                         'video_frame': arrays['video_frame'][start:end]})


def completed_task(section, kind, action, arrays, current_columns):
    key = (str(section), kind, action)
    if key not in resume_scores:
        return False
    source = resume_root / str(section) / action
    required = ['feature_columns.json', 'oof_pred_probs.pkl', 'oof_predictions.parquet', 'threshold.pkl']
    models = list(source.glob('*_trainer_*.pkl'))
    if len(models) != 1 or not all((source / name).is_file() for name in required):
        # Never pretend a half-written checkpoint is usable.
        raise ValueError(f'Recorded completed task is missing files: {source}')
    marker_path = source / 'task_complete.json'
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        for name, expected in marker['sha256'].items():
            if digest_file(source / name) != expected:
                raise ValueError(f'Resume file checksum mismatch: {source / name}')
    probabilities = joblib.load(source / 'oof_pred_probs.pkl')
    model = joblib.load(models[0])
    columns = json.loads((source / 'feature_columns.json').read_text())
    if columns != current_columns:
        raise ValueError(f'Resume feature names or ordering differ: {source}')
    if len(model.estimators) != CFG.n_splits or not model.is_fitted:
        raise ValueError(f'Incomplete model bundle: {source}')
    if any(est.n_features_in_ != len(columns) for est in model.estimators):
        raise ValueError(f'Resume model feature count mismatch: {source}')
    if not np.array_equal(model.oof_preds, probabilities):
        raise ValueError(f'Resume probability files disagree: {source}')
    if not np.array_equal(np.asarray(model.cv_args['groups']), arrays['video_id']):
        raise ValueError(f'Resume training video order differs: {source}')
    offset = 0
    for batch in pq.ParquetFile(source / 'oof_predictions.parquet').iter_batches(batch_size=BATCH_ROWS):
        frame = batch.to_pandas()
        end = offset + len(frame)
        expected = identity_frame(arrays, offset, end)
        if any(not np.array_equal(frame[col].to_numpy(), expected[col].to_numpy()) for col in expected):
            raise ValueError(f'Resume frame identity differs: {source}')
        if not np.array_equal(frame['label'], arrays['label'][offset:end]) or not np.array_equal(frame['prediction'], probabilities[offset:end]):
            raise ValueError(f'Resume labels/probabilities differ: {source}')
        offset = end
    if offset != len(arrays['label']) or len(probabilities) != offset:
        raise ValueError(f'Resume row count differs: {source}')
    threshold = float(joblib.load(source / 'threshold.pkl'))
    value = f1_score(arrays['label'], probabilities >= threshold, zero_division=0)
    if not np.isclose(value, resume_scores[key]['binary F1 score'], atol=1e-12, rtol=0):
        raise ValueError(f'Resume validation score differs: {source}')
    target = Path(CFG.output_dir) / str(section) / action
    if not target.exists():
        shutil.copytree(source, target)
    names = [models[0].name, *required]
    atomic_json({'format': FORMAT_VERSION, 'score': dict(resume_scores[key], threshold=threshold),
                 'signature': 'completed-task-validated',
                 'sha256': {name:digest_file(target/name) for name in names}}, target/'task_complete.json')
    row = dict(resume_scores[key], threshold=threshold, reused=True)
    checkpoint_task(row)
    log_print(f'REUSE section={section}, {kind}/{action}: all {CFG.n_splits} fold models validated and copied')
    del model, probabilities
    gc.collect()
    return True


def fold_signature(records, arrays, columns, kind, section, action):
    h = hashlib.sha256(json.dumps({'model': model_semantics(CFG.model.get_params()), 'cv': repr(CFG.cv),
                                  'features': run_config['feature_signature'], 'columns': columns,
                                  'kind': kind, 'section': section, 'action': action}, sort_keys=True).encode())
    for record in records:
        h.update(json.dumps({key: record[key] for key in ('feature_hash', 'meta_hash', 'label_hash')}, sort_keys=True).encode())
    for array in arrays.values():
        h.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return h.hexdigest()


class PauseTraining(Exception):
    """A normal exit that preserves checkpoints before reaching a resource limit."""


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def check_budget(stage, extra=0, check_time=True):
    if check_time and max_hours and perf_counter() - run_started >= max_hours * 3600:
        raise PauseTraining(f'Time budget reached before {stage}; resume this output in the next run.')
    root = Path(CFG.output_dir).parent.parent
    used = tree_bytes(root)
    if used + extra + 256 * 1024**2 > output_budget:
        raise PauseTraining(f'Output budget reached before {stage}: {used/1024**3:.2f} GiB. Preserve this output; split runs or increase MABE_OUTPUT_GB after checking disk space.')
    if shutil.disk_usage(root).free < extra + 1024**3:
        raise PauseTraining(f'Less than 1 GiB disk reserve before {stage}; checkpoints have been retained.')


def make_features(kind, data, meta, parts, section):
    fps = _fps_from_meta(meta, {}, default_fps=30.0)
    tracking = (transform_single(data, parts, fps, section, video_id=int(meta.video_id.iloc[0]))
                if kind == 'single' else transform_pair(data, parts, fps))
    metadata = meta_to_features(meta.reset_index(drop=True))
    features = pd.concat([tracking.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1).astype(np.float32)
    if features.columns.duplicated().any():
        raise ValueError('Duplicate feature names.')
    return features, list(tracking.columns), list(metadata.columns)


def feature_digest(features):
    h = hashlib.sha256()
    for start in range(0, len(features), BATCH_ROWS):
        values = np.array(features.iloc[start:start+BATCH_ROWS], dtype=np.float32, order='C', copy=True)
        values[np.isnan(values)] = np.nan
        h.update(memoryview(values).cast('B'))
    return h.hexdigest()


class FeatureStore:
    """A bounded disk cache; evicted features are recomputed from the original input."""
    def __init__(self, root, budget):
        self.root, self.budget = Path(root), int(budget)
        self.files = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.block_key = self.block = None
        self.active_key = self.active_frame = None

    def put(self, record, features):
        self.active_key, self.active_frame = record['directory'], features
        key = record['directory']
        # Reserve a conservative uncompressed bound before opening a new file.
        bound = int(features.size * 4 * 1.25) + 16 * 1024**2
        if bound > self.budget or shutil.disk_usage(self.root).free < bound + 1024**3:
            return  # Only the current video's features stay in memory; no oversized file.
        while self.files and self.bytes + bound > self.budget:
            old, size = self.files.popitem(last=False)
            (Path(old)/'features.parquet').unlink(missing_ok=True)
            self.bytes -= size
        path = Path(key)/'features.parquet'
        with pq.ParquetWriter(path, pa.Schema.from_pandas(features, preserve_index=False),
                              compression='zstd', use_dictionary=False) as writer:
            for start in range(0, len(features), BATCH_ROWS):
                writer.write_table(pa.Table.from_pandas(features.iloc[start:start+BATCH_ROWS], preserve_index=False))
        size = path.stat().st_size
        if self.bytes + size > self.budget:
            path.unlink()
            return
        self.files[key] = size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.active_key = self.active_frame = None

    def regenerate(self, record):
        self.block_key = self.block = None
        self.active_key = self.active_frame = None
        subset = train[train.video_id == record['video_id']]
        generator = generate_mouse_data(subset, 'train', CFG.train_tracking_path,
                                       generate_single=record['kind']=='single', generate_pair=record['kind']=='pair')
        try:
            for _, data, meta, labels in generator:
                if meta.agent_id.iloc[0] != record['agent_id'] or meta.target_id.iloc[0] != record['target_id']:
                    continue
                features, _, _ = make_features(record['kind'], data, meta, record['parts'], record['section'])
                if list(features.columns) != record['columns'] or feature_digest(features) != record['feature_hash']:
                    raise ValueError('Regenerated features differ from the original cache scan.')
                self.put(record, features)
                return
        finally:
            generator.close()
        raise ValueError(f'Cannot regenerate input record {record["directory"]}')

    def get_rows(self, record, indices):
        result = np.empty((len(indices), len(record['columns'])), dtype=np.float32)
        for block in np.unique(indices // BATCH_ROWS):
            key = (record['directory'], int(block))
            if self.block_key != key:
                path = Path(record['directory'])/'features.parquet'
                if not path.exists() and self.active_key != record['directory']:
                    self.regenerate(record)
                if path.exists():
                    self.files.move_to_end(record['directory'])
                    values = pq.ParquetFile(path).read_row_group(int(block)).to_pandas().to_numpy(dtype=np.float32)
                else:
                    values = self.active_frame.iloc[block*BATCH_ROWS:(block+1)*BATCH_ROWS].to_numpy(dtype=np.float32)
                self.block_key, self.block = key, values
            where = np.flatnonzero(indices // BATCH_ROWS == block)
            result[where] = self.block[indices[where] % BATCH_ROWS]
        return result


def cache_kind(subset, kind, parts, section, folder):
    global feature_store
    feature_store = FeatureStore(folder, cache_budget)
    records = []
    def source():
        return generate_mouse_data(subset, 'train', CFG.train_tracking_path,
                                   generate_single=kind=='single', generate_pair=kind=='pair')
    # First collect small row/label indexes; completed tasks need no feature cache.
    for i, (_, data, meta, labels) in enumerate(source()):
        check_budget('reading frame labels')
        directory = Path(folder)/f'{i:06d}'
        directory.mkdir()
        labels = labels.reset_index(drop=True)
        labels.to_parquet(directory/'labels.parquet', index=False, compression='zstd')
        meta[['video_id','agent_id','target_id','video_frame']].to_parquet(directory/'meta.parquet', index=False, compression='zstd')
        actions = {str(a): {'rows':int(labels[a].notna().sum()), 'positives':int((labels[a]==1).sum())} for a in labels}
        records.append({'directory':str(directory), 'video_id':int(meta.video_id.iloc[0]),
                        'agent_id':str(meta.agent_id.iloc[0]), 'target_id':str(meta.target_id.iloc[0]),
                        'kind':kind, 'parts':parts, 'section':section, 'actions':actions,
                        'columns':[], 'tracking_columns':[], 'metadata_columns':[], 'feature_hash':None,
                        'meta_hash':digest_file(directory/'meta.parquet'), 'label_hash':digest_file(directory/'labels.parquet')})
    pending = {a for r in records for a in r['actions'] if (str(section),kind,a) not in fast_resume_keys}
    if not pending:
        return records
    if max_new_tasks and new_tasks_this_run >= max_new_tasks and any(
            (str(section), kind, action) not in resume_scores for action in pending):
        raise PauseTraining('New-task limit reached; resume this output in the next run.')
    for record, (_, data, meta, labels) in zip(records, source(), strict=True):
        check_budget('feature extraction')
        feature_store.active_key = feature_store.active_frame = None
        features, tracking, metadata = make_features(kind, data, meta, parts, section)
        if len(features) != len(labels) or len(features) != len(meta):
            raise ValueError('Feature and label row counts differ.')
        record.update(columns=list(features.columns), tracking_columns=tracking,
                      metadata_columns=metadata, feature_hash=feature_digest(features))
        if any(record['actions'].get(a, {}).get('rows', 0) for a in pending):
            feature_store.put(record, features)
        del features
    log_print(f'CACHE DONE section={section}, {kind}: {len(records)} records; compressed feature cache={feature_store.bytes/1024**3:.2f} GiB, limit={cache_budget/1024**3:.2f} GiB')
    return records


def feature_batches(records, action, columns):
    destinations = {name:i for i,name in enumerate(columns)}
    for record in records:
        labels = pd.read_parquet(Path(record['directory'])/'labels.parquet', columns=[action])[action]
        selected = np.flatnonzero(labels.notna().to_numpy())
        col_idx = [destinations[name] for name in record['columns']]
        for start in range(0, len(selected), BATCH_ROWS):
            rows = selected[start:start+BATCH_ROWS]
            values = np.full((len(rows),len(columns)), np.nan, dtype=np.float32)
            values[:,col_idx] = feature_store.get_rows(record, rows)
            yield record['offset']+start, values


class FoldSequence(lgb.Sequence):
    """LightGBM reads sampled rows then bounded batches, without a full fold file."""
    batch_size = BATCH_ROWS

    def __init__(self, records, action, columns, folds, fold):
        self.records, self.columns = [], columns
        self.ends = []
        self.length = 0
        destinations = {name:i for i,name in enumerate(columns)}
        for record in records:
            labels = pd.read_parquet(Path(record['directory'])/'labels.parquet',columns=[action])[action]
            selected = np.flatnonzero(labels.notna().to_numpy())
            selected = selected[folds[record['offset']:record['offset']+len(selected)] != fold]
            if not len(selected):
                continue
            self.length += len(selected)
            self.ends.append(self.length)
            self.records.append((record,selected,[destinations[name] for name in record['columns']]))
        self.block_key = self.block = None

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        scalar = isinstance(key, (int,np.integer))
        if scalar:
            if key < 0 or key >= self.length:
                raise IndexError(key)
            start, stop = int(key), int(key)+1
        else:
            start, stop, step = key.indices(self.length)
            if step != 1:
                raise ValueError('Only consecutive LightGBM sequence slices are supported.')
        # Sampled scalar reads are monotonic. Cache a small aligned range rather
        # than reading a parquet block separately for each sampled row.
        if scalar:
            block = start // BATCH_ROWS
            if self.block_key != block:
                self.block = self._slice(block*BATCH_ROWS,min((block+1)*BATCH_ROWS,self.length))
                self.block_key = block
            return self.block[start % BATCH_ROWS].copy()
        return self._slice(start,stop)

    def _slice(self,start,stop):
        values = np.full((stop-start,len(self.columns)),np.nan,dtype=np.float64)
        offset = start
        while offset < stop:
            i = int(np.searchsorted(self.ends,offset,side='right'))
            previous = self.ends[i-1] if i else 0
            end = min(stop,self.ends[i])
            record, rows, columns = self.records[i]
            values[offset-start:end-start,columns] = feature_store.get_rows(record,rows[offset-previous:end-previous])
            offset = end
        return values


class StreamingDataset(lgb.Dataset):
    def get_params(self):
        # In LightGBM 4.6.0, Sequence's sample constructor calls get_params(),
        # whose default allowlist omits random_state and min_child_samples.
        # Pass the same full parameters as the dense constructor so sampling
        # seeds and feature pre-filtering remain identical to sklearn.fit().
        return dict(self.params or {})


def fit_sequence_model(sequence, y):
    # This adapter mirrors the pinned LightGBM 4.6.0 sklearn fit metadata, while
    # using its supported native Sequence input. The saved object remains a
    # standard LGBMClassifier; no custom sequence/cache object is serialized.
    if version('lightgbm') != '4.6.0':
        raise ValueError('This sequence adapter requires lightgbm==4.6.0.')
    estimator = clone(CFG.model)
    if estimator.class_weight is not None or callable(estimator.objective):
        raise ValueError('The streaming adapter expects the original unweighted binary classifier.')
    estimator._le = LabelEncoder().fit(y)
    encoded = estimator._le.transform(y)
    estimator._classes = estimator._le.classes_
    estimator._n_classes = len(estimator._classes)
    estimator._class_map = dict(zip(estimator._classes,range(estimator._n_classes)))
    params = estimator._process_params(stage='fit')
    dataset = StreamingDataset(sequence,label=encoded,feature_name=sequence.columns,params=params)
    estimator._Booster = lgb.train(params,dataset,num_boost_round=estimator.n_estimators)
    estimator._n_features = estimator._Booster.num_feature()
    estimator.n_features_in_ = len(sequence.columns)
    estimator._evals_result = {}
    estimator._best_iteration = estimator._Booster.best_iteration
    estimator._best_score = estimator._Booster.best_score
    estimator.fitted_ = True
    estimator._Booster.free_dataset()
    return estimator


def fit_one_fold(records, action, columns, arrays, folds, fold, scratch):
    sequence = FoldSequence(records, action, columns, folds, fold)
    estimator = fit_sequence_model(sequence, arrays['label'][folds != fold])
    del sequence
    gc.collect()
    probabilities = np.empty(int((folds == fold).sum()), dtype=np.float64)
    position = 0
    for offset, batch in feature_batches(records, action, columns):
        values = batch[folds[offset:offset+len(batch)] == fold]
        if len(values):
            predicted = estimator.predict_proba(pd.DataFrame(values,columns=columns,copy=False))[:,1]
            probabilities[position:position+len(values)] = predicted
            position += len(values)
    assert position == len(probabilities)
    return estimator, probabilities


def save_fold_checkpoint(folder, fold, model, probabilities, signature):
    folder.mkdir(parents=True, exist_ok=True)
    model_name, prediction_name = f'fold_{fold}_model.pkl', f'fold_{fold}_predictions.npy'
    atomic_dump(model, folder / model_name)
    atomic_npy(probabilities, folder / prediction_name)
    atomic_json({'signature': signature, 'fold': fold,
                 'sha256': {name: digest_file(folder / name) for name in (model_name, prediction_name)}},
                folder / f'fold_{fold}.json')


def read_fold_checkpoint(folder, fold, signature, expected_rows, expected_columns):
    marker = Path(folder) / f'fold_{fold}.json'
    if not marker.is_file():
        return None
    try:
        record = json.loads(marker.read_text())
        if record['signature'] != signature or record['fold'] != fold:
            raise ValueError('configuration or data changed')
        for name, checksum in record['sha256'].items():
            if digest_file(Path(folder) / name) != checksum:
                raise ValueError('file checksum mismatch')
        model = joblib.load(Path(folder) / f'fold_{fold}_model.pkl')
        probabilities = np.load(Path(folder) / f'fold_{fold}_predictions.npy', allow_pickle=False)
        if probabilities.shape != (expected_rows,) or model.n_features_in_ != expected_columns or not np.isfinite(probabilities).all():
            raise ValueError('invalid model or predictions')
        return model, probabilities
    except (ValueError, OSError, EOFError, KeyError) as error:
        raise ValueError(f'Cannot reuse fold checkpoint {marker}: {error}') from error


def train_action(records, arrays, kind, section, action, scratch, columns=None):
    global new_tasks_this_run
    y, groups = arrays['label'], arrays['video_id']
    columns = task_columns(records) if columns is None else columns
    if completed_task(section, kind, action, arrays, columns):
        return
    if len(np.unique(groups)) < CFG.n_splits or not (y == 1).any():
        log_print(f'SKIP section={section}, {kind}/{action}: fewer than {CFG.n_splits} videos or no positive labels')
        return
    if max_new_tasks and new_tasks_this_run >= max_new_tasks:
        raise PauseTraining('New-task limit reached; resume from this output in the next run.')
    check_budget('next behavior')
    folder = Path(CFG.output_dir) / str(section) / action
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoints'
    folds = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(CFG.cv.split(np.empty((len(y), 0)), y, groups)):
        folds[valid] = fold
    arrays_with_folds = dict(arrays, fold=folds)
    signature = fold_signature(records, arrays_with_folds, columns, kind, section, action)
    probabilities = np.zeros(len(y), dtype=np.float64)
    models, fold_scores = [], []
    log_print(f'TRAIN section={section}, {kind}/{action}, rows={len(y):,}, features={len(columns)}, streamed input; no full-fold matrix file')
    for fold in range(CFG.n_splits):
        check_budget(f'fold {fold+1}', extra=len(y)*16)
        start = perf_counter()
        valid_mask = folds == fold
        saved = None
        if resume_root is not None and resume_format == FORMAT_VERSION:
            saved = read_fold_checkpoint(resume_root / str(section) / action / 'checkpoints', fold,
                                         signature, int(valid_mask.sum()), len(columns))
        if saved is None:
            estimator, predicted = fit_one_fold(records, action, columns, arrays, folds, fold, scratch)
            log_print(f'FOLD FIT section={section}, {action}, fold={fold+1}/{CFG.n_splits}, elapsed={perf_counter()-start:.1f}s')
        else:
            estimator, predicted = saved
            log_print(f'FOLD REUSE section={section}, {action}, fold={fold+1}/{CFG.n_splits}')
        save_fold_checkpoint(checkpoint, fold, estimator, predicted, signature)
        log_print(f'CHECKPOINT SAVED section={section}, {action}, fold={fold+1}: cache={feature_store.bytes/1024**3:.2f} GiB; output={tree_bytes(Path(CFG.output_dir))/1024**3:.2f} GiB')
        probabilities[valid_mask] = predicted
        models.append(estimator)
        fold_scores.append(f1_score(y[valid_mask], predicted >= 0.5, zero_division=0))
        write_status('running')
    trainer = Trainer(estimator=clone(CFG.model), cv=CFG.cv, cv_args={'groups': groups},
                      metric=f1_score, task='binary', verbose=False, save=False)
    trainer.estimators, trainer.oof_preds, trainer.is_fitted = models, probabilities, True
    trainer.y_min, trainer.y_max = y.min(), y.max()
    trainer.fold_scores = fold_scores
    trainer.overall_score = f1_score(y, probabilities >= 0.5, zero_division=0)
    check_budget('final task files', extra=len(y)*40, check_time=False)
    model_name = 'lgbmclassifier_trainer_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.pkl'
    atomic_dump(trainer, folder / model_name)
    atomic_json(columns, folder / 'feature_columns.json')
    atomic_dump(probabilities, folder / 'oof_pred_probs.pkl')
    log_print(f'THRESHOLD section={section}, {action}: {CFG.threshold_trials} trials')
    threshold = float(tune_threshold(probabilities, y))
    atomic_dump(threshold, folder / 'threshold.pkl')
    tmp = folder / 'oof_predictions.parquet.tmp'
    writer = None
    try:
        for start in range(0, len(y), BATCH_ROWS):
            end = min(start+BATCH_ROWS, len(y))
            data = identity_frame(arrays, start, end)
            data['label'], data['prediction'], data['fold'] = y[start:end], probabilities[start:end], folds[start:end]
            table = pa.Table.from_pandas(data, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(folder / 'oof_predictions.parquet')
    score = {'section': section, 'kind': kind, 'action': action, 'body_parts_tracked_str': body_parts_tracked_list[section],
             'threshold': threshold, 'binary F1 score': f1_score(y, probabilities >= threshold, zero_division=0), 'reused': False}
    names = [model_name, 'feature_columns.json', 'oof_pred_probs.pkl', 'oof_predictions.parquet', 'threshold.pkl']
    atomic_json({'format': FORMAT_VERSION, 'signature': signature, 'score': score,
                 'sha256': {name: digest_file(folder / name) for name in names}}, folder / 'task_complete.json')
    checkpoint_task(score)
    new_tasks_this_run += 1
    shutil.rmtree(checkpoint)  # The full bundle now contains all five models; no duplicate fold files are needed.
    log_print(f'TASK DONE section={section}, {kind}/{action}: F1={score["binary F1 score"]:.4f}, threshold={threshold:.2f}')
    del trainer, models, probabilities
    gc.collect()


def carry_progress(out, old_config):
    for section in resume_root.iterdir():
        if not section.is_dir() or not section.name.isdigit():
            continue
        for action in section.iterdir():
            if not action.is_dir():
                continue
            try:
                check_budget('copying previous checkpoints', extra=tree_bytes(action), check_time=False)
            except PauseTraining as error:
                raise RuntimeError('Previous outputs do not fit the configured output budget. Keep using the ORIGINAL resume source; this new output is incomplete. ' + str(error)) from error
            shutil.copytree(action, out/section.name/action.name, ignore=shutil.ignore_patterns('*.tmp'))
    for key, row in resume_scores.items():
        folder = out/key[0]/key[2]
        required = ['threshold.pkl','feature_columns.json','oof_pred_probs.pkl','oof_predictions.parquet']
        if not all((folder/name).is_file() for name in required) or len(list(folder.glob('*_trainer_*.pkl'))) != 1:
            raise ValueError(f'Incomplete previous completed task: {folder}')
        marker = folder/'task_complete.json'
        if marker.is_file():
            value = json.loads(marker.read_text())
            for name, expected in value['sha256'].items():
                if digest_file(folder/name) != expected:
                    raise ValueError(f'Resume checksum mismatch: {folder/name}')
            if 'dataset_signature' in old_config:
                fast_resume_keys.add(key)
        checkpoint_task(dict(row, threshold=float(joblib.load(folder/'threshold.pkl')), reused=True))
    log_print(f'CARRIED: {len(resume_scores)} completed tasks and saved folds; {len(fast_resume_keys)} tasks verified by matching data version and file checksums.')


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    out = prepare_training()
    scratch_base = Path(os.environ.get('MABE_CACHE_DIR', '/kaggle/temp' if Path('/kaggle').exists() else tempfile.gettempdir()))
    scratch_base.mkdir(parents=True, exist_ok=True)
    stop = len(body_parts_tracked_list) if CFG.section_stop is None else min(CFG.section_stop, len(body_parts_tracked_list))
    try:
        for section in range(CFG.section_start, stop):
            parts = json.loads(body_parts_tracked_list[section])
            if len(parts) > 5:
                parts = [part for part in parts if part not in drop_body_parts]
            subset = train[train.body_parts_tracked == body_parts_tracked_list[section]]
            log_print(f'\nSection {section}/{len(body_parts_tracked_list)-1}; filesystem free={shutil.disk_usage(scratch_base).free/1024**3:.1f} GiB; saved output={tree_bytes(out)/1024**3:.2f} GiB')
            for kind in ('single', 'pair'):
                with tempfile.TemporaryDirectory(prefix=f'mabe_{section}_{kind}_', dir=scratch_base) as temporary:
                    records = cache_kind(subset, kind, parts, section, temporary)
                    actions = list(dict.fromkeys(action for record in records for action in record['actions']))
                    for action in actions:
                        if not action or not all(c.isalnum() or c in '_-' for c in action):
                            raise ValueError(f'Invalid action name: {action!r}')
                        if (str(section), kind, action) in fast_resume_keys:
                            log_print(f'REUSE section={section}, {kind}/{action}: data version and saved files verified')
                            continue
                        chosen, arrays = action_index(records, action)
                        train_action(chosen, arrays, kind, section, action, temporary, task_columns(records))
                        del chosen, arrays
                        gc.collect()
                    del records
        if not checkpoint_scores:
            raise RuntimeError('No models trained or reused; check selected sections and labels.')
        write_status('complete')
        log_print(f'COMPLETE: {len(checkpoint_scores)} behavior tasks saved in {out}')
    except PauseTraining as error:
        write_status('paused', str(error))
        log_print(f'PAUSED SAFELY: {error}')
        log_print(f'KEEP OUTPUT AND RESUME: {out}')
    finally:
        log_print(f'OUTPUT: {out}; saved={tree_bytes(out)/1024**3:.2f} GiB. No ZIP copy created.')
        if log_file is not None and not log_file.closed:
            log_file.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if CFG.output_dir:
            write_status('incomplete', traceback.format_exc())
        raise
    finally:
        if log_file is not None and not log_file.closed:
            log_file.close()
