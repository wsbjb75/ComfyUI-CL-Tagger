"""CL Tagger v2 模型 / 词表 / 校准数据管理工具。

模型目录（cl_tagger_v2 的实际文件布局）：
    <ComfyUI>/models/onnx/cl_tagger/
        model.onnx              推理模型（外部权重在 model.onnx.data）
        model_vocabulary.json   标签词表 {"tag_to_idx": {tag: idx, ...}}
        model_tag_metrics.npz   校准数据（calibration_table / best_thr 等）

本模块负责：目录注册、模型/词表/校准文件扫描与加载、onnxruntime 会话懒加载。
"""

import json
import os

import folder_paths

# 注册两个候选模型目录：用户实际的 onnx/cl_tagger，以及兼容旧布局的 cl_tagger_v2
MODEL_DIR_NAME = "cl_tagger_v2"
SUB_DIRS = ["onnx/cl_tagger", "cl_tagger_v2", "cl_tagger"]

try:
    models_dir = folder_paths.models_dir
    for sub in SUB_DIRS:
        try:
            folder_paths.add_model_folder_path(MODEL_DIR_NAME, os.path.join(models_dir, sub))
        except Exception:
            pass
except Exception:
    pass

_session_cache = {}
_vocab_cache = {}
_calib_cache = {}


def get_model_dirs():
    """返回所有候选模型目录（兼容 ComfyUI 1.x / 0.x 的 API 差异）。"""
    try:
        return folder_paths.get_folder_paths(MODEL_DIR_NAME)
    except Exception:
        out = []
        try:
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = None
        if models_dir:
            for sub in SUB_DIRS:
                out.append(os.path.join(models_dir, sub))
        return out


def ensure_model_dir():
    for d in get_model_dirs():
        os.makedirs(d, exist_ok=True)
    return get_model_dirs()[0]


def list_models():
    """扫描模型目录，返回所有 .onnx 文件的绝对路径（按文件名排序）。"""
    found = []
    for d in get_model_dirs():
        if os.path.isdir(d):
            found.extend(
                p for p in (os.path.join(d, f) for f in os.listdir(d))
                if os.path.isfile(p) and p.lower().endswith(".onnx"))
    return sorted(set(found))


def _dir_of(model_path):
    return os.path.dirname(model_path)


def find_vocab(model_path):
    """词表：model_vocabulary.json（与模型同目录）。"""
    cand = os.path.join(_dir_of(model_path), "model_vocabulary.json")
    if os.path.isfile(cand):
        return cand
    return None


def load_vocab(vocab_path):
    """解析 model_vocabulary.json，返回按输出通道索引排序的标签名数组。

    返回 (tag_names, tag_index)，tag_names[i] 为通道 i 的标签名（缺失通道为空串）。
    """
    if vocab_path in _vocab_cache:
        return _vocab_cache[vocab_path]
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tag_to_idx = data.get("tag_to_idx", data)
    size = max(tag_to_idx.values()) + 1 if tag_to_idx else 0
    names = [""] * size
    for tag, idx in tag_to_idx.items():
        if 0 <= idx < size:
            names[idx] = tag
    _vocab_cache[vocab_path] = (names, size)
    return _vocab_cache[vocab_path]


def find_calibration(model_path):
    """校准数据：model_tag_metrics.npz（与模型同目录）。"""
    cand = os.path.join(_dir_of(model_path), "model_tag_metrics.npz")
    if os.path.isfile(cand):
        return cand
    return None


def load_calibration(npz_path):
    """加载校准数据，返回 (calibration_table, best_thr)。

    calibration_table: (num_tags, 100) float16 -> 校准概率
    best_thr:          (num_tags,) 各标签最佳 F1 阈值（校准后概率尺度）
    加载时把 99 号 bin（低样本噪声桶）平滑为 98 号 bin 的值。
    """
    if npz_path in _calib_cache:
        return _calib_cache[npz_path]

    import numpy as np

    with np.load(npz_path) as data:
        cal = data["calibration_table"].copy()
        best_thr = data["best_thr"].copy()

    if cal.ndim == 2 and cal.shape[1] >= 100:
        cal[:, 99] = cal[:, 98]          # 平滑末端噪声桶
    elif cal.ndim == 2:
        cal = cal[:, :100] if cal.shape[1] > 100 else cal

    _calib_cache[npz_path] = (cal, best_thr)
    return _calib_cache[npz_path]


def get_session(model_path):
    """懒加载 onnxruntime 会话，带缓存（模型外部权重 model.onnx.data 自动定位）。"""
    if model_path in _session_cache:
        return _session_cache[model_path]

    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                             "CPUExecutionProvider") if p in available]
    providers = providers or ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    _session_cache[model_path] = session
    return session


def clear_cache():
    _session_cache.clear()
    _vocab_cache.clear()
    _calib_cache.clear()
