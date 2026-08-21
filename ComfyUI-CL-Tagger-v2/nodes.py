"""CL Tagger v2 —— 图像标签反推（booru 风格打标）ComfyUI 节点。

适配 cella110n/cl_tagger_v2：
    - 模型：SigLIP2 (google/siglip2-so400m-patch14-384) + 108139 标签头
    - 输入：384x384，归一化 mean=std=0.5
    - 输出：logits -> sigmoid -> 直方图校准（model_tag_metrics.npz 查表）
推理基于 onnxruntime，模型文件放在 <ComfyUI>/models/onnx/cl_tagger/。
"""

import os
import re

import numpy as np
from PIL import Image

from .model_utils import (
    list_models,
    find_vocab,
    load_vocab,
    find_calibration,
    load_calibration,
    get_session,
    ensure_model_dir,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

INPUT_SIZE = 384          # SigLIP2 so400m patch14-384
NUM_BINS = 100

# danbooru 角色消歧义惯例：「名字 (作品名)」
_CHAR_RE = re.compile(r"^[^(]+\(([^()]+)\)$")


# ---------------------------------------------------------------- 节点

class CLTaggerV2:
    """用 CL Tagger v2 模型给图像打标签，输出逗号分隔的标签串。"""

    @classmethod
    def INPUT_TYPES(cls):
        models = list_models()
        if not models:
            models = ["(未找到模型，请先下载并放入 models/onnx/cl_tagger/)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models,),
                "threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "标签置信度阈值（校准后概率）。调低输出更多标签，调高更精",
                }),
                "character_threshold": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "角色类标签的置信度阈值。cl_tagger_v2 词表无角色分类列，"
                               "角色判定依据：括号消歧义形式「名字 (作品)」+ character_tags 自定义名单",
                }),
                "exclude_tags": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "需要排除的标签，逗号分隔（如 watermark, signature）",
                }),
                "tag_replacement": ("STRING", {
                    "default": "_",
                    "tooltip": "标签中下划线的替换字符，常用空格以适配自然语言提示词",
                }),
                "sort_by": (["confidence", "alphabetical", "none"],),
                "with_scores": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "输出标签时附带校准概率",
                }),
            },
            "optional": {
                "character_tags": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "自定义角色名单，逗号分隔（如 hatsune miku, kagamine rin）。"
                               "名单内标签按 character_threshold 过滤",
                }),
                "use_best_threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用各标签训练集最优 F1 阈值（model_tag_metrics.npz 的 best_thr），"
                               "优先于 threshold",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("tags", "tags_with_scores", "count", "scores")
    FUNCTION = "tag"
    CATEGORY = "image/tagger"
    OUTPUT_NODE = False

    def tag(self, image, model="", threshold=0.50, character_threshold=0.65,
            exclude_tags="", tag_replacement="_", sort_by="confidence",
            with_scores=False, character_tags="", use_best_threshold=False):
        # 1. 模型 / 词表 / 校准数据
        model_path = self._resolve_model(model)
        vocab_path = find_vocab(model_path)
        if vocab_path is None:
            raise RuntimeError("未找到 model_vocabulary.json，请放入 " + _dir_of(model_path))
        tag_names, num_tags = load_vocab(vocab_path)

        calib_path = find_calibration(model_path)
        cal_table = best_thr = None
        if calib_path is not None:
            cal_table, best_thr = load_calibration(calib_path)

        # 2. 预处理 + 推理
        x = self._preprocess(image)
        session = get_session(model_path)
        input_name = session.get_inputs()[0].name
        logits = np.asarray(session.run(None, {input_name: x})[0],
                            dtype=np.float32).reshape(-1)

        n = min(num_tags, logits.shape[0])
        tag_names = tag_names[:n]
        logits = logits[:n]

        # 3. logits -> sigmoid -> 校准查表
        probs = 1.0 / (1.0 + np.exp(-logits))                 # 0~1
        if cal_table is not None:
            bins = np.clip((probs * NUM_BINS).astype(np.int64), 0, NUM_BINS - 1)
            cal_rows = np.asarray(cal_table[:n], dtype=np.float32)
            cal_probs = cal_rows[np.arange(n), bins]
        else:
            cal_probs = probs

        # 4. 阈值过滤（角色类标签用 character_threshold）
        exclude = self._parse_exclude(exclude_tags)
        is_char = self._character_mask(tag_names, character_tags)
        if use_best_threshold and best_thr is not None:
            base_thr = np.asarray(best_thr[:n], dtype=np.float32)
        else:
            base_thr = np.full(n, threshold, dtype=np.float32)
        thr = np.where(is_char, character_threshold, base_thr)

        kept = []
        for i, (name, cp) in enumerate(zip(tag_names, cal_probs)):
            if not name or cp < thr[i]:
                continue
            key = name.replace("_", " ").strip().lower()
            if key in exclude or name.strip().lower() in exclude:
                continue
            kept.append((name, float(cp)))

        # 5. 排序
        if sort_by == "confidence":
            kept.sort(key=lambda kv: kv[1], reverse=True)
        elif sort_by == "alphabetical":
            kept.sort(key=lambda kv: kv[0].lower())

        # 6. 输出
        def fmt(name):
            return name.replace("_", tag_replacement) if tag_replacement else name

        tags = ", ".join(fmt(name) for name, _ in kept)
        tags_scores = ", ".join(f"{fmt(name)} ({cp:.3f})" for name, cp in kept)
        return (tags, tags_scores, len(kept), [cp for _, cp in kept])

    # ------------------------------------------------------------ 内部

    def _resolve_model(self, model):
        if model and os.path.isfile(model):
            return model
        models = list_models()
        if not models:
            raise RuntimeError(
                "未找到任何 .onnx 模型。请把 cl_tagger_v2 的模型文件放入 "
                f"{ensure_model_dir()} 目录（model.onnx + model.onnx.data + "
                "model_vocabulary.json + model_tag_metrics.npz）。")
        if model and not model.startswith("("):
            for m in models:
                if os.path.basename(m) == model:
                    return m
        return models[0]

    def _parse_exclude(self, text):
        out = set()
        for part in (text or "").split(","):
            part = part.strip().lower()
            if part:
                out.add(part)
        return out

    def _character_mask(self, tag_names, character_tags):
        """返回布尔数组：判定每个标签是否为角色类。

        判定规则（cl_tagger_v2 词表无角色分类列，用启发式 + 自定义名单）：
          1. 标签在 character_tags 自定义名单中（精确匹配，忽略大小写与下划线）
          2. danbooru 角色名消歧义惯例「名字 (作品名)」形式，如
             "kagamine rin (len)"、"asuka langley soryu (evangelion)"
        """
        custom = set()
        for part in (character_tags or "").split(","):
            part = part.strip().lower().replace(" ", "_")
            if part:
                custom.add(part)

        mask = np.zeros(len(tag_names), dtype=bool)
        for i, name in enumerate(tag_names):
            low = name.lower().replace(" ", "_")
            if low in custom:
                mask[i] = True
                continue
            # 快速过滤后再走正则：避免 10 万次正则匹配
            if "(" not in name:
                continue
            m = _CHAR_RE.match(name)
            if m and len(m.group(1).strip()) >= 2:
                mask[i] = True
        return mask

    def _preprocess(self, image):
        """IMAGE (B,H,W,C, 0~1) -> (1,3,384,384) float32 numpy。

        SigLIP 预处理：resize 384x384，归一化 mean=std=0.5。
        """
        img = image[0].detach().cpu().numpy()
        pil = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
        pil = pil.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        x = arr.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)


def _dir_of(model_path):
    return os.path.dirname(model_path)


# ---------------------------------------------------------------- 注册

NODE_CLASS_MAPPINGS["CLTaggerV2"] = CLTaggerV2
NODE_DISPLAY_NAME_MAPPINGS["CLTaggerV2"] = "CL Tagger v2 (标签反推)"
